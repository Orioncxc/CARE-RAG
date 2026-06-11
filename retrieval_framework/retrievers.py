from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from heapq import nlargest
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    rank: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def beir_doc_text(doc: Dict[str, Any], include_title: bool = True) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or "").strip()
    if include_title and title:
        return f"{title}\n{text}" if text else title
    return text


def corpus_to_documents(
    corpus: Dict[str, Dict[str, Any]],
    include_title: bool = True,
) -> List[Document]:
    return [
        Document(
            doc_id=str(doc_id),
            text=beir_doc_text(doc, include_title=include_title),
            metadata={"is_adv": False, "provenance": "indexed_corpus"},
        )
        for doc_id, doc in corpus.items()
    ]


def resolve_device(device: Optional[str] = None, gpu_id: int = 0) -> torch.device:
    if torch is None:
        raise ModuleNotFoundError("Dense retrieval requires PyTorch, but torch is not installed.")
    if device and device != "auto":
        if device.startswith("cuda") and not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def progress_range(start: int, stop: int, step: int, desc: str):
    iterator = range(start, stop, step)
    if tqdm is None:
        return iterator
    total = math.ceil(max(0, stop - start) / step)
    return tqdm(iterator, total=total, desc=desc, unit="batch")


def progress_iter(items: Iterable[Any], total: int, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)


def safe_dot_scores(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores = left @ right
    return np.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)


def cluster_tokens(text: str) -> set:
    return set(re.findall(r"(?u)\b\w+\b", text.lower()))


def jaccard_similarity(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


class BaseRetriever:
    name = "base"

    def index(self, documents: Sequence[Document]) -> None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    """Dense retrieval with one embedding model for queries and documents."""

    name = "dense"

    def __init__(
        self,
        model_code: str = "contriever",
        score_function: str = "cos_sim",
        batch_size: int = 64,
        max_length: int = 128,
        device: str = "auto",
        gpu_id: int = 0,
        cache_dir: Optional[str] = None,
        cache_name: Optional[str] = None,
    ) -> None:
        if score_function not in {"cos_sim", "dot"}:
            raise ValueError("score_function must be 'cos_sim' or 'dot'")
        self.model_code = model_code
        self.score_function = score_function
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = resolve_device(device=device, gpu_id=gpu_id)
        self.cache_dir = cache_dir
        self.cache_name = cache_name

        self.model = None
        self.c_model = None
        self.tokenizer = None
        self.get_emb = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.doc_embs: Optional[np.ndarray] = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from src.utils import load_models

        model, c_model, tokenizer, get_emb = load_models(self.model_code)
        model.eval()
        c_model.eval()
        model.to(self.device)
        c_model.to(self.device)
        self.model = model
        self.c_model = c_model
        self.tokenizer = tokenizer
        self.get_emb = get_emb

    def _cache_paths(self) -> Optional[Dict[str, str]]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.cache_name:
            name = self.cache_name
        else:
            sample_ids = self.doc_ids[:100] + self.doc_ids[-100:]
            digest = hashlib.sha1(
                ("\n".join(sample_ids) + f"\n{len(self.doc_ids)}").encode("utf-8")
            ).hexdigest()[:12]
            name = f"dense_{self.model_code}_{len(self.doc_ids)}_{digest}"
        prefix = os.path.join(self.cache_dir, name)
        return {"emb": f"{prefix}.npy", "meta": f"{prefix}.json"}

    def _load_cache(self) -> bool:
        paths = self._cache_paths()
        if not paths or not (os.path.exists(paths["emb"]) and os.path.exists(paths["meta"])):
            return False
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_code") != self.model_code:
            return False
        if meta.get("doc_ids") != self.doc_ids:
            return False
        self.doc_embs = np.load(paths["emb"]).astype(np.float32)
        if self.score_function == "cos_sim":
            self.doc_embs = self._normalize(self.doc_embs)
        return True

    def _save_cache(self, raw_embeddings: np.ndarray) -> None:
        paths = self._cache_paths()
        if not paths:
            return
        np.save(paths["emb"], raw_embeddings.astype(np.float32))
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_code": self.model_code,
                    "doc_ids": self.doc_ids,
                    "count": len(self.doc_ids),
                    "dim": int(raw_embeddings.shape[1]) if raw_embeddings.ndim == 2 else 0,
                },
                f,
            )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return values / denom

    def _encode_texts(
        self,
        texts: Sequence[str],
        encode_docs: bool,
        show_progress: bool = False,
        desc: Optional[str] = None,
    ) -> np.ndarray:
        self._ensure_model()
        encoder = self.c_model if encode_docs else self.model
        all_embs: List[np.ndarray] = []
        starts = range(0, len(texts), self.batch_size)
        if show_progress:
            starts = progress_range(
                0,
                len(texts),
                self.batch_size,
                desc or "Encoding dense documents",
            )
        for start in starts:
            batch = list(texts[start : start + self.batch_size])
            model_inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            with torch.no_grad():
                embs = self.get_emb(encoder, model_inputs)
            all_embs.append(embs.detach().float().cpu().numpy())
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(all_embs).astype(np.float32)

    def index(self, documents: Sequence[Document]) -> None:
        self.doc_ids = [doc.doc_id for doc in documents]
        self.doc_texts = [doc.text for doc in documents]
        self.doc_metadata = [dict(doc.metadata) for doc in documents]
        if self._load_cache():
            return
        raw_embeddings = self._encode_texts(
            self.doc_texts,
            encode_docs=True,
            show_progress=True,
            desc=f"Dense indexing ({self.model_code})",
        )
        self._save_cache(raw_embeddings)
        self.doc_embs = raw_embeddings
        if self.score_function == "cos_sim":
            self.doc_embs = self._normalize(self.doc_embs)

    def _score(self, query: str, doc_embs: np.ndarray) -> np.ndarray:
        query_emb = self._encode_texts([query], encode_docs=False)
        if self.score_function == "cos_sim":
            query_emb = self._normalize(query_emb)
        return safe_dot_scores(doc_embs, query_emb[0])

    def _base_results(self, query: str, top_k: int) -> List[SearchResult]:
        if self.doc_embs is None:
            raise RuntimeError("DenseRetriever.index() must be called before search().")
        if len(self.doc_ids) == 0 or top_k <= 0:
            return []
        scores = self._score(query, self.doc_embs)
        keep = min(top_k, len(scores))
        candidate_idx = np.argpartition(-scores, keep - 1)[:keep]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.doc_texts[idx],
                score=float(scores[idx]),
                rank=rank,
                source=self.name,
                metadata=dict(self.doc_metadata[idx]),
            )
            for rank, idx in enumerate(candidate_idx, start=1)
        ]

    def _extra_results(
        self,
        query: str,
        extra_docs: Optional[Sequence[Document]],
    ) -> List[SearchResult]:
        if not extra_docs:
            return []
        extra_embs = self._encode_texts([doc.text for doc in extra_docs], encode_docs=True)
        if self.score_function == "cos_sim":
            extra_embs = self._normalize(extra_embs)
        scores = self._score(query, extra_embs)
        order = np.argsort(-scores)
        return [
            SearchResult(
                doc_id=extra_docs[idx].doc_id,
                text=extra_docs[idx].text,
                score=float(scores[idx]),
                rank=rank,
                source=self.name,
                metadata=dict(extra_docs[idx].metadata),
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        candidates = self._base_results(query, top_k)
        candidates.extend(self._extra_results(query, extra_docs))
        candidates.sort(key=lambda item: item.score, reverse=True)
        results = candidates[:top_k]
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results


class PrecomputedDenseRetriever(BaseRetriever):
    """Dense retrieval backed by the original precomputed BEIR result files."""

    name = "dense_precomputed"

    def __init__(
        self,
        results_path: str,
        model_code: str = "contriever",
        score_function: str = "dot",
        batch_size: int = 64,
        max_length: int = 128,
        device: str = "auto",
        gpu_id: int = 0,
    ) -> None:
        if score_function not in {"cos_sim", "dot"}:
            raise ValueError("score_function must be 'cos_sim' or 'dot'")
        self.results_path = results_path
        self.model_code = model_code
        self.score_function = score_function
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_name = device
        self.gpu_id = gpu_id

        self.precomputed_results: Dict[str, Dict[str, float]] = {}
        self.doc_by_id: Dict[str, Document] = {}

        self.device = None
        self.model = None
        self.c_model = None
        self.tokenizer = None
        self.get_emb = None

    def _load_results(self) -> None:
        if self.precomputed_results:
            return
        with open(self.results_path, "r", encoding="utf-8") as f:
            self.precomputed_results = json.load(f)

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from src.utils import load_models

        self.device = resolve_device(device=self.device_name, gpu_id=self.gpu_id)
        model, c_model, tokenizer, get_emb = load_models(self.model_code)
        model.eval()
        c_model.eval()
        model.to(self.device)
        c_model.to(self.device)
        self.model = model
        self.c_model = c_model
        self.tokenizer = tokenizer
        self.get_emb = get_emb

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return values / denom

    def _encode_texts(self, texts: Sequence[str], encode_docs: bool) -> np.ndarray:
        self._ensure_model()
        encoder = self.c_model if encode_docs else self.model
        all_embs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            model_inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            with torch.no_grad():
                embs = self.get_emb(encoder, model_inputs)
            all_embs.append(embs.detach().float().cpu().numpy())
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(all_embs).astype(np.float32)

    def index(self, documents: Sequence[Document]) -> None:
        self.doc_by_id = {doc.doc_id: doc for doc in documents}
        self._load_results()

    def _base_results(self, query_id: str, top_k: int) -> List[SearchResult]:
        if query_id not in self.precomputed_results:
            raise KeyError(f"{query_id} not found in {self.results_path}")
        ranked_items = list(self.precomputed_results[query_id].items())[:top_k]
        results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(ranked_items, start=1):
            doc = self.doc_by_id.get(doc_id)
            if doc is None:
                continue
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    rank=rank,
                    source=self.name,
                    metadata=dict(doc.metadata),
                )
            )
        return results

    def _extra_results(
        self,
        query: str,
        extra_docs: Optional[Sequence[Document]],
    ) -> List[SearchResult]:
        if not extra_docs:
            return []
        extra_embs = self._encode_texts([doc.text for doc in extra_docs], encode_docs=True)
        query_emb = self._encode_texts([query], encode_docs=False)
        if self.score_function == "cos_sim":
            extra_embs = self._normalize(extra_embs)
            query_emb = self._normalize(query_emb)
        scores = safe_dot_scores(extra_embs, query_emb[0])
        order = np.argsort(-scores)
        return [
            SearchResult(
                doc_id=extra_docs[idx].doc_id,
                text=extra_docs[idx].text,
                score=float(scores[idx]),
                rank=rank,
                source=self.name,
                metadata=dict(extra_docs[idx].metadata),
            )
            for rank, idx in enumerate(order, start=1)
        ]

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        if query_id is None:
            raise ValueError("PrecomputedDenseRetriever.search() requires query_id.")
        candidates = self._base_results(query_id, top_k)
        candidates.extend(self._extra_results(query, extra_docs))
        candidates.sort(key=lambda item: item.score, reverse=True)
        results = candidates[:top_k]
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results


class BM25Retriever(BaseRetriever):
    """Pure-Python BM25 retriever for sparse keyword matching."""

    name = "bm25"
    token_pattern = re.compile(r"(?u)\b\w+\b")

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.index_postings: Dict[str, List[tuple]] = defaultdict(list)
        self.df: Dict[str, int] = {}
        self.num_docs = 0

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return cls.token_pattern.findall(text.lower())

    def index(self, documents: Sequence[Document]) -> None:
        self.doc_ids = [doc.doc_id for doc in documents]
        self.doc_texts = [doc.text for doc in documents]
        self.doc_metadata = [dict(doc.metadata) for doc in documents]
        self.doc_len = []
        self.index_postings = defaultdict(list)
        self.df = {}
        self.num_docs = len(documents)

        for idx, doc in progress_iter(
            enumerate(documents),
            total=len(documents),
            desc="BM25 indexing",
        ):
            counts = Counter(self.tokenize(doc.text))
            length = sum(counts.values())
            self.doc_len.append(length)
            for term, tf in counts.items():
                self.index_postings[term].append((idx, tf))
        self.df = {term: len(postings) for term, postings in self.index_postings.items()}
        self.avgdl = sum(self.doc_len) / max(1, self.num_docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def _term_score(self, term: str, tf: int, doc_len: int) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
        return self._idf(term) * (tf * (self.k1 + 1.0)) / max(denom, 1e-9)

    def _score_extra_doc(self, query_counts: Counter, doc: Document) -> float:
        counts = Counter(self.tokenize(doc.text))
        doc_len = sum(counts.values())
        score = 0.0
        for term, qtf in query_counts.items():
            tf = counts.get(term, 0)
            if tf:
                score += qtf * self._term_score(term, tf, doc_len)
        return score

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        if self.num_docs == 0:
            raise RuntimeError("BM25Retriever.index() must be called before search().")
        query_counts = Counter(self.tokenize(query))
        scores: Dict[int, float] = defaultdict(float)
        for term, qtf in query_counts.items():
            for doc_idx, tf in self.index_postings.get(term, []):
                scores[doc_idx] += qtf * self._term_score(term, tf, self.doc_len[doc_idx])

        base_keep = nlargest(top_k, scores.items(), key=lambda item: item[1])
        candidates: List[SearchResult] = [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.doc_texts[idx],
                score=float(score),
                rank=rank,
                source=self.name,
                metadata=dict(self.doc_metadata[idx]),
            )
            for rank, (idx, score) in enumerate(base_keep, start=1)
        ]

        if extra_docs:
            for doc in extra_docs:
                candidates.append(
                    SearchResult(
                        doc_id=doc.doc_id,
                        text=doc.text,
                        score=float(self._score_extra_doc(query_counts, doc)),
                        rank=0,
                        source=self.name,
                        metadata=dict(doc.metadata),
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        results = candidates[:top_k]
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results


class RRFHybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over dense and BM25 rankings."""

    name = "rrf"

    def __init__(
        self,
        dense: BaseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        candidate_depth: int = 100,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def index(self, documents: Sequence[Document]) -> None:
        self.dense.index(documents)
        self.bm25.index(documents)

    def _add_channel(
        self,
        fused: Dict[str, Dict[str, Any]],
        results: Iterable[SearchResult],
        channel: str,
        weight: float,
    ) -> None:
        for result in results:
            entry = fused.setdefault(
                result.doc_id,
                {
                    "doc_id": result.doc_id,
                    "text": result.text,
                    "score": 0.0,
                    "metadata": dict(result.metadata),
                    "channels": {},
                },
            )
            entry["score"] += weight / (self.rrf_k + result.rank)
            entry["channels"][channel] = {
                "rank": result.rank,
                "score": result.score,
                "source": result.source,
            }

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        depth = max(top_k, self.candidate_depth)
        dense_results = self.dense.search(
            query,
            depth,
            extra_docs=extra_docs,
            query_id=query_id,
        )
        bm25_results = self.bm25.search(
            query,
            depth,
            extra_docs=extra_docs,
            query_id=query_id,
        )
        fused: Dict[str, Dict[str, Any]] = {}
        self._add_channel(fused, dense_results, "dense", self.dense_weight)
        self._add_channel(fused, bm25_results, "bm25", self.bm25_weight)

        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        return [
            SearchResult(
                doc_id=item["doc_id"],
                text=item["text"],
                score=float(item["score"]),
                rank=rank,
                source=self.name,
                metadata={**item["metadata"], "channels": item["channels"]},
            )
            for rank, item in enumerate(ranked, start=1)
        ]


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in scores}
    return {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores.items()
    }


class NormalizedHybridRetriever(BaseRetriever):
    """Paper-style dense/BM25 hybrid with per-query min-max score normalization."""

    name = "normalized_hybrid"

    def __init__(
        self,
        dense: BaseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.5,
        candidate_depth: int = 100,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha
        self.candidate_depth = candidate_depth

    def index(self, documents: Sequence[Document]) -> None:
        self.dense.index(documents)
        self.bm25.index(documents)

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        depth = max(top_k, self.candidate_depth)
        dense_results = self.dense.search(
            query,
            depth,
            extra_docs=extra_docs,
            query_id=query_id,
        )
        bm25_results = self.bm25.search(
            query,
            depth,
            extra_docs=extra_docs,
            query_id=query_id,
        )

        dense_scores = {result.doc_id: result.score for result in dense_results}
        bm25_scores = {result.doc_id: result.score for result in bm25_results}
        dense_norm = min_max_normalize(dense_scores)
        bm25_norm = min_max_normalize(bm25_scores)

        by_doc: Dict[str, Dict[str, Any]] = {}
        for result in dense_results:
            by_doc[result.doc_id] = {
                "doc_id": result.doc_id,
                "text": result.text,
                "metadata": dict(result.metadata),
                "dense_raw": result.score,
                "bm25_raw": 0.0,
                "dense_norm": dense_norm.get(result.doc_id, 0.0),
                "bm25_norm": 0.0,
            }

        for result in bm25_results:
            entry = by_doc.setdefault(
                result.doc_id,
                {
                    "doc_id": result.doc_id,
                    "text": result.text,
                    "metadata": dict(result.metadata),
                    "dense_raw": 0.0,
                    "bm25_raw": 0.0,
                    "dense_norm": 0.0,
                    "bm25_norm": 0.0,
                },
            )
            entry["bm25_raw"] = result.score
            entry["bm25_norm"] = bm25_norm.get(result.doc_id, 0.0)

        fused = []
        for entry in by_doc.values():
            entry["score"] = (
                self.alpha * entry["dense_norm"]
                + (1.0 - self.alpha) * entry["bm25_norm"]
            )
            fused.append(entry)

        ranked = sorted(fused, key=lambda item: item["score"], reverse=True)[:top_k]
        return [
            SearchResult(
                doc_id=item["doc_id"],
                text=item["text"],
                score=float(item["score"]),
                rank=rank,
                source=self.name,
                metadata={
                    **item["metadata"],
                    "dense_raw": item["dense_raw"],
                    "bm25_raw": item["bm25_raw"],
                    "dense_norm": item["dense_norm"],
                    "bm25_norm": item["bm25_norm"],
                    "alpha": self.alpha,
                },
            )
            for rank, item in enumerate(ranked, start=1)
        ]


class SecureEnsembleRetriever(BaseRetriever):
    """Multi-retriever ensemble with security-oriented fusion and cluster caps."""

    name = "secure_ensemble"

    def __init__(
        self,
        channels: Sequence[Dict[str, Any]],
        fusion: str = "consensus_rrf",
        rrf_k: int = 60,
        candidate_depth: int = 100,
        support_bonus: float = 0.02,
        single_penalty: float = 0.05,
        missing_penalty: float = 0.02,
        min_support: int = 1,
        cluster_cap: int = 0,
        cluster_jaccard_threshold: float = 0.72,
    ) -> None:
        if fusion not in {"rrf", "consensus_rrf", "robust_rank"}:
            raise ValueError("fusion must be 'rrf', 'consensus_rrf', or 'robust_rank'.")
        if not channels:
            raise ValueError("SecureEnsembleRetriever requires at least one channel.")
        self.channels = [
            {
                "name": channel["name"],
                "weight": float(channel.get("weight", 1.0)),
                "retriever": build_channel_retriever(channel),
            }
            for channel in channels
            if channel.get("enabled", True)
        ]
        if not self.channels:
            raise ValueError("No enabled channels configured for SecureEnsembleRetriever.")
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth
        self.support_bonus = support_bonus
        self.single_penalty = single_penalty
        self.missing_penalty = missing_penalty
        self.min_support = min_support
        self.cluster_cap = cluster_cap
        self.cluster_jaccard_threshold = cluster_jaccard_threshold

    def index(self, documents: Sequence[Document]) -> None:
        for channel in self.channels:
            print(f"Indexing channel={channel['name']}...")
            channel["retriever"].index(documents)

    def _add_channel_results(
        self,
        fused: Dict[str, Dict[str, Any]],
        channel_name: str,
        channel_weight: float,
        results: Iterable[SearchResult],
    ) -> None:
        for result in results:
            entry = fused.setdefault(
                result.doc_id,
                {
                    "doc_id": result.doc_id,
                    "text": result.text,
                    "metadata": dict(result.metadata),
                    "rrf_score": 0.0,
                    "channels": {},
                },
            )
            entry["rrf_score"] += channel_weight / (self.rrf_k + result.rank)
            entry["channels"][channel_name] = {
                "rank": result.rank,
                "score": result.score,
                "source": result.source,
                "weight": channel_weight,
            }

    def _score_entry(self, entry: Dict[str, Any]) -> float:
        support_count = len(entry["channels"])
        missing_count = len(self.channels) - support_count
        score = entry["rrf_score"]
        if self.fusion == "consensus_rrf":
            score += self.support_bonus * support_count
            if support_count == 1:
                score -= self.single_penalty
        elif self.fusion == "robust_rank":
            score += self.support_bonus * support_count
            score -= self.missing_penalty * missing_count
            if support_count == 1:
                score -= self.single_penalty
        return score

    def _apply_cluster_cap(self, ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if self.cluster_cap <= 0:
            return ranked[:top_k]

        accepted: List[Dict[str, Any]] = []
        clusters: List[Dict[str, Any]] = []
        for entry in ranked:
            tokens = cluster_tokens(entry["text"])
            matched_cluster = None
            for cluster in clusters:
                if jaccard_similarity(tokens, cluster["tokens"]) >= self.cluster_jaccard_threshold:
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                matched_cluster = {
                    "cluster_id": len(clusters),
                    "tokens": tokens,
                    "count": 0,
                }
                clusters.append(matched_cluster)

            entry["cluster_id"] = matched_cluster["cluster_id"]
            if matched_cluster["count"] >= self.cluster_cap:
                entry["cluster_filtered"] = True
                continue

            matched_cluster["count"] += 1
            entry["cluster_filtered"] = False
            accepted.append(entry)
            if len(accepted) >= top_k:
                break
        return accepted

    def search(
        self,
        query: str,
        top_k: int,
        extra_docs: Optional[Sequence[Document]] = None,
        query_id: Optional[str] = None,
    ) -> List[SearchResult]:
        depth = max(top_k, self.candidate_depth)
        fused: Dict[str, Dict[str, Any]] = {}
        for channel in self.channels:
            results = channel["retriever"].search(
                query,
                depth,
                extra_docs=extra_docs,
                query_id=query_id,
            )
            self._add_channel_results(
                fused,
                channel_name=channel["name"],
                channel_weight=channel["weight"],
                results=results,
            )

        candidates = []
        for entry in fused.values():
            support_count = len(entry["channels"])
            if support_count < self.min_support:
                continue
            entry["support_count"] = support_count
            entry["missing_count"] = len(self.channels) - support_count
            entry["score"] = self._score_entry(entry)
            candidates.append(entry)

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        capped = self._apply_cluster_cap(ranked, top_k)
        return [
            SearchResult(
                doc_id=item["doc_id"],
                text=item["text"],
                score=float(item["score"]),
                rank=rank,
                source=self.name,
                metadata={
                    **item["metadata"],
                    "fusion": self.fusion,
                    "rrf_score": item["rrf_score"],
                    "support_count": item["support_count"],
                    "missing_count": item["missing_count"],
                    "channels": item["channels"],
                    "cluster_id": item.get("cluster_id"),
                    "cluster_filtered": item.get("cluster_filtered", False),
                    "num_channels": len(self.channels),
                },
            )
            for rank, item in enumerate(capped, start=1)
        ]


def build_dense_retriever(config: Dict[str, Any]) -> BaseRetriever:
    dense_config = dict(config.get("dense", {}))
    use_precomputed = bool(dense_config.pop("use_precomputed", False))
    results_path = dense_config.pop("precomputed_results_path", None)
    if use_precomputed:
        if not results_path:
            raise ValueError("dense.use_precomputed requires dense.precomputed_results_path.")
        dense_config.pop("cache_dir", None)
        dense_config.pop("cache_name", None)
        return PrecomputedDenseRetriever(results_path=results_path, **dense_config)
    return DenseRetriever(**dense_config)


def build_channel_retriever(config: Dict[str, Any]) -> BaseRetriever:
    retriever_type = config.get("type", "dense").lower()
    if retriever_type == "dense":
        return build_dense_retriever(config)
    if retriever_type == "bm25":
        return BM25Retriever(**config.get("bm25", {}))
    raise ValueError(f"Unsupported secure ensemble channel type: {retriever_type}")


def build_retriever(config: Dict[str, Any]) -> BaseRetriever:
    retriever_type = config.get("type", "dense").lower()
    if retriever_type == "dense":
        return build_dense_retriever(config)
    if retriever_type == "bm25":
        return BM25Retriever(**config.get("bm25", {}))
    if retriever_type in {"rrf", "rrf_hybrid"}:
        dense = build_dense_retriever(config)
        bm25 = BM25Retriever(**config.get("bm25", {}))
        return RRFHybridRetriever(
            dense=dense,
            bm25=bm25,
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_depth=int(config.get("candidate_depth", 100)),
            dense_weight=float(config.get("dense_weight", 1.0)),
            bm25_weight=float(config.get("bm25_weight", 1.0)),
        )
    if retriever_type in {"hybrid", "normalized_hybrid", "paper_hybrid"}:
        dense = build_dense_retriever(config)
        bm25 = BM25Retriever(**config.get("bm25", {}))
        return NormalizedHybridRetriever(
            dense=dense,
            bm25=bm25,
            alpha=float(config.get("alpha", 0.5)),
            candidate_depth=int(config.get("candidate_depth", 100)),
        )
    if retriever_type in {"secure_ensemble", "consensus_ensemble", "robust_ensemble"}:
        return SecureEnsembleRetriever(
            channels=config.get("channels", []),
            fusion=config.get("fusion", "consensus_rrf"),
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_depth=int(config.get("candidate_depth", 100)),
            support_bonus=float(config.get("support_bonus", 0.02)),
            single_penalty=float(config.get("single_penalty", 0.05)),
            missing_penalty=float(config.get("missing_penalty", 0.02)),
            min_support=int(config.get("min_support", 1)),
            cluster_cap=int(config.get("cluster_cap", 0)),
            cluster_jaccard_threshold=float(config.get("cluster_jaccard_threshold", 0.72)),
        )
    raise ValueError(f"Unknown retriever type: {retriever_type}")

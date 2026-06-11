from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .consequence_kb import ConsequenceKB
    from .retrievers import SearchResult, cluster_tokens, jaccard_similarity
except ImportError:  # pragma: no cover - used when running as a plain script
    from retrieval_framework.consequence_kb import ConsequenceKB
    from retrieval_framework.retrievers import SearchResult, cluster_tokens, jaccard_similarity


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z'&.-]+|[A-Z]{2,})"
    r"(?:\s+(?:of|the|and|for|in|de|da|van|von|[A-Z][A-Za-z'&.-]+|[A-Z]{2,})){0,5}"
)
NUMBER_PATTERN = re.compile(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b")
NUMBER_PHRASE_PATTERN = re.compile(
    r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s+(?:percent|percentage|years?|months?|weeks?|days?|hours?|minutes?|"
    r"seconds?|miles?|kilometers?|metres?|meters?|feet|ft|inches?|episodes?|"
    r"seasons?|points?|runs?|goals?|people|members|states?|countries|cities|"
    r"dollars?|usd|pounds?|kg|kilograms?|degrees?)){0,3}\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
MONTH_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CUE_PATTERN = re.compile(
    r"\b(?:answer is|the answer is|became|called|known as|"
    r"named|located in|located at|written by|recorded by|sung by|played by|"
    r"voiced by|created by|founded by|owned by|produced by|directed by)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([A-Za-z0-9][A-Za-z0-9'&./-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'&./-]*){0,9})",
    re.IGNORECASE,
)
WHO_SUBJECT_CUE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,4})\s+"
    r"(?:recorded|sang|sings|played|plays|wrote|writes|voiced|created|founded|"
    r"directed|produced|composed|invented|introduced|proposed)\b"
)
WHERE_CUE_PATTERN = re.compile(
    r"(?i:\b(?:located in|located at|arrived at|arrived in|went to|go to|goes to|"
    r"left for|departed for|sails on to|sailed to|travelled to|traveled to|"
    r"destination,\s*|destination is|settled in|landed in|land on|took him to))\s+"
    r"(?:the\s+|a\s+|an\s+|his\s+|her\s+|their\s+)?"
    r"([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,4})"
)
WHAT_CUE_PATTERN = re.compile(
    r"\b(?:is called|are called|was called|were called|is known as|are known as|"
    r"was known as|were known as|is named|was named|refers to|is termed|"
    r"are termed|is defined as|means|is)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([A-Za-z0-9][A-Za-z0-9'&./-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'&./-]*){0,7})",
    re.IGNORECASE,
)

COMMON_ENTITY_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
    "which",
}

QUERY_FOCUS_STOPWORDS = COMMON_ENTITY_WORDS | {
    "about",
    "after",
    "all",
    "also",
    "am",
    "be",
    "been",
    "before",
    "between",
    "can",
    "did",
    "do",
    "does",
    "during",
    "had",
    "has",
    "have",
    "how",
    "into",
    "it",
    "its",
    "may",
    "not",
    "that",
    "their",
    "this",
    "through",
    "with",
}

LOW_VALUE_CANDIDATE_PREFIXES = {
    "according",
    "answer",
    "context",
    "contexts",
    "document",
    "documents",
    "evidence",
    "following",
    "information",
    "passage",
    "question",
    "text",
}

LOW_VALUE_SINGLE_ENTITIES = {
    "american",
    "british",
    "canadian",
    "catholic",
    "european",
    "famous",
    "her",
    "his",
    "public",
    "roman",
    "trojan",
    "trojans",
    "throughout",
}

CUE_STOP_TOKENS = {
    "and",
    "but",
    "because",
    "although",
    "while",
    "whereas",
    "with",
    "from",
    "for",
    "that",
    "which",
    "who",
    "when",
    "where",
}


@dataclass
class AnswerMention:
    key: str
    label: str
    source: str
    kind: str = "heuristic"
    score: float = 0.0


@dataclass
class CandidateEntry:
    result: SearchResult
    original_rank: int
    original_score: float
    cluster_id: int = -1
    hardening_score: float = 0.0
    query_echo_exact_prefix: bool = False
    query_echo_overlap: float = 0.0
    query_echo_ngram_overlap: float = 0.0
    query_echo_novelty_ratio: float = 1.0
    query_echo_novelty_token_count: int = 0
    query_echo_lexical_penalty: float = 0.0
    query_echo_penalty: float = 0.0
    injection_risk_penalty: float = 0.0
    injection_risk_signals: List[str] = None
    topic_grounding_overlap: int = 0
    topic_grounding_query_coverage: float = 0.0
    topic_grounding_content_token_count: int = 0
    topic_grounding_low: bool = False
    topic_grounding_penalty: float = 0.0
    rank_guard_blocked: bool = False
    top1_dominance_adjusted: bool = False
    top1_dominance_penalty: float = 0.0
    margin_gate_adjusted: bool = False
    margin_gate_penalty: float = 0.0
    margin_gate_bonus: float = 0.0
    margin_gate_supplement_promoted: bool = False
    heuristic_mentions: List[AnswerMention] = None
    diagnostic_mentions: List[AnswerMention] = None
    reference_mentions: List[AnswerMention] = None

    def __post_init__(self) -> None:
        if self.heuristic_mentions is None:
            self.heuristic_mentions = []
        if self.diagnostic_mentions is None:
            self.diagnostic_mentions = []
        if self.reference_mentions is None:
            self.reference_mentions = []
        if self.injection_risk_signals is None:
            self.injection_risk_signals = []


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_key(value: Any) -> str:
    return normalize_text(value)


def coerce_answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def infer_query_type(query: str) -> str:
    query_l = query.lower().strip()
    first_word = query_l.split(" ", 1)[0] if query_l else ""
    if first_word in {"is", "are", "was", "were", "do", "does", "did", "can"}:
        return "boolean"
    if (
        query_l.startswith("how many")
        or query_l.startswith("how much")
        or " number " in f" {query_l} "
        or " percentage " in f" {query_l} "
        or " percent " in f" {query_l} "
    ):
        return "number"
    if query_l.startswith("when") or "what year" in query_l or " date " in f" {query_l} ":
        return "date"
    if query_l.startswith("who") or query_l.startswith("where") or query_l.startswith("which"):
        return "entity"
    return "mixed"


def query_content_tokens(query: str) -> Set[str]:
    return {
        token
        for token in normalize_text(query).split()
        if len(token) > 2 and token not in QUERY_FOCUS_STOPWORDS
    }


def split_sentences(text: str) -> List[str]:
    sentences = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text or "")
    ]
    return [sentence for sentence in sentences if sentence]


def sentence_focus_score(sentence: str, content_tokens: Set[str]) -> int:
    if not content_tokens:
        return 0
    sentence_tokens = set(normalize_text(sentence).split())
    return len(sentence_tokens & content_tokens)


def focused_text_spans(
    query: str,
    text: str,
    focused_sentence_count: int = 4,
    use_query_focus: bool = True,
) -> List[Tuple[str, str]]:
    sentences = split_sentences(text)
    if not sentences:
        return [("full", text)]
    if not use_query_focus:
        return [("sentence", sentence) for sentence in sentences[:focused_sentence_count]]

    content_tokens = query_content_tokens(query)
    scored = [
        (sentence_focus_score(sentence, content_tokens), idx, sentence)
        for idx, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [
        (score, idx, sentence)
        for score, idx, sentence in scored
        if score > 0
    ][:focused_sentence_count]
    if not selected:
        selected = scored[:focused_sentence_count]
    selected.sort(key=lambda item: item[1])
    return [("focused_sentence", sentence) for _, _, sentence in selected]


def clean_candidate_label(label: str, max_words: int = 8) -> str:
    label = re.sub(r"(?<=[a-z0-9])\.(?=[A-Z])", ". ", label or "")
    label = re.sub(r"\s+", " ", label or "").strip(" \t\n\r,;:()[]{}\"'")
    if not label:
        return ""
    tokens = label.split()
    cleaned: List[str] = []
    for token in tokens:
        normalized = normalize_text(token)
        if normalized in CUE_STOP_TOKENS:
            break
        cleaned.append(token.strip(" \t\n\r,;:()[]{}\"'"))
        if len(cleaned) >= max_words:
            break
    label = " ".join(token for token in cleaned if token).strip()
    return label


def candidate_is_low_value(candidate: str, query_tokens: Set[str]) -> bool:
    key = answer_key(candidate)
    tokens = key.split()
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0] in LOW_VALUE_SINGLE_ENTITIES:
        return True
    if tokens[-1] == "s":
        return True
    if len(tokens) > 10:
        return True
    if tokens[0] in LOW_VALUE_CANDIDATE_PREFIXES:
        return True
    if len(tokens) >= 3:
        overlap = len(set(tokens) & query_tokens)
        if overlap / max(len(set(tokens)), 1) >= 0.5:
            return True
    if candidate_is_query_echo(candidate, query_tokens):
        return True
    if len(tokens) == 1 and tokens[0] in QUERY_FOCUS_STOPWORDS:
        return True
    return False


def candidate_is_sentence_like_answer(candidate: str) -> bool:
    tokens = answer_key(candidate).split()
    if len(tokens) <= 5:
        return False
    return any(
        marker in f" {' '.join(tokens)} "
        for marker in [
            " is ",
            " are ",
            " was ",
            " were ",
            " has ",
            " have ",
            " had ",
            " became ",
        ]
    )


def cue_patterns_for_query(query_type: str) -> List[Tuple[re.Pattern, str]]:
    if query_type == "entity":
        return [
            (WHO_SUBJECT_CUE_PATTERN, "who_subject_cue"),
            (WHERE_CUE_PATTERN, "where_cue"),
            (CUE_PATTERN, "relation_cue"),
        ]
    if query_type == "mixed":
        return [
            (WHAT_CUE_PATTERN, "what_cue"),
            (CUE_PATTERN, "relation_cue"),
        ]
    return []


def candidate_is_query_echo(candidate: str, query_tokens: Set[str]) -> bool:
    tokens = set(normalize_text(candidate).split())
    if not tokens:
        return True
    if tokens <= query_tokens:
        return True
    return bool(tokens) and all(token in COMMON_ENTITY_WORDS for token in tokens)


def unique_mentions(mentions: Iterable[AnswerMention]) -> List[AnswerMention]:
    seen = set()
    unique: List[AnswerMention] = []
    for mention in mentions:
        if not mention.key or mention.key in seen:
            continue
        seen.add(mention.key)
        unique.append(mention)
    return unique


def extract_heuristic_answers(
    query: str,
    text: str,
    max_mentions: int = 5,
    focused_sentence_count: int = 4,
    max_candidate_words: int = 8,
    use_query_focus: bool = True,
    use_answer_cues: bool = True,
) -> List[AnswerMention]:
    query_type = infer_query_type(query)
    query_tokens = set(normalize_text(query).split())
    if not use_query_focus and not use_answer_cues:
        mentions: List[AnswerMention] = []
        if query_type in {"number", "mixed"}:
            for match in NUMBER_PATTERN.finditer(text):
                label = match.group(0)
                mentions.append(
                    AnswerMention(key=answer_key(label), label=label, source="number")
                )
        if query_type in {"date", "mixed"}:
            for pattern, source in [(MONTH_PATTERN, "date"), (YEAR_PATTERN, "year")]:
                for match in pattern.finditer(text):
                    label = match.group(0)
                    mentions.append(
                        AnswerMention(key=answer_key(label), label=label, source=source)
                    )
        if query_type in {"entity", "mixed"}:
            for match in ENTITY_PATTERN.finditer(text):
                label = re.sub(r"\s+", " ", match.group(0)).strip()
                if len(label) < 2 or len(label) > 80:
                    continue
                if candidate_is_query_echo(label, query_tokens):
                    continue
                mentions.append(
                    AnswerMention(key=answer_key(label), label=label, source="entity")
                )
        return unique_mentions(mentions)[:max_mentions]

    spans = focused_text_spans(
        query,
        text,
        focused_sentence_count=focused_sentence_count,
        use_query_focus=use_query_focus,
    )
    mentions: List[AnswerMention] = []

    def add_candidate(label: str, source: str) -> None:
        cleaned = clean_candidate_label(label, max_words=max_candidate_words)
        if candidate_is_low_value(cleaned, query_tokens):
            return
        mentions.append(
            AnswerMention(key=answer_key(cleaned), label=cleaned, source=source)
        )

    for span_source, span in spans:
        if query_type == "boolean":
            normalized_span = f" {normalize_text(span)} "
            if re.search(r"\b(?:yes|true|correct)\b", normalized_span):
                mentions.append(AnswerMention(key="yes", label="yes", source="boolean"))
            if re.search(r"\b(?:no|false|incorrect|not|never)\b", normalized_span):
                mentions.append(AnswerMention(key="no", label="no", source="boolean"))

        if use_answer_cues:
            for pattern, cue_source in cue_patterns_for_query(query_type):
                for match in pattern.finditer(span):
                    add_candidate(match.group(1), f"{span_source}:{cue_source}")

        if query_type in {"number", "mixed"}:
            for match in NUMBER_PHRASE_PATTERN.finditer(span):
                add_candidate(match.group(0), f"{span_source}:number_phrase")
            for match in NUMBER_PATTERN.finditer(span):
                add_candidate(match.group(0), f"{span_source}:number")

        if query_type in {"date", "mixed"}:
            for pattern, source in [(MONTH_PATTERN, "date"), (YEAR_PATTERN, "year")]:
                for match in pattern.finditer(span):
                    add_candidate(match.group(0), f"{span_source}:{source}")

        if query_type in {"entity", "mixed"}:
            for match in ENTITY_PATTERN.finditer(span):
                label = re.sub(r"\s+", " ", match.group(0)).strip()
                if len(label) < 2 or len(label) > 80:
                    continue
                add_candidate(label, f"{span_source}:entity")

    return unique_mentions(mentions)[:max_mentions]


def extract_reference_mentions(
    text: str,
    reference_answers: Optional[Dict[str, Any]],
) -> List[AnswerMention]:
    if not reference_answers:
        return []
    normalized_doc = f" {normalize_text(text)} "
    mentions: List[AnswerMention] = []
    for kind, answers in reference_answers.items():
        for answer in coerce_answer_list(answers):
            key = answer_key(answer)
            if key and f" {key} " in normalized_doc:
                mentions.append(
                    AnswerMention(key=f"{kind}:{key}", label=answer, source="reference", kind=kind)
                )
    return unique_mentions(mentions)


def mention_context_window(text: str, label: str, max_chars: int = 360) -> str:
    label_key = answer_key(label)
    if not label_key:
        return re.sub(r"\s+", " ", text or "").strip()[:max_chars]
    for sentence in split_sentences(text):
        if label_key in answer_key(sentence):
            return re.sub(r"\s+", " ", sentence).strip()[:max_chars]
    return re.sub(r"\s+", " ", text or "").strip()[:max_chars]


class SemanticAnswerExtractor:
    """Lightweight model-based reranker for heuristic answer candidates."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
        batch_size: int = 32,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts: List[str]):
        vectors = []
        torch = self.torch
        for offset in range(0, len(texts), self.batch_size):
            batch_texts = texts[offset : offset + self.batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
                token_embeddings = output.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
        return torch.cat(vectors, dim=0)

    def extract(
        self,
        query: str,
        text: str,
        candidates: List[AnswerMention],
        max_mentions: int,
        min_score: float,
    ) -> List[AnswerMention]:
        unique_candidates = unique_mentions(candidates)
        if not unique_candidates:
            return []

        query_text = f"question: {query}"
        evidence_texts = [
            "answer candidate: "
            f"{candidate.label}. evidence: {mention_context_window(text, candidate.label)}"
            for candidate in unique_candidates
        ]
        vectors = self._encode([query_text] + evidence_texts)
        query_vector = vectors[0]
        evidence_vectors = vectors[1:]
        scores = (evidence_vectors * query_vector).sum(dim=1).tolist()
        scored_mentions = []
        for idx, (candidate, score) in enumerate(zip(unique_candidates, scores)):
            if score < min_score:
                continue
            scored_mentions.append((float(score), -idx, candidate))
        scored_mentions.sort(reverse=True)

        selected: List[AnswerMention] = []
        for score, _, candidate in scored_mentions[:max_mentions]:
            selected.append(
                AnswerMention(
                    key=candidate.key,
                    label=candidate.label,
                    source=f"semantic:{candidate.source}",
                    kind=candidate.kind,
                    score=score,
                )
            )
        return selected

    def extract_batch(
        self,
        query: str,
        texts: List[str],
        candidates_per_text: List[List[AnswerMention]],
        max_mentions: int,
        min_score: float,
    ) -> List[List[AnswerMention]]:
        query_text = f"question: {query}"
        evidence_texts: List[str] = []
        flat_items: List[Tuple[int, int, AnswerMention]] = []
        for text_idx, (text, candidates) in enumerate(zip(texts, candidates_per_text)):
            for candidate_idx, candidate in enumerate(unique_mentions(candidates)):
                flat_items.append((text_idx, candidate_idx, candidate))
                evidence_texts.append(
                    "answer candidate: "
                    f"{candidate.label}. evidence: {mention_context_window(text, candidate.label)}"
                )
        if not flat_items:
            return [[] for _ in texts]

        vectors = self._encode([query_text] + evidence_texts)
        query_vector = vectors[0]
        evidence_vectors = vectors[1:]
        scores = (evidence_vectors * query_vector).sum(dim=1).tolist()
        grouped: List[List[Tuple[float, int, AnswerMention]]] = [[] for _ in texts]
        for flat_idx, ((text_idx, candidate_idx, candidate), score) in enumerate(
            zip(flat_items, scores)
        ):
            if score < min_score:
                continue
            grouped[text_idx].append((float(score), -candidate_idx, candidate))

        outputs: List[List[AnswerMention]] = []
        for scored_mentions in grouped:
            scored_mentions.sort(reverse=True)
            selected = [
                AnswerMention(
                    key=candidate.key,
                    label=candidate.label,
                    source=f"semantic:{candidate.source}",
                    kind=candidate.kind,
                    score=score,
                )
                for score, _, candidate in scored_mentions[:max_mentions]
            ]
            outputs.append(selected)
        return outputs


class ExtractiveQAAnswerExtractor:
    """Small extractive QA model for question-conditioned answer span extraction."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 384,
        batch_size: int = 8,
    ) -> None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import torch

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def extract(
        self,
        query: str,
        text: str,
        max_mentions: int,
        top_starts: int = 8,
        top_ends: int = 8,
        max_answer_tokens: int = 12,
        min_score: float = -1e9,
    ) -> List[AnswerMention]:
        torch = self.torch
        encoded = self.tokenizer(
            query,
            text,
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        context_positions = [
            idx
            for idx, seq_id in enumerate(sequence_ids)
            if seq_id == 1 and offsets[idx][1] > offsets[idx][0]
        ]
        if not context_positions:
            return []

        batch = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**batch)
            start_logits = output.start_logits[0].detach().cpu()
            end_logits = output.end_logits[0].detach().cpu()

        mask = torch.full_like(start_logits, -1e9)
        mask[context_positions] = 0.0
        start_scores = start_logits + mask
        end_scores = end_logits + mask
        start_top = torch.topk(start_scores, k=min(top_starts, len(context_positions))).indices.tolist()
        end_top = torch.topk(end_scores, k=min(top_ends, len(context_positions))).indices.tolist()

        scored_mentions: List[Tuple[float, int, AnswerMention]] = []
        order = 0
        query_tokens = set(normalize_text(query).split())
        for start_idx in start_top:
            for end_idx in end_top:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > max_answer_tokens:
                    continue
                start_char, _ = offsets[start_idx]
                _, end_char = offsets[end_idx]
                if end_char <= start_char:
                    continue
                label = clean_candidate_label(text[start_char:end_char], max_words=max_answer_tokens)
                if candidate_is_low_value(label, query_tokens) or candidate_is_sentence_like_answer(label):
                    continue
                score = float(start_scores[start_idx].item() + end_scores[end_idx].item())
                if score < min_score:
                    continue
                scored_mentions.append(
                    (
                        score,
                        -order,
                        AnswerMention(
                            key=answer_key(label),
                            label=label,
                            source="extractive_qa",
                            kind="qa",
                            score=score,
                        ),
                    )
                )
                order += 1

        scored_mentions.sort(reverse=True)
        return unique_mentions(item[2] for item in scored_mentions)[:max_mentions]

    def extract_batch(
        self,
        query: str,
        texts: List[str],
        max_mentions: int,
        top_starts: int = 8,
        top_ends: int = 8,
        max_answer_tokens: int = 12,
        min_score: float = -1e9,
    ) -> List[List[AnswerMention]]:
        outputs: List[List[AnswerMention]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch_texts = texts[offset : offset + self.batch_size]
            outputs.extend(
                self._extract_batch_chunk(
                    query=query,
                    texts=batch_texts,
                    max_mentions=max_mentions,
                    top_starts=top_starts,
                    top_ends=top_ends,
                    max_answer_tokens=max_answer_tokens,
                    min_score=min_score,
                )
            )
        return outputs

    def _extract_batch_chunk(
        self,
        query: str,
        texts: List[str],
        max_mentions: int,
        top_starts: int,
        top_ends: int,
        max_answer_tokens: int,
        min_score: float,
    ) -> List[List[AnswerMention]]:
        torch = self.torch
        encoded = self.tokenizer(
            [query] * len(texts),
            texts,
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        batch = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**batch)
            start_logits_batch = output.start_logits.detach().cpu()
            end_logits_batch = output.end_logits.detach().cpu()

        query_tokens = set(normalize_text(query).split())
        outputs: List[List[AnswerMention]] = []
        for item_idx, text in enumerate(texts):
            offsets = offsets_batch[item_idx]
            sequence_ids = encoded.sequence_ids(item_idx)
            context_positions = [
                idx
                for idx, seq_id in enumerate(sequence_ids)
                if seq_id == 1 and offsets[idx][1] > offsets[idx][0]
            ]
            if not context_positions:
                outputs.append([])
                continue

            start_logits = start_logits_batch[item_idx]
            end_logits = end_logits_batch[item_idx]
            mask = torch.full_like(start_logits, -1e9)
            mask[context_positions] = 0.0
            start_scores = start_logits + mask
            end_scores = end_logits + mask
            start_top = torch.topk(
                start_scores,
                k=min(top_starts, len(context_positions)),
            ).indices.tolist()
            end_top = torch.topk(
                end_scores,
                k=min(top_ends, len(context_positions)),
            ).indices.tolist()

            scored_mentions: List[Tuple[float, int, AnswerMention]] = []
            order = 0
            for start_idx in start_top:
                for end_idx in end_top:
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_tokens:
                        continue
                    start_char, _ = offsets[start_idx]
                    _, end_char = offsets[end_idx]
                    if end_char <= start_char:
                        continue
                    label = clean_candidate_label(
                        text[start_char:end_char],
                        max_words=max_answer_tokens,
                    )
                    if candidate_is_low_value(label, query_tokens) or candidate_is_sentence_like_answer(label):
                        continue
                    score = float(start_scores[start_idx].item() + end_scores[end_idx].item())
                    if score < min_score:
                        continue
                    scored_mentions.append(
                        (
                            score,
                            -order,
                            AnswerMention(
                                key=answer_key(label),
                                label=label,
                                source="extractive_qa",
                                kind="qa",
                                score=score,
                            ),
                        )
                    )
                    order += 1

            scored_mentions.sort(reverse=True)
            outputs.append(unique_mentions(item[2] for item in scored_mentions)[:max_mentions])
        return outputs


class EvidenceHardener:
    """Post-retrieval evidence hardening for RAG poisoning experiments.

    Reference answers are used only for diagnostics unless
    answer_support.use_reference_answers_for_scoring is explicitly enabled.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        cluster_cfg = config.get("cluster", {})
        answer_cfg = config.get("answer_support", {})
        contradiction_cfg = config.get("contradiction", {})
        head_cfg = config.get("head_filter", {})
        query_echo_cfg = config.get("query_echo", {})
        provenance_gate_cfg = config.get("provenance_gate", {})
        rank_guard_cfg = config.get("rank_guard", {})
        top1_cfg = config.get("top1_dominance", {})
        margin_cfg = config.get("margin_gate", {})
        constrained_cfg = config.get("constrained_selection", {})
        consequence_cfg = config.get("consequence_policy", {})

        self.cluster_enabled = bool(cluster_cfg.get("enabled", True))
        self.cluster_cap = int(cluster_cfg.get("cap", 1))
        self.cluster_jaccard_threshold = float(cluster_cfg.get("jaccard_threshold", 0.72))

        self.answer_enabled = bool(answer_cfg.get("enabled", True))
        self.max_answer_mentions = int(answer_cfg.get("max_mentions_per_doc", 5))
        self.use_reference_for_scoring = bool(
            answer_cfg.get("use_reference_answers_for_scoring", False)
        )
        reference_kinds = answer_cfg.get("reference_answer_kinds_for_scoring")
        if reference_kinds is None:
            self.reference_answer_kinds_for_scoring: Optional[Set[str]] = None
        else:
            self.reference_answer_kinds_for_scoring = {
                str(kind).lower()
                for kind in reference_kinds
                if str(kind).strip()
            }
        self.max_docs_per_answer = int(answer_cfg.get("max_docs_per_answer", 0))
        self.diversity_bonus = float(answer_cfg.get("diversity_bonus", 0.006))
        self.duplicate_cluster_penalty = float(
            answer_cfg.get("duplicate_cluster_penalty", 0.004)
        )
        self.max_answer_bonus_clusters = int(answer_cfg.get("max_bonus_clusters", 3))
        self.answer_scoring_mode = str(answer_cfg.get("answer_scoring_mode", "support")).lower()
        self.robust_channel_bonus = float(answer_cfg.get("robust_channel_bonus", 0.003))
        self.robust_non_echo_bonus = float(answer_cfg.get("robust_non_echo_bonus", self.diversity_bonus))
        self.robust_query_echo_support_penalty = float(
            answer_cfg.get("robust_query_echo_support_penalty", 0.006)
        )
        self.robust_isolated_answer_penalty = float(
            answer_cfg.get("robust_isolated_answer_penalty", 0.006)
        )
        self.topic_grounding_enabled = bool(answer_cfg.get("grounding_enabled", False))
        self.topic_grounding_min_query_coverage = float(
            answer_cfg.get("grounding_min_query_coverage", 0.25)
        )
        self.topic_grounding_min_overlap = int(answer_cfg.get("grounding_min_overlap", 2))
        self.topic_grounding_short_doc_token_threshold = int(
            answer_cfg.get("grounding_short_doc_token_threshold", 18)
        )
        self.topic_grounding_low_support_penalty = float(
            answer_cfg.get("grounding_low_support_penalty", 0.012)
        )
        self.topic_grounding_short_doc_penalty = float(
            answer_cfg.get("grounding_short_doc_penalty", 0.04)
        )
        self.topic_grounding_block_support_bonus = bool(
            answer_cfg.get("grounding_block_support_bonus", True)
        )
        self.evidence_fallback_enabled = bool(
            answer_cfg.get("fallback_to_evidence_mode", True)
        )
        self.evidence_fallback_min_answer_coverage = float(
            answer_cfg.get("evidence_mode_min_answer_coverage", 0.25)
        )
        self.evidence_fallback_min_answer_count = int(
            answer_cfg.get("evidence_mode_min_answer_count", 1)
        )
        self.evidence_mode_use_query_echo = bool(
            answer_cfg.get("evidence_mode_use_query_echo", True)
        )
        self.evidence_mode_use_topic_grounding = bool(
            answer_cfg.get("evidence_mode_use_topic_grounding", True)
        )
        self.evidence_mode_grounding_bonus = float(
            answer_cfg.get("evidence_mode_grounding_bonus", 0.004)
        )
        self.evidence_mode_query_coverage_bonus = float(
            answer_cfg.get("evidence_mode_query_coverage_bonus", 0.006)
        )
        self.evidence_mode_singleton_cluster_bonus = float(
            answer_cfg.get("evidence_mode_singleton_cluster_bonus", 0.003)
        )
        self.evidence_mode_channel_bonus = float(
            answer_cfg.get("evidence_mode_channel_bonus", self.robust_channel_bonus)
        )
        self.evidence_mode_query_echo_penalty = float(
            answer_cfg.get(
                "evidence_mode_query_echo_penalty",
                self.robust_query_echo_support_penalty,
            )
        )
        self.evidence_mode_low_grounding_penalty = float(
            answer_cfg.get(
                "evidence_mode_low_grounding_penalty",
                self.topic_grounding_low_support_penalty,
            )
        )
        self.evidence_mode_short_doc_penalty = float(
            answer_cfg.get(
                "evidence_mode_short_doc_penalty",
                self.topic_grounding_short_doc_penalty,
            )
        )
        self.use_query_focus = bool(answer_cfg.get("use_query_focus", False))
        self.focused_sentence_count = int(answer_cfg.get("focused_sentence_count", 4))
        self.max_candidate_words = int(answer_cfg.get("max_candidate_words", 8))
        self.use_answer_cues = bool(answer_cfg.get("use_answer_cues", False))
        self.answer_extractor_name = str(answer_cfg.get("extractor", "heuristic")).lower()
        self.semantic_model_name = str(
            answer_cfg.get("semantic_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.semantic_device = str(answer_cfg.get("semantic_device", "auto"))
        self.semantic_local_files_only = bool(answer_cfg.get("semantic_local_files_only", True))
        self.semantic_max_length = int(answer_cfg.get("semantic_max_length", 192))
        self.semantic_batch_size = int(answer_cfg.get("semantic_batch_size", 32))
        self.semantic_candidate_pool = int(
            answer_cfg.get("semantic_candidate_pool", max(self.max_answer_mentions, 12))
        )
        self.semantic_min_score = float(answer_cfg.get("semantic_min_score", -1.0))
        self.semantic_fallback_to_heuristic = bool(
            answer_cfg.get("semantic_fallback_to_heuristic", True)
        )
        self.semantic_answer_extractor: Optional[SemanticAnswerExtractor] = None
        self.semantic_answer_extractor_loaded = False
        self.semantic_answer_extractor_error: Optional[str] = None
        self.qa_model_name = str(answer_cfg.get("qa_model_name", "deepset/minilm-uncased-squad2"))
        self.qa_device = str(answer_cfg.get("qa_device", "auto"))
        self.qa_local_files_only = bool(answer_cfg.get("qa_local_files_only", True))
        self.qa_max_length = int(answer_cfg.get("qa_max_length", 384))
        self.qa_batch_size = int(answer_cfg.get("qa_batch_size", 8))
        self.qa_top_starts = int(answer_cfg.get("qa_top_starts", 8))
        self.qa_top_ends = int(answer_cfg.get("qa_top_ends", 8))
        self.qa_top_spans = int(answer_cfg.get("qa_top_spans", self.max_answer_mentions))
        self.qa_max_answer_tokens = int(answer_cfg.get("qa_max_answer_tokens", 12))
        self.qa_min_score = float(answer_cfg.get("qa_min_score", -1e9))
        self.qa_include_heuristic = bool(answer_cfg.get("qa_include_heuristic", True))
        self.qa_heuristic_top_k = int(answer_cfg.get("qa_heuristic_top_k", 2))
        self.qa_use_semantic_rerank = bool(answer_cfg.get("qa_use_semantic_rerank", True))
        self.qa_fallback_to_heuristic = bool(answer_cfg.get("qa_fallback_to_heuristic", True))
        self.qa_scoring_mode = str(answer_cfg.get("qa_scoring_mode", "score")).lower()
        self.qa_answer_extractor: Optional[ExtractiveQAAnswerExtractor] = None
        self.qa_answer_extractor_loaded = False
        self.qa_answer_extractor_error: Optional[str] = None

        self.contradiction_enabled = bool(contradiction_cfg.get("enabled", True))
        self.conflict_penalty = float(contradiction_cfg.get("conflict_penalty", 0.004))
        self.min_conflict_answers = int(contradiction_cfg.get("min_conflict_answers", 2))

        self.margin_gate_enabled = bool(margin_cfg.get("enabled", False))
        self.margin_gate_mode = str(margin_cfg.get("mode", "simple")).lower()
        self.margin_gate_window = int(margin_cfg.get("window", 10))
        self.margin_gate_threshold = float(margin_cfg.get("threshold", 0.75))
        self.margin_gate_dynamic_threshold = bool(
            margin_cfg.get("dynamic_threshold", self.margin_gate_mode == "complex")
        )
        self.margin_gate_max_threshold = float(
            margin_cfg.get("max_threshold", self.margin_gate_threshold + 0.75)
        )
        self.margin_gate_cluster_weight = float(margin_cfg.get("cluster_weight", 1.0))
        self.margin_gate_channel_weight = float(margin_cfg.get("channel_weight", 0.25))
        self.margin_gate_echo_penalty = float(margin_cfg.get("echo_penalty", 0.75))
        self.margin_gate_isolated_penalty = float(margin_cfg.get("isolated_penalty", 0.5))
        self.margin_gate_conflict_penalty = float(margin_cfg.get("conflict_penalty", 0.25))
        self.margin_gate_top_answer_penalty = float(
            margin_cfg.get("top_answer_penalty", 0.02)
        )
        self.margin_gate_alternative_bonus = float(
            margin_cfg.get("alternative_bonus", 0.012)
        )
        self.margin_gate_supplement_bonus = float(
            margin_cfg.get("supplement_bonus", 0.012)
        )
        self.margin_gate_min_conflict_answers = int(
            margin_cfg.get("min_conflict_answers", self.min_conflict_answers)
        )
        self.margin_gate_use_non_echo_clusters = bool(
            margin_cfg.get("use_non_echo_clusters", True)
        )
        self.margin_gate_weak_top_delta = float(margin_cfg.get("weak_top_delta", 0.25))
        self.margin_gate_echo_top_delta = float(margin_cfg.get("echo_top_delta", 0.15))
        self.margin_gate_multi_supported_delta = float(
            margin_cfg.get("multi_supported_delta", 0.15)
        )
        self.margin_gate_no_strong_answer_delta = float(
            margin_cfg.get("no_strong_answer_delta", 0.2)
        )
        self.margin_gate_high_risk_delta = float(margin_cfg.get("high_risk_delta", 0.35))
        self.margin_gate_weak_independent_clusters = int(
            margin_cfg.get("weak_independent_clusters", 1)
        )
        self.margin_gate_weak_channel_count = int(margin_cfg.get("weak_channel_count", 1))
        self.margin_gate_echo_ratio_threshold = float(
            margin_cfg.get("echo_ratio_threshold", 0.5)
        )
        self.margin_gate_penalize_top_only_if_weak = bool(
            margin_cfg.get("penalize_top_only_if_weak", self.margin_gate_mode == "complex")
        )
        self.margin_gate_strong_alternative_min_clusters = int(
            margin_cfg.get("strong_alternative_min_clusters", 2)
        )
        self.margin_gate_max_alternatives = int(margin_cfg.get("max_alternatives", 2))
        self.margin_gate_top_penalty_multiplier = float(
            margin_cfg.get("top_penalty_multiplier", 1.0)
        )
        self.margin_gate_alternative_bonus_multiplier = float(
            margin_cfg.get("alternative_bonus_multiplier", 1.0)
        )
        self.margin_gate_supplement_bonus_multiplier = float(
            margin_cfg.get("supplement_bonus_multiplier", 1.0)
        )
        self.margin_gate_echo_doc_extra_penalty = float(
            margin_cfg.get("echo_doc_extra_penalty", 0.006)
        )
        self.margin_gate_low_grounding_extra_penalty = float(
            margin_cfg.get("low_grounding_extra_penalty", 0.006)
        )
        self.margin_gate_min_supplement_rank = int(margin_cfg.get("min_supplement_rank", 4))
        self.margin_gate_max_supplement_rank = int(
            margin_cfg.get("max_supplement_rank", self.margin_gate_window)
        )
        self.margin_gate_preserve_rank1_if_no_alternative = bool(
            margin_cfg.get("preserve_rank1_if_no_alternative", True)
        )
        self.margin_gate_high_risk_abstain = bool(margin_cfg.get("high_risk_abstain", False))
        # Fallback answering: on a low-margin case whose top answer shows no
        # poison signature (not echo-heavy, not isolated, and a strong answer
        # exists), keep the confident ordering instead of penalising it into an
        # abstention. Recovers correct answers on clean contexts at low ASR risk.
        self.margin_gate_fallback_when_clean = bool(
            margin_cfg.get("fallback_answer_when_clean", False)
        )

        self.consequence_policy_enabled = bool(consequence_cfg.get("enabled", False))
        self.consequence_policy_kb_path = str(
            consequence_cfg.get("kb_path", "retrieval_framework/consequence_kb/policies.jsonl")
        )
        self.consequence_policy_use_margin_threshold = bool(
            consequence_cfg.get("use_margin_threshold", True)
        )
        self.consequence_policy_override_keyword_risk = bool(
            consequence_cfg.get("override_keyword_risk", True)
        )
        self.consequence_policy_enforce_authoritative_source = bool(
            consequence_cfg.get("enforce_authoritative_source", False)
        )
        self.consequence_policy_authoritative_provenance = {
            normalize_text(item)
            for item in consequence_cfg.get("authoritative_provenance", ["indexed_corpus"])
        }
        self.consequence_policy_provenance_field = str(
            consequence_cfg.get("provenance_field", "provenance")
        )
        self.consequence_kb: Optional[ConsequenceKB] = None
        if self.consequence_policy_enabled:
            self.consequence_kb = ConsequenceKB.from_jsonl(
                Path(self.consequence_policy_kb_path)
            )

        self.constrained_selection_enabled = bool(constrained_cfg.get("enabled", False))
        self.constrained_selection_pool_depth = int(
            constrained_cfg.get("pool_depth", self.margin_gate_window)
        )
        self.constrained_selection_max_cluster_docs = int(
            constrained_cfg.get("max_cluster_docs", max(1, self.cluster_cap))
        )
        self.constrained_selection_max_answer_docs = int(
            constrained_cfg.get("max_answer_docs", max(0, self.max_docs_per_answer))
        )
        self.constrained_selection_max_query_overlap_docs = int(
            constrained_cfg.get("max_query_overlap_docs", 1)
        )
        self.constrained_selection_cluster_penalty = float(
            constrained_cfg.get("duplicate_cluster_penalty", 0.025)
        )
        self.constrained_selection_answer_penalty = float(
            constrained_cfg.get("answer_concentration_penalty", 0.008)
        )
        self.constrained_selection_query_overlap_penalty = float(
            constrained_cfg.get("query_overlap_concentration_penalty", 0.02)
        )
        self.constrained_selection_low_margin_penalty = float(
            constrained_cfg.get("low_margin_answer_penalty", 0.012)
        )
        self.constrained_selection_isolated_rank1_penalty = float(
            constrained_cfg.get("isolated_rank1_penalty", 0.025)
        )
        self.constrained_selection_support_bonus = float(
            constrained_cfg.get("support_bonus", 0.006)
        )
        self.constrained_selection_channel_bonus = float(
            constrained_cfg.get("channel_bonus", 0.002)
        )
        self.constrained_selection_new_cluster_bonus = float(
            constrained_cfg.get("new_answer_cluster_bonus", 0.004)
        )
        self.constrained_selection_min_gain = float(
            constrained_cfg.get("min_gain", -1e9)
        )

        self.query_echo_enabled = bool(query_echo_cfg.get("enabled", False))
        self.query_echo_mode = str(
            query_echo_cfg.get("mode", "query_overlap_anomaly")
        ).lower()
        self.query_echo_prefix_window = int(query_echo_cfg.get("prefix_window_tokens", 40))
        self.query_echo_overlap_threshold = float(query_echo_cfg.get("overlap_threshold", 0.9))
        self.query_echo_ngram_n = int(query_echo_cfg.get("ngram_n", 3))
        self.query_echo_ngram_threshold = float(
            query_echo_cfg.get("ngram_overlap_threshold", 0.5)
        )
        self.query_echo_min_novelty_tokens = int(
            query_echo_cfg.get("min_novelty_tokens", 4)
        )
        self.query_echo_min_novelty_ratio = float(
            query_echo_cfg.get("min_novelty_ratio", 0.18)
        )
        self.query_echo_exact_prefix_penalty = float(
            query_echo_cfg.get("exact_prefix_penalty", 0.03)
        )
        self.query_echo_high_overlap_penalty = float(
            query_echo_cfg.get("high_overlap_penalty", 0.008)
        )
        self.injection_risk_enabled = bool(
            query_echo_cfg.get(
                "injection_risk_enabled",
                self.query_echo_mode in {"hybrid_injection_risk", "injection_risk"},
            )
        )
        self.injection_query_copy_enabled = bool(
            query_echo_cfg.get("query_copy_enabled", True)
        )
        self.injection_textual_cues_enabled = bool(
            query_echo_cfg.get("textual_cues_enabled", self.injection_risk_enabled)
        )
        self.injection_assertion_cues = tuple(
            normalize_text(cue)
            for cue in query_echo_cfg.get(
                "assertion_cues",
                [
                    "the answer is",
                    "answer is",
                    "correct answer is",
                    "the correct answer is",
                    "final answer",
                    "answer should be",
                ],
            )
            if normalize_text(cue)
        )
        self.injection_instruction_cues = tuple(
            normalize_text(cue)
            for cue in query_echo_cfg.get(
                "instruction_cues",
                [
                    "ignore previous",
                    "ignore the previous",
                    "respond with",
                    "you should answer",
                    "must answer",
                    "use this answer",
                    "do not mention",
                ],
            )
            if normalize_text(cue)
        )
        self.injection_assertion_penalty = float(
            query_echo_cfg.get("assertion_penalty", 0.012)
        )
        self.injection_instruction_penalty = float(
            query_echo_cfg.get("instruction_penalty", 0.02)
        )
        self.injection_short_doc_tokens = int(
            query_echo_cfg.get("short_assertion_doc_tokens", 48)
        )
        self.injection_short_doc_extra_penalty = float(
            query_echo_cfg.get("short_assertion_extra_penalty", 0.004)
        )
        self.injection_provenance_enabled = bool(
            query_echo_cfg.get("provenance_enabled", False)
        )
        self.injection_provenance_field = str(
            query_echo_cfg.get("provenance_field", "provenance")
        )
        self.injection_untrusted_provenance = {
            normalize_text(value)
            for value in query_echo_cfg.get(
                "untrusted_provenance",
                ["runtime_external", "external_upload", "unverified_source"],
            )
            if normalize_text(value)
        }
        self.injection_untrusted_claim_penalty = float(
            query_echo_cfg.get("untrusted_claim_penalty", 0.03)
        )
        self.provenance_gate_enabled = bool(provenance_gate_cfg.get("enabled", False))
        self.provenance_gate_mode = str(
            provenance_gate_cfg.get("mode", "trusted_context_only")
        ).lower()
        if self.provenance_gate_mode not in {"trusted_context_only"}:
            raise ValueError(
                f"Unsupported provenance_gate mode: {self.provenance_gate_mode}"
            )
        self.provenance_gate_field = str(
            provenance_gate_cfg.get("provenance_field", self.injection_provenance_field)
        )
        self.provenance_gate_trusted_provenance = {
            normalize_text(value)
            for value in provenance_gate_cfg.get(
                "trusted_provenance",
                ["indexed_corpus"],
            )
            if normalize_text(value)
        }
        self.provenance_gate_untrusted_provenance = {
            normalize_text(value)
            for value in provenance_gate_cfg.get(
                "untrusted_provenance",
                sorted(self.injection_untrusted_provenance),
            )
            if normalize_text(value)
        }
        self.provenance_gate_unknown_policy = str(
            provenance_gate_cfg.get("unknown_policy", "quarantine")
        ).lower()
        if self.provenance_gate_unknown_policy not in {"allow", "quarantine"}:
            raise ValueError(
                "provenance_gate.unknown_policy must be either allow or quarantine."
            )

        self.rank_guard_enabled = bool(rank_guard_cfg.get("enabled", False))
        self.rank_guard_top_n = int(rank_guard_cfg.get("top_n", 1))
        self.rank_guard_min_promotion_margin = float(
            rank_guard_cfg.get("min_promotion_margin", 0.002)
        )
        self.rank_guard_max_promotion_original_rank = int(
            rank_guard_cfg.get("max_promotion_original_rank", 20)
        )
        self.rank_guard_min_answer_clusters = int(
            rank_guard_cfg.get("min_answer_clusters", 1)
        )

        self.top1_dominance_enabled = bool(top1_cfg.get("enabled", False))
        self.top1_dominance_window = int(top1_cfg.get("window", 5))
        self.top1_dominance_min_conflict_answers = int(
            top1_cfg.get("min_conflict_answers", 2)
        )
        self.top1_dominance_isolated_cluster_threshold = int(
            top1_cfg.get("isolated_cluster_threshold", 1)
        )
        self.top1_dominance_alternative_cluster_threshold = int(
            top1_cfg.get("alternative_cluster_threshold", 2)
        )
        self.top1_dominance_penalty = float(top1_cfg.get("penalty", 0.025))
        self.top1_dominance_supported_bonus = float(
            top1_cfg.get("supported_answer_bonus", 0.006)
        )
        self.top1_dominance_query_echo_extra_penalty = float(
            top1_cfg.get("query_echo_extra_penalty", 0.008)
        )
        self.top1_dominance_max_supplement_rank = int(
            top1_cfg.get("max_supplement_rank", 5)
        )

        self.head_filter_enabled = bool(head_cfg.get("enabled", False))
        self.head_filter_mode = str(head_cfg.get("mode", "conservative")).lower()
        self.head_k = int(head_cfg.get("head_k", 3))
        self.head_supplement_k = int(head_cfg.get("supplement_k", 2))
        self.head_min_conflict_answers = int(head_cfg.get("min_conflict_answers", 2))
        self.head_independent_support_threshold = int(
            head_cfg.get("independent_support_threshold", 2)
        )
        self.head_isolated_penalty = float(head_cfg.get("isolated_penalty", 0.02))
        self.head_supported_answer_bonus = float(
            head_cfg.get("supported_answer_bonus", 0.01)
        )
        self.head_supplement_bonus = float(head_cfg.get("supplement_bonus", 0.012))
        self.head_protect_rank1 = bool(head_cfg.get("protect_rank1", True))
        self.head_allow_supplement_new_answers = bool(
            head_cfg.get("allow_supplement_new_answers", False)
        )
        self.head_max_supplement_promotion_rank = int(
            head_cfg.get("max_supplement_promotion_rank", 3)
        )
        self.head_high_risk_abstain = bool(head_cfg.get("high_risk_abstain", False))
        self.head_high_risk_keywords = [
            str(item).lower()
            for item in head_cfg.get(
                "high_risk_keywords",
                [
                    "medical",
                    "medicine",
                    "health",
                    "legal",
                    "law",
                    "financial",
                    "finance",
                    "investment",
                    "safety",
                    "emergency",
                    "dosage",
                    "diagnosis",
                    "treatment",
                ],
            )
        ]

        self.rank_k = int(config.get("rank_k", 60))

    def _uses_semantic_answer_extractor(self) -> bool:
        return self.answer_extractor_name in {"semantic", "semantic_qa", "model", "model_rerank"}

    def _uses_qa_answer_extractor(self) -> bool:
        return self.answer_extractor_name in {"qa", "qa_lite", "qa_lite_plus", "extractive_qa"}

    def _get_semantic_answer_extractor(self) -> Optional[SemanticAnswerExtractor]:
        if self.semantic_answer_extractor_loaded:
            return self.semantic_answer_extractor
        self.semantic_answer_extractor_loaded = True
        try:
            self.semantic_answer_extractor = SemanticAnswerExtractor(
                model_name=self.semantic_model_name,
                device=self.semantic_device,
                local_files_only=self.semantic_local_files_only,
                max_length=self.semantic_max_length,
                batch_size=self.semantic_batch_size,
            )
        except Exception as exc:
            self.semantic_answer_extractor_error = str(exc)
            self.semantic_answer_extractor = None
            if not self.semantic_fallback_to_heuristic:
                raise
        return self.semantic_answer_extractor

    def _get_qa_answer_extractor(self) -> Optional[ExtractiveQAAnswerExtractor]:
        if self.qa_answer_extractor_loaded:
            return self.qa_answer_extractor
        self.qa_answer_extractor_loaded = True
        try:
            self.qa_answer_extractor = ExtractiveQAAnswerExtractor(
                model_name=self.qa_model_name,
                device=self.qa_device,
                local_files_only=self.qa_local_files_only,
                max_length=self.qa_max_length,
                batch_size=self.qa_batch_size,
            )
        except Exception as exc:
            self.qa_answer_extractor_error = str(exc)
            self.qa_answer_extractor = None
            if not self.qa_fallback_to_heuristic:
                raise
        return self.qa_answer_extractor

    def _assign_clusters(self, entries: List[CandidateEntry]) -> Dict[int, Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for entry in entries:
            tokens = cluster_tokens(entry.result.text)
            matched_cluster = None
            for cluster in clusters:
                if jaccard_similarity(tokens, cluster["tokens"]) >= self.cluster_jaccard_threshold:
                    matched_cluster = cluster
                    break
            if matched_cluster is None:
                matched_cluster = {
                    "cluster_id": len(clusters),
                    "tokens": tokens,
                    "doc_ids": [],
                    "adv_doc_count": 0,
                }
                clusters.append(matched_cluster)
            entry.cluster_id = int(matched_cluster["cluster_id"])
            matched_cluster["doc_ids"].append(entry.result.doc_id)
            if entry.result.metadata.get("is_adv"):
                matched_cluster["adv_doc_count"] += 1
        return {
            int(cluster["cluster_id"]): {
                "size": len(cluster["doc_ids"]),
                "doc_ids": list(cluster["doc_ids"]),
                "adv_doc_count": int(cluster["adv_doc_count"]),
            }
            for cluster in clusters
        }

    def _attach_answer_mentions(
        self,
        entries: List[CandidateEntry],
        query: str,
        reference_answers: Optional[Dict[str, Any]],
    ) -> None:
        if self.answer_enabled and self._uses_qa_answer_extractor():
            candidate_mentions_by_entry: List[List[AnswerMention]] = []
            for entry in entries:
                candidate_mentions_by_entry.append(
                    extract_heuristic_answers(
                        query,
                        entry.result.text,
                        max_mentions=max(self.max_answer_mentions, self.qa_heuristic_top_k),
                        focused_sentence_count=self.focused_sentence_count,
                        max_candidate_words=self.max_candidate_words,
                        use_query_focus=self.use_query_focus,
                        use_answer_cues=self.use_answer_cues,
                    )
                )

            extractor = self._get_qa_answer_extractor()
            texts = [entry.result.text for entry in entries]
            if extractor is not None:
                qa_mentions_by_entry = extractor.extract_batch(
                    query=query,
                    texts=texts,
                    max_mentions=max(self.max_answer_mentions, self.qa_top_spans),
                    top_starts=self.qa_top_starts,
                    top_ends=self.qa_top_ends,
                    max_answer_tokens=self.qa_max_answer_tokens,
                    min_score=self.qa_min_score,
                )
            else:
                qa_mentions_by_entry = [[] for _ in entries]

            combined_by_entry: List[List[AnswerMention]] = []
            for qa_mentions, candidate_mentions in zip(
                qa_mentions_by_entry,
                candidate_mentions_by_entry,
            ):
                combined_mentions: List[AnswerMention] = list(qa_mentions)
                if self.qa_include_heuristic:
                    combined_mentions.extend(candidate_mentions[: max(0, self.qa_heuristic_top_k)])
                if not combined_mentions and self.qa_fallback_to_heuristic:
                    combined_mentions = candidate_mentions[: self.max_answer_mentions]
                combined_by_entry.append(unique_mentions(combined_mentions))

            if self.qa_use_semantic_rerank and any(combined_by_entry):
                semantic_extractor = self._get_semantic_answer_extractor()
                if semantic_extractor is not None:
                    reranked_by_entry = semantic_extractor.extract_batch(
                        query=query,
                        texts=texts,
                        candidates_per_text=combined_by_entry,
                        max_mentions=self.max_answer_mentions,
                        min_score=self.semantic_min_score,
                    )
                else:
                    reranked_by_entry = combined_by_entry
            else:
                reranked_by_entry = combined_by_entry

            for entry, mentions, candidate_mentions in zip(
                entries,
                reranked_by_entry,
                candidate_mentions_by_entry,
            ):
                entry.diagnostic_mentions = mentions[: self.max_answer_mentions]
                if self.qa_scoring_mode in {"annotate", "diagnostic", "label_only"}:
                    entry.heuristic_mentions = candidate_mentions[: self.max_answer_mentions]
                else:
                    entry.heuristic_mentions = mentions[: self.max_answer_mentions]
                entry.reference_mentions = extract_reference_mentions(
                    entry.result.text,
                    reference_answers,
                )
            return

        for entry in entries:
            if self.answer_enabled:
                candidate_pool = (
                    max(self.max_answer_mentions, self.semantic_candidate_pool)
                    if self._uses_semantic_answer_extractor()
                    else self.max_answer_mentions
                )
                candidate_mentions = extract_heuristic_answers(
                    query,
                    entry.result.text,
                    max_mentions=candidate_pool,
                    focused_sentence_count=self.focused_sentence_count,
                    max_candidate_words=self.max_candidate_words,
                    use_query_focus=self.use_query_focus,
                    use_answer_cues=self.use_answer_cues,
                )
                if self._uses_semantic_answer_extractor():
                    extractor = self._get_semantic_answer_extractor()
                    if extractor is not None:
                        semantic_mentions = extractor.extract(
                            query=query,
                            text=entry.result.text,
                            candidates=candidate_mentions,
                            max_mentions=self.max_answer_mentions,
                            min_score=self.semantic_min_score,
                        )
                        if semantic_mentions or not self.semantic_fallback_to_heuristic:
                            entry.heuristic_mentions = semantic_mentions
                        else:
                            entry.heuristic_mentions = candidate_mentions[: self.max_answer_mentions]
                    else:
                        entry.heuristic_mentions = candidate_mentions[: self.max_answer_mentions]
                else:
                    entry.heuristic_mentions = candidate_mentions[: self.max_answer_mentions]
            entry.reference_mentions = extract_reference_mentions(entry.result.text, reference_answers)

    def _mentions_for_scoring(self, entry: CandidateEntry) -> List[AnswerMention]:
        if not self.answer_enabled:
            return []
        if self.use_reference_for_scoring:
            if self.reference_answer_kinds_for_scoring is None:
                return entry.reference_mentions
            return [
                mention
                for mention in entry.reference_mentions
                if str(mention.kind).lower() in self.reference_answer_kinds_for_scoring
            ]
        return entry.heuristic_mentions

    def _build_answer_support(
        self,
        entries: List[CandidateEntry],
        grounded_only: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        support: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if grounded_only and entry.topic_grounding_low:
                continue
            for mention in self._mentions_for_scoring(entry):
                item = support.setdefault(
                    mention.key,
                    {
                        "label": mention.label,
                        "doc_ids": set(),
                        "cluster_ids": set(),
                        "channel_names": set(),
                        "mentions": 0,
                    },
                )
                item["doc_ids"].add(entry.result.doc_id)
                item["cluster_ids"].add(entry.cluster_id)
                item["channel_names"].update(self._entry_channel_names(entry))
                item["mentions"] += 1
        return support

    @staticmethod
    def _entry_channel_names(entry: CandidateEntry) -> Set[str]:
        channels = entry.result.metadata.get("channels")
        if isinstance(channels, dict) and channels:
            return {str(name) for name in channels}
        source = entry.result.source or "unknown"
        return {str(source)}

    def _build_reference_support(
        self,
        entries: List[CandidateEntry],
    ) -> Dict[str, Dict[str, Any]]:
        support: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"doc_ids": set(), "cluster_ids": set(), "labels": set()}
        )
        for entry in entries:
            for mention in entry.reference_mentions:
                item = support[mention.kind]
                item["doc_ids"].add(entry.result.doc_id)
                item["cluster_ids"].add(entry.cluster_id)
                item["labels"].add(mention.label)
        return {
            kind: {
                "doc_count": len(item["doc_ids"]),
                "cluster_count": len(item["cluster_ids"]),
                "labels": sorted(item["labels"]),
            }
            for kind, item in support.items()
        }

    def _query_ngrams(self, tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
        if n <= 0 or len(tokens) < n:
            return set()
        return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}

    def _query_echo_stats(self, query: str, text: str) -> Tuple[bool, float, float, float, int]:
        query_norm = normalize_text(query)
        doc_norm = normalize_text(text)
        if not query_norm or not doc_norm:
            return False, 0.0, 0.0, 1.0, 0

        exact_prefix = doc_norm.startswith(query_norm)
        query_token_list = [
            token
            for token in query_norm.split()
            if token and token not in COMMON_ENTITY_WORDS
        ]
        query_tokens = set(query_token_list)
        if not query_tokens:
            query_token_list = query_norm.split()
            query_tokens = set(query_token_list)

        prefix_token_list = doc_norm.split()[: self.query_echo_prefix_window]
        prefix_tokens = set(prefix_token_list)
        overlap = len(query_tokens & prefix_tokens) / max(len(query_tokens), 1)

        query_ngrams = self._query_ngrams(query_token_list, self.query_echo_ngram_n)
        prefix_ngrams = self._query_ngrams(prefix_token_list, self.query_echo_ngram_n)
        ngram_overlap = (
            len(query_ngrams & prefix_ngrams) / max(len(query_ngrams), 1)
            if query_ngrams
            else 0.0
        )

        content_tokens = [
            token
            for token in prefix_token_list
            if len(token) > 2 and token not in QUERY_FOCUS_STOPWORDS
        ]
        novelty_tokens = {
            token for token in content_tokens if token not in query_tokens
        }
        novelty_ratio = len(novelty_tokens) / max(len(set(content_tokens)), 1)
        return exact_prefix, overlap, ngram_overlap, novelty_ratio, len(novelty_tokens)

    def _query_overlap_anomaly_penalty(
        self,
        exact_prefix: bool,
        overlap: float,
        ngram_overlap: float,
        novelty_ratio: float,
        novelty_token_count: int,
    ) -> float:
        if self.query_echo_mode in {"legacy", "query_echo"}:
            if exact_prefix:
                return self.query_echo_exact_prefix_penalty
            if overlap >= self.query_echo_overlap_threshold:
                return self.query_echo_high_overlap_penalty
            return 0.0

        low_novelty = (
            novelty_token_count < self.query_echo_min_novelty_tokens
            or novelty_ratio < self.query_echo_min_novelty_ratio
        )
        if exact_prefix:
            return self.query_echo_exact_prefix_penalty

        penalty = 0.0
        if overlap >= self.query_echo_overlap_threshold:
            penalty = max(penalty, self.query_echo_high_overlap_penalty)
        if ngram_overlap >= self.query_echo_ngram_threshold:
            penalty = max(penalty, self.query_echo_high_overlap_penalty)
        if overlap >= max(0.75, self.query_echo_overlap_threshold - 0.15) and ngram_overlap >= 0.25:
            penalty = max(penalty, self.query_echo_high_overlap_penalty * 0.5)

        if penalty <= 0.0:
            return 0.0
        if low_novelty:
            return penalty * 1.5
        # Keep this as a retrieval-bait signal rather than a correctness signal:
        # independent evidence avoids the extra penalty, but excessive copying
        # still receives a small downweight.
        return penalty

    def _injection_risk_penalty(
        self,
        text: str,
        lexical_penalty: float,
    ) -> Tuple[float, List[str]]:
        signals: List[str] = []
        effective_penalty = lexical_penalty if self.injection_query_copy_enabled else 0.0
        if effective_penalty > 0.0:
            signals.append("query_copy")
        if not self.injection_risk_enabled or not self.injection_textual_cues_enabled:
            return effective_penalty, signals

        doc_norm = normalize_text(text)
        if not doc_norm:
            return effective_penalty, signals

        matched_assertion = any(cue in doc_norm for cue in self.injection_assertion_cues)
        matched_instruction = any(cue in doc_norm for cue in self.injection_instruction_cues)
        candidate_penalty = 0.0
        if matched_assertion:
            signals.append("answer_assertion")
            candidate_penalty = max(candidate_penalty, self.injection_assertion_penalty)
        if matched_instruction:
            signals.append("answer_instruction")
            candidate_penalty = max(candidate_penalty, self.injection_instruction_penalty)
        if candidate_penalty > 0.0 and len(doc_norm.split()) <= self.injection_short_doc_tokens:
            signals.append("short_assertion_context")
            candidate_penalty += self.injection_short_doc_extra_penalty

        # Max-combination avoids increasing penalties on existing query-copy
        # attacks while adding coverage for non-copy answer insertions.
        return max(effective_penalty, candidate_penalty), signals

    def _entry_has_untrusted_provenance(self, entry: CandidateEntry) -> bool:
        provenance = normalize_text(
            entry.result.metadata.get(self.injection_provenance_field, "")
        )
        return bool(provenance and provenance in self.injection_untrusted_provenance)

    def _apply_provenance_gate(
        self,
        entries: List[CandidateEntry],
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "enabled": self.provenance_gate_enabled,
            "mode": self.provenance_gate_mode,
            "input_count": len(entries),
            "eligible_count": len(entries),
            "trusted_count": 0,
            "quarantined_count": 0,
            "untrusted_quarantined_count": 0,
            "unknown_quarantined_count": 0,
            "quarantined_doc_ids": [],
        }
        if not self.provenance_gate_enabled:
            return entries, diagnostics

        eligible: List[CandidateEntry] = []
        quarantined: List[CandidateEntry] = []
        for entry in entries:
            provenance = normalize_text(
                entry.result.metadata.get(self.provenance_gate_field, "")
            )
            if provenance in self.provenance_gate_trusted_provenance:
                diagnostics["trusted_count"] += 1
                eligible.append(entry)
                continue
            if provenance in self.provenance_gate_untrusted_provenance:
                diagnostics["untrusted_quarantined_count"] += 1
                quarantined.append(entry)
                continue
            if self.provenance_gate_unknown_policy == "quarantine":
                diagnostics["unknown_quarantined_count"] += 1
                quarantined.append(entry)
            else:
                eligible.append(entry)

        diagnostics["eligible_count"] = len(eligible)
        diagnostics["quarantined_count"] = len(quarantined)
        diagnostics["quarantined_doc_ids"] = [
            entry.result.doc_id for entry in quarantined[:20]
        ]
        return eligible, diagnostics

    def _apply_untrusted_claim_risk(self, entries: List[CandidateEntry]) -> None:
        if not self.injection_provenance_enabled:
            return

        trusted_answer_keys: Set[str] = set()
        for entry in entries:
            if self._entry_has_untrusted_provenance(entry):
                continue
            trusted_answer_keys.update(
                mention.key
                for mention in self._mentions_for_scoring(entry)
                if mention.key
            )

        for entry in entries:
            if not self._entry_has_untrusted_provenance(entry):
                continue
            proposed_keys = {
                mention.key
                for mention in self._mentions_for_scoring(entry)
                if mention.key
            }
            if not proposed_keys:
                continue
            if proposed_keys & trusted_answer_keys:
                entry.injection_risk_signals.append("untrusted_corroborated")
                continue
            entry.injection_risk_signals.append("untrusted_uncorroborated_claim")
            entry.injection_risk_penalty = max(
                entry.injection_risk_penalty,
                self.injection_untrusted_claim_penalty,
            )
            entry.query_echo_penalty = entry.injection_risk_penalty

    def _topic_grounding_stats(
        self,
        query: str,
        text: str,
    ) -> Tuple[int, float, int, bool, float]:
        query_tokens = query_content_tokens(query)
        doc_tokens = [
            token
            for token in normalize_text(text).split()
            if len(token) > 2 and token not in QUERY_FOCUS_STOPWORDS
        ]
        if not query_tokens or not doc_tokens:
            return 0, 0.0, len(doc_tokens), False, 0.0

        doc_token_set = set(doc_tokens)
        overlap = len(query_tokens & doc_token_set)
        coverage = overlap / max(len(query_tokens), 1)
        short_doc = len(doc_tokens) <= self.topic_grounding_short_doc_token_threshold
        low_grounding = (
            overlap < self.topic_grounding_min_overlap
            and coverage < self.topic_grounding_min_query_coverage
        )
        if short_doc and overlap < max(1, self.topic_grounding_min_overlap):
            low_grounding = True

        penalty = 0.0
        if low_grounding:
            penalty += self.topic_grounding_low_support_penalty
            if short_doc:
                penalty += self.topic_grounding_short_doc_penalty
        return overlap, coverage, len(doc_tokens), low_grounding, penalty

    def _populate_query_echo(
        self,
        entries: List[CandidateEntry],
        query: str,
        force: bool = False,
    ) -> None:
        if not self.query_echo_enabled and not force:
            return
        for entry in entries:
            (
                exact_prefix,
                overlap,
                ngram_overlap,
                novelty_ratio,
                novelty_token_count,
            ) = self._query_echo_stats(query, entry.result.text)
            lexical_penalty = self._query_overlap_anomaly_penalty(
                exact_prefix=exact_prefix,
                overlap=overlap,
                ngram_overlap=ngram_overlap,
                novelty_ratio=novelty_ratio,
                novelty_token_count=novelty_token_count,
            )
            penalty, risk_signals = self._injection_risk_penalty(
                entry.result.text,
                lexical_penalty,
            )
            entry.query_echo_exact_prefix = exact_prefix
            entry.query_echo_overlap = overlap
            entry.query_echo_ngram_overlap = ngram_overlap
            entry.query_echo_novelty_ratio = novelty_ratio
            entry.query_echo_novelty_token_count = novelty_token_count
            entry.query_echo_lexical_penalty = lexical_penalty
            entry.query_echo_penalty = penalty
            entry.injection_risk_penalty = penalty
            entry.injection_risk_signals = risk_signals
        self._apply_untrusted_claim_risk(entries)

    def _populate_topic_grounding(
        self,
        entries: List[CandidateEntry],
        query: str,
        force: bool = False,
    ) -> None:
        if not self.topic_grounding_enabled and not force:
            return
        for entry in entries:
            (
                overlap,
                coverage,
                content_token_count,
                low_grounding,
                penalty,
            ) = self._topic_grounding_stats(query, entry.result.text)
            entry.topic_grounding_overlap = overlap
            entry.topic_grounding_query_coverage = coverage
            entry.topic_grounding_content_token_count = content_token_count
            entry.topic_grounding_low = low_grounding
            entry.topic_grounding_penalty = penalty

    def _answer_extraction_coverage(
        self,
        entries: List[CandidateEntry],
        answer_support: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        mentioned_doc_ids = {
            entry.result.doc_id
            for entry in entries
            if self._mentions_for_scoring(entry)
        }
        mention_count = sum(
            len(self._mentions_for_scoring(entry))
            for entry in entries
        )
        doc_count = len(entries)
        coverage = len(mentioned_doc_ids) / max(doc_count, 1)
        max_answer_doc_count = max(
            [len(item["doc_ids"]) for item in answer_support.values()] or [0]
        )
        max_answer_cluster_count = max(
            [len(item["cluster_ids"]) for item in answer_support.values()] or [0]
        )
        return {
            "doc_count": doc_count,
            "mentioned_doc_count": len(mentioned_doc_ids),
            "doc_coverage": coverage,
            "mention_count": mention_count,
            "unique_answer_count": len(answer_support),
            "max_answer_doc_count": max_answer_doc_count,
            "max_answer_cluster_count": max_answer_cluster_count,
            "min_doc_coverage": self.evidence_fallback_min_answer_coverage,
            "min_answer_count": self.evidence_fallback_min_answer_count,
        }

    def _should_use_evidence_mode(self, coverage: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.answer_enabled:
            return False, "answer_support_disabled"
        if not self.evidence_fallback_enabled:
            return False, "fallback_disabled"
        if coverage["doc_count"] <= 0:
            return False, "empty_candidates"
        if coverage["doc_coverage"] < self.evidence_fallback_min_answer_coverage:
            return True, "low_answer_doc_coverage"
        if coverage["unique_answer_count"] < self.evidence_fallback_min_answer_count:
            return True, "too_few_unique_answers"
        return False, "answer_support_sufficient"

    def _score_entries_evidence_mode(
        self,
        entries: List[CandidateEntry],
    ) -> None:
        cluster_sizes: Dict[int, int] = defaultdict(int)
        for entry in entries:
            cluster_sizes[entry.cluster_id] += 1

        for entry in entries:
            score = 1.0 / (self.rank_k + entry.original_rank)
            if self.evidence_mode_use_topic_grounding:
                score += self.evidence_mode_grounding_bonus * min(
                    entry.topic_grounding_overlap,
                    3,
                )
                score += (
                    self.evidence_mode_query_coverage_bonus
                    * entry.topic_grounding_query_coverage
                )
                if entry.topic_grounding_low:
                    score -= self.evidence_mode_low_grounding_penalty
                    if (
                        entry.topic_grounding_content_token_count
                        <= self.topic_grounding_short_doc_token_threshold
                    ):
                        score -= self.evidence_mode_short_doc_penalty
            if cluster_sizes[entry.cluster_id] <= 1:
                score += self.evidence_mode_singleton_cluster_bonus
            else:
                score -= self.duplicate_cluster_penalty * (
                    cluster_sizes[entry.cluster_id] - 1
                )
            score += self.evidence_mode_channel_bonus * max(
                0,
                len(self._entry_channel_names(entry)) - 1,
            )
            if self.evidence_mode_use_query_echo:
                score -= entry.query_echo_penalty
                if entry.query_echo_penalty > 0:
                    score -= self.evidence_mode_query_echo_penalty
            entry.hardening_score = score

    def _score_entries(
        self,
        entries: List[CandidateEntry],
        answer_support: Dict[str, Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        coverage = self._answer_extraction_coverage(entries, answer_support)
        use_evidence_mode, evidence_mode_reason = self._should_use_evidence_mode(coverage)

        self._populate_query_echo(
            entries,
            query,
            force=use_evidence_mode and self.evidence_mode_use_query_echo,
        )
        self._populate_topic_grounding(
            entries,
            query,
            force=use_evidence_mode and self.evidence_mode_use_topic_grounding,
        )

        diagnostics = {
            "mode": self.answer_scoring_mode,
            "evidence_fallback_enabled": self.evidence_fallback_enabled,
            "evidence_fallback_triggered": use_evidence_mode,
            "evidence_fallback_reason": evidence_mode_reason,
            "answer_extraction_coverage": coverage,
        }

        if use_evidence_mode:
            self._score_entries_evidence_mode(entries)
            diagnostics["mode"] = "evidence_support"
            diagnostics["evidence_support"] = {
                "use_query_echo": self.evidence_mode_use_query_echo,
                "use_topic_grounding": self.evidence_mode_use_topic_grounding,
                "grounding_bonus": self.evidence_mode_grounding_bonus,
                "query_coverage_bonus": self.evidence_mode_query_coverage_bonus,
                "singleton_cluster_bonus": self.evidence_mode_singleton_cluster_bonus,
                "channel_bonus": self.evidence_mode_channel_bonus,
                "query_echo_penalty": self.evidence_mode_query_echo_penalty,
                "low_grounding_penalty": self.evidence_mode_low_grounding_penalty,
            }
            return diagnostics

        if self.answer_scoring_mode in {"robust", "robust_support", "qa_robust"}:
            self._score_entries_robust(entries, answer_support)
            return diagnostics

        scoring_support = (
            self._build_answer_support(entries, grounded_only=True)
            if self.topic_grounding_enabled and self.topic_grounding_block_support_bonus
            else answer_support
        )
        conflict_answer_count = sum(
            1 for item in scoring_support.values() if len(item["doc_ids"]) > 0
        )
        max_support_clusters = max(
            [len(item["cluster_ids"]) for item in scoring_support.values()] or [0]
        )

        for entry in entries:
            score = 1.0 / (self.rank_k + entry.original_rank)
            for mention in self._mentions_for_scoring(entry):
                support = scoring_support.get(mention.key)
                if not support:
                    continue
                doc_count = len(support["doc_ids"])
                cluster_count = len(support["cluster_ids"])
                score += self.diversity_bonus * min(
                    cluster_count,
                    self.max_answer_bonus_clusters,
                )
                if doc_count > cluster_count:
                    score -= self.duplicate_cluster_penalty * (doc_count - cluster_count)
                if (
                    self.contradiction_enabled
                    and conflict_answer_count >= self.min_conflict_answers
                    and max_support_clusters > cluster_count
                ):
                    score -= self.conflict_penalty * (max_support_clusters - cluster_count)
            if self.query_echo_enabled:
                score -= entry.query_echo_penalty
            if self.topic_grounding_enabled:
                score -= entry.topic_grounding_penalty
            entry.hardening_score = score
        return diagnostics

    def _robust_answer_support_stats(
        self,
        entries: List[CandidateEntry],
    ) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            for mention in self._mentions_for_scoring(entry):
                item = stats.setdefault(
                    mention.key,
                    {
                        "doc_ids": set(),
                        "cluster_ids": set(),
                        "non_echo_cluster_ids": set(),
                        "grounded_cluster_ids": set(),
                        "channel_names": set(),
                        "query_echo_doc_ids": set(),
                        "low_grounding_doc_ids": set(),
                    },
                )
                item["doc_ids"].add(entry.result.doc_id)
                item["cluster_ids"].add(entry.cluster_id)
                item["channel_names"].update(self._entry_channel_names(entry))
                if entry.topic_grounding_low:
                    item["low_grounding_doc_ids"].add(entry.result.doc_id)
                else:
                    item["grounded_cluster_ids"].add(entry.cluster_id)
                if entry.query_echo_penalty > 0 or entry.topic_grounding_low:
                    item["query_echo_doc_ids"].add(entry.result.doc_id)
                else:
                    item["non_echo_cluster_ids"].add(entry.cluster_id)
        return stats

    def _score_entries_robust(
        self,
        entries: List[CandidateEntry],
        answer_support: Dict[str, Dict[str, Any]],
    ) -> None:
        robust_support = self._robust_answer_support_stats(entries)
        conflict_answer_count = sum(
            1 for item in robust_support.values() if len(item["doc_ids"]) > 0
        )
        max_non_echo_clusters = max(
            [len(item["non_echo_cluster_ids"]) for item in robust_support.values()] or [0]
        )

        for entry in entries:
            score = 1.0 / (self.rank_k + entry.original_rank)
            seen_answers: Set[str] = set()
            for mention in self._mentions_for_scoring(entry):
                if mention.key in seen_answers:
                    continue
                seen_answers.add(mention.key)
                support = robust_support.get(mention.key) or answer_support.get(mention.key)
                if not support:
                    continue
                doc_count = len(support["doc_ids"])
                cluster_count = len(support["cluster_ids"])
                non_echo_cluster_count = len(support.get("non_echo_cluster_ids", set()))
                channel_count = len(support.get("channel_names", set()))
                echo_doc_count = len(support.get("query_echo_doc_ids", set()))
                low_grounding_doc_count = len(support.get("low_grounding_doc_ids", set()))

                if non_echo_cluster_count > 0 and not entry.topic_grounding_low:
                    score += self.robust_non_echo_bonus * min(
                        non_echo_cluster_count,
                        self.max_answer_bonus_clusters,
                    )
                    score += self.robust_channel_bonus * max(0, channel_count - 1)
                else:
                    score -= self.robust_query_echo_support_penalty

                if cluster_count <= 1:
                    score -= self.robust_isolated_answer_penalty
                if doc_count > cluster_count:
                    score -= self.duplicate_cluster_penalty * (doc_count - cluster_count)
                if echo_doc_count:
                    score -= self.robust_query_echo_support_penalty * min(echo_doc_count, 2)
                if low_grounding_doc_count:
                    score -= self.topic_grounding_low_support_penalty * min(
                        low_grounding_doc_count,
                        2,
                    )
                if (
                    self.contradiction_enabled
                    and conflict_answer_count >= self.min_conflict_answers
                    and max_non_echo_clusters > non_echo_cluster_count
                ):
                    score -= self.conflict_penalty * (
                        max_non_echo_clusters - non_echo_cluster_count
                    )
            if self.query_echo_enabled:
                score -= entry.query_echo_penalty
            if self.topic_grounding_enabled:
                score -= entry.topic_grounding_penalty
            entry.hardening_score = score

    def _answer_cluster_count(
        self,
        answer_key_value: Optional[str],
        answer_support: Dict[str, Dict[str, Any]],
    ) -> int:
        if not answer_key_value:
            return 0
        support = answer_support.get(answer_key_value)
        if not support:
            return 0
        return len(support["cluster_ids"])

    def _apply_rank_guard(
        self,
        ranked: List[CandidateEntry],
        answer_support: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        diagnostics = {
            "enabled": self.rank_guard_enabled,
            "blocked_doc_ids": [],
            "original_head_doc_ids": [],
            "final_head_doc_ids": [],
        }
        if not self.rank_guard_enabled or not ranked:
            diagnostics["final_head_doc_ids"] = [
                entry.result.doc_id for entry in ranked[: self.rank_guard_top_n]
            ]
            return ranked, diagnostics

        original_head = sorted(ranked, key=lambda entry: entry.original_rank)[
            : self.rank_guard_top_n
        ]
        diagnostics["original_head_doc_ids"] = [
            entry.result.doc_id for entry in original_head
        ]
        guarded = list(ranked)
        protected_prefix: List[CandidateEntry] = []
        blocked_doc_ids: List[str] = []

        for position, original_entry in enumerate(original_head):
            if position >= len(guarded):
                break
            current_entry = guarded[position]
            if current_entry.result.doc_id == original_entry.result.doc_id:
                protected_prefix.append(current_entry)
                continue

            primary_answer = self._primary_answer_key(current_entry)
            answer_clusters = self._answer_cluster_count(primary_answer, answer_support)
            promotion_margin = current_entry.hardening_score - original_entry.hardening_score
            should_block = (
                promotion_margin < self.rank_guard_min_promotion_margin
                or current_entry.original_rank > self.rank_guard_max_promotion_original_rank
                or answer_clusters < self.rank_guard_min_answer_clusters
            )
            if should_block:
                current_entry.rank_guard_blocked = True
                blocked_doc_ids.append(current_entry.result.doc_id)
                guarded = [
                    original_entry,
                    *[
                        entry
                        for entry in guarded
                        if entry.result.doc_id != original_entry.result.doc_id
                    ],
                ]
                protected_prefix.append(original_entry)
            else:
                protected_prefix.append(current_entry)

        diagnostics["blocked_doc_ids"] = blocked_doc_ids
        diagnostics["final_head_doc_ids"] = [
            entry.result.doc_id for entry in guarded[: self.rank_guard_top_n]
        ]
        return guarded, diagnostics

    def _primary_answer_key(self, entry: CandidateEntry) -> Optional[str]:
        mentions = self._mentions_for_scoring(entry)
        return mentions[0].key if mentions else None

    def _apply_caps(
        self,
        ranked: List[CandidateEntry],
        top_k: int,
    ) -> Tuple[List[CandidateEntry], List[str], List[str]]:
        accepted: List[CandidateEntry] = []
        accepted_per_cluster: Dict[int, int] = defaultdict(int)
        accepted_per_answer: Dict[str, int] = defaultdict(int)
        cluster_filtered_doc_ids: List[str] = []
        answer_filtered_doc_ids: List[str] = []
        for entry in ranked:
            if (
                self.cluster_enabled
                and self.cluster_cap > 0
                and accepted_per_cluster[entry.cluster_id] >= self.cluster_cap
            ):
                cluster_filtered_doc_ids.append(entry.result.doc_id)
                continue
            primary_answer = self._primary_answer_key(entry)
            if (
                self.answer_enabled
                and self.max_docs_per_answer > 0
                and primary_answer
                and accepted_per_answer[primary_answer] >= self.max_docs_per_answer
            ):
                answer_filtered_doc_ids.append(entry.result.doc_id)
                continue
            accepted_per_cluster[entry.cluster_id] += 1
            if primary_answer:
                accepted_per_answer[primary_answer] += 1
            accepted.append(entry)
            if len(accepted) >= top_k:
                break
        return accepted, cluster_filtered_doc_ids, answer_filtered_doc_ids

    def _is_high_risk_query(self, query: str) -> bool:
        query_l = query.lower()
        return any(keyword in query_l for keyword in self.head_high_risk_keywords)

    def _consequence_policy_for_query(self, query: str) -> Dict[str, Any]:
        if not self.consequence_policy_enabled or self.consequence_kb is None:
            return {
                "enabled": False,
                "risk_tier": "not_applied",
                "risk_weight": None,
                "risk_category": None,
                "risk_source": None,
                "risk_rationale": None,
                "required_policy": {},
                "matched_consequence_ids": [],
            }
        route = self.consequence_kb.route(query)
        return {
            "enabled": True,
            "risk_tier": route["risk_tier"],
            "risk_weight": route["risk_weight"],
            "risk_category": route["risk_category"],
            "risk_source": route["risk_source"],
            "risk_rationale": route["risk_rationale"],
            "required_policy": route["required_policy"],
            "matched_consequence_ids": [
                entry["consequence_id"] for entry in route.get("matched_entries", [])
            ],
        }

    def _apply_consequence_authority_gate(
        self,
        entries: List[CandidateEntry],
        policy: Dict[str, Any],
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        require_authority = bool(
            policy.get("required_policy", {}).get("require_authoritative_source")
        )
        enabled = bool(
            self.consequence_policy_enabled
            and self.consequence_policy_enforce_authoritative_source
            and require_authority
        )
        diagnostics = {
            "enabled": enabled,
            "required_by_policy": require_authority,
            "risk_tier": policy.get("risk_tier"),
            "input_count": len(entries),
            "eligible_count": len(entries),
            "quarantined_count": 0,
            "authoritative_provenance": sorted(
                self.consequence_policy_authoritative_provenance
            ),
            "quarantined_doc_ids": [],
        }
        if not enabled:
            return entries, diagnostics
        accepted = []
        quarantined = []
        for entry in entries:
            provenance = normalize_text(
                entry.result.metadata.get(self.consequence_policy_provenance_field, "")
            )
            if provenance in self.consequence_policy_authoritative_provenance:
                accepted.append(entry)
            else:
                quarantined.append(entry.result.doc_id)
        diagnostics["eligible_count"] = len(accepted)
        diagnostics["quarantined_count"] = len(quarantined)
        diagnostics["quarantined_doc_ids"] = quarantined[:20]
        return accepted, diagnostics

    def _head_answer_stats(
        self,
        entries: List[CandidateEntry],
        head_k: int,
    ) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries, start=1):
            answer = self._primary_answer_key(entry)
            if not answer:
                continue
            item = stats.setdefault(
                answer,
                {
                    "doc_ids": set(),
                    "cluster_ids": set(),
                    "ranks": [],
                    "head_count": 0,
                    "supplement_count": 0,
                },
            )
            item["doc_ids"].add(entry.result.doc_id)
            item["cluster_ids"].add(entry.cluster_id)
            item["ranks"].append(idx)
            if idx <= head_k:
                item["head_count"] += 1
            else:
                item["supplement_count"] += 1
        return stats

    def _primary_answer_stats(
        self,
        entries: List[CandidateEntry],
    ) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries, start=1):
            answer = self._primary_answer_key(entry)
            if not answer:
                continue
            mentions = self._mentions_for_scoring(entry)
            label = mentions[0].label if mentions else answer
            item = stats.setdefault(
                answer,
                {
                    "label": label,
                    "doc_ids": set(),
                    "cluster_ids": set(),
                    "non_echo_cluster_ids": set(),
                    "grounded_cluster_ids": set(),
                    "channel_names": set(),
                    "query_echo_doc_ids": set(),
                    "low_grounding_doc_ids": set(),
                    "ranks": [],
                    "scores": [],
                },
            )
            item["doc_ids"].add(entry.result.doc_id)
            item["cluster_ids"].add(entry.cluster_id)
            item["channel_names"].update(self._entry_channel_names(entry))
            item["ranks"].append(idx)
            item["scores"].append(float(entry.hardening_score))
            if entry.topic_grounding_low:
                item["low_grounding_doc_ids"].add(entry.result.doc_id)
            else:
                item["grounded_cluster_ids"].add(entry.cluster_id)
            if entry.query_echo_penalty > 0 or entry.topic_grounding_low:
                item["query_echo_doc_ids"].add(entry.result.doc_id)
            else:
                item["non_echo_cluster_ids"].add(entry.cluster_id)
        return stats

    def _serializable_primary_answer_stats(
        self,
        stats: Dict[str, Dict[str, Any]],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for answer, item in stats.items():
            rows.append(
                {
                    "answer_key": answer,
                    "label": item["label"],
                    "doc_count": len(item["doc_ids"]),
                    "cluster_count": len(item["cluster_ids"]),
                    "non_echo_cluster_count": len(item["non_echo_cluster_ids"]),
                    "grounded_cluster_count": len(item["grounded_cluster_ids"]),
                    "channel_count": len(item["channel_names"]),
                    "query_echo_doc_count": len(item["query_echo_doc_ids"]),
                    "low_grounding_doc_count": len(item["low_grounding_doc_ids"]),
                    "ranks": list(item["ranks"]),
                    "best_score": max(item["scores"]) if item["scores"] else 0.0,
                    "doc_ids": sorted(item["doc_ids"])[:10],
                }
            )
        rows.sort(
            key=lambda row: (
                row["non_echo_cluster_count"],
                row["cluster_count"],
                row["channel_count"],
                row["doc_count"],
                -min(row["ranks"] or [9999]),
            ),
            reverse=True,
        )
        return rows[:limit]

    def _margin_answer_scores(
        self,
        stats: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        max_independent_clusters = 0
        rows: List[Dict[str, Any]] = []
        for answer, item in stats.items():
            doc_count = len(item["doc_ids"])
            cluster_count = len(item["cluster_ids"])
            non_echo_cluster_count = len(item["non_echo_cluster_ids"])
            independent_clusters = (
                non_echo_cluster_count
                if self.margin_gate_use_non_echo_clusters and non_echo_cluster_count > 0
                else cluster_count
            )
            max_independent_clusters = max(max_independent_clusters, independent_clusters)
            channel_count = len(item["channel_names"])
            echo_ratio = len(item["query_echo_doc_ids"]) / max(doc_count, 1)
            isolated = cluster_count <= 1
            ranks = list(item["ranks"])
            scores = list(item.get("scores", []))
            rows.append(
                {
                    "answer_key": answer,
                    "label": item["label"],
                    "doc_count": doc_count,
                    "cluster_count": cluster_count,
                    "non_echo_cluster_count": non_echo_cluster_count,
                    "independent_cluster_count": independent_clusters,
                    "channel_count": channel_count,
                    "query_echo_doc_count": len(item["query_echo_doc_ids"]),
                    "query_echo_ratio": echo_ratio,
                    "isolated": isolated,
                    "ranks": ranks,
                    "best_rank": min(ranks or [9999]),
                    "best_doc_score": max(scores) if scores else 0.0,
                    "doc_ids": sorted(item["doc_ids"])[:10],
                }
            )

        for row in rows:
            conflict_gap = max(0, max_independent_clusters - row["independent_cluster_count"])
            row["support_score"] = (
                self.margin_gate_cluster_weight * row["independent_cluster_count"]
                + self.margin_gate_channel_weight * max(0, row["channel_count"] - 1)
                - self.margin_gate_echo_penalty * row["query_echo_ratio"]
                - self.margin_gate_isolated_penalty * int(row["isolated"])
                - self.margin_gate_conflict_penalty * conflict_gap
            )
        rows.sort(
            key=lambda row: (
                row["support_score"],
                row["independent_cluster_count"],
                row["cluster_count"],
                row["channel_count"],
                -min(row["ranks"] or [9999]),
            ),
            reverse=True,
        )
        return rows

    def _margin_gate_complex_context(
        self,
        score_rows: List[Dict[str, Any]],
        query: str,
        consequence_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        top_row = score_rows[0]
        runner_up = score_rows[1] if len(score_rows) > 1 else None
        routed_tier = str((consequence_policy or {}).get("risk_tier") or "")
        routed_high_risk = routed_tier == "high"
        keyword_high_risk = self._is_high_risk_query(query)
        high_risk_query = (
            routed_high_risk
            if self.consequence_policy_enabled and self.consequence_policy_override_keyword_risk
            else routed_high_risk or keyword_high_risk
        )
        strong_answers = [
            row
            for row in score_rows
            if int(row["independent_cluster_count"])
            >= self.margin_gate_strong_alternative_min_clusters
        ]
        top_weak = (
            int(top_row["independent_cluster_count"])
            <= self.margin_gate_weak_independent_clusters
            and int(top_row["channel_count"]) <= self.margin_gate_weak_channel_count
        )
        top_echo_heavy = (
            float(top_row["query_echo_ratio"]) >= self.margin_gate_echo_ratio_threshold
            or int(top_row["query_echo_doc_count"]) >= int(top_row["doc_count"])
        )
        top_isolated = bool(top_row["isolated"])
        runner_supported = bool(
            runner_up
            and int(runner_up["independent_cluster_count"])
            >= self.margin_gate_strong_alternative_min_clusters
        )
        multi_supported_conflict = len(strong_answers) >= 2
        no_strong_answer = len(strong_answers) == 0
        top_not_best_independent = bool(
            runner_up
            and int(runner_up["independent_cluster_count"])
            > int(top_row["independent_cluster_count"])
        )

        if multi_supported_conflict:
            conflict_type = "multi_supported_conflict"
        elif top_not_best_independent:
            conflict_type = "top_not_best_support"
        elif top_weak and runner_supported:
            conflict_type = "weak_top_with_supported_alternative"
        elif no_strong_answer:
            conflict_type = "no_strong_answer"
        elif top_echo_heavy:
            conflict_type = "echo_heavy_top_answer"
        elif top_isolated and len(score_rows) >= self.margin_gate_min_conflict_answers:
            conflict_type = "isolated_top_answer"
        else:
            conflict_type = "weak_conflict"

        effective_threshold = self.margin_gate_threshold
        threshold_reasons: List[str] = []
        if self.margin_gate_dynamic_threshold:
            if top_weak or top_isolated:
                effective_threshold += self.margin_gate_weak_top_delta
                threshold_reasons.append("weak_or_isolated_top")
            if top_echo_heavy:
                effective_threshold += self.margin_gate_echo_top_delta
                threshold_reasons.append("echo_heavy_top")
            if multi_supported_conflict:
                effective_threshold += self.margin_gate_multi_supported_delta
                threshold_reasons.append("multi_supported_conflict")
            if no_strong_answer:
                effective_threshold += self.margin_gate_no_strong_answer_delta
                threshold_reasons.append("no_strong_answer")
            if high_risk_query:
                effective_threshold += self.margin_gate_high_risk_delta
                threshold_reasons.append("high_risk_query")
        if self.consequence_policy_use_margin_threshold and consequence_policy:
            required_threshold = consequence_policy.get("required_policy", {}).get(
                "min_support_margin"
            )
            if required_threshold is not None and float(required_threshold) > effective_threshold:
                effective_threshold = float(required_threshold)
                threshold_reasons.append(f"consequence_policy_{routed_tier}")
        effective_threshold = min(effective_threshold, self.margin_gate_max_threshold)

        return {
            "mode": self.margin_gate_mode,
            "effective_threshold": effective_threshold,
            "threshold_reasons": threshold_reasons,
            "conflict_type": conflict_type,
            "top_answer_weak": bool(top_weak or top_isolated or top_echo_heavy),
            "top_answer_isolated": top_isolated,
            "top_answer_echo_heavy": top_echo_heavy,
            "runner_up_supported": runner_supported,
            "multi_supported_conflict": multi_supported_conflict,
            "no_strong_answer": no_strong_answer,
            "high_risk_query": high_risk_query,
            "consequence_policy": consequence_policy or {"enabled": False},
        }

    def _margin_gate_alternative_answers(
        self,
        score_rows: List[Dict[str, Any]],
        top_row: Dict[str, Any],
        effective_threshold: float,
    ) -> Set[str]:
        alternatives: List[Tuple[Tuple[float, int, int, int], str]] = []
        for row in score_rows[1:]:
            score_gap = float(top_row["support_score"]) - float(row["support_score"])
            if score_gap >= effective_threshold:
                continue
            rank_key = (
                float(row["support_score"]),
                int(row["independent_cluster_count"]),
                int(row["channel_count"]),
                -int(row["best_rank"]),
            )
            alternatives.append((rank_key, str(row["answer_key"])))
        alternatives.sort(reverse=True)
        if self.margin_gate_max_alternatives > 0:
            alternatives = alternatives[: self.margin_gate_max_alternatives]
        return {answer for _, answer in alternatives}

    def _apply_margin_gate(
        self,
        accepted: List[CandidateEntry],
        ranked: List[CandidateEntry],
        query: str,
        top_k: int,
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        consequence_policy = self._consequence_policy_for_query(query)
        diagnostics: Dict[str, Any] = {
            "enabled": self.margin_gate_enabled,
            "mode": self.margin_gate_mode,
            "triggered": False,
            "order_changed": False,
            "low_margin": False,
            "threshold": self.margin_gate_threshold,
            "effective_threshold": self.margin_gate_threshold,
            "threshold_reasons": [],
            "margin": None,
            "top_answer": None,
            "runner_up_answer": None,
            "conflict_type": None,
            "top_answer_weak": False,
            "top_answer_isolated": False,
            "top_answer_echo_heavy": False,
            "runner_up_supported": False,
            "multi_supported_conflict": False,
            "no_strong_answer": False,
            "high_risk_query": False,
            "uncertain_recommended": False,
            "consequence_policy": consequence_policy,
            "answer_scores": [],
            "penalized_doc_ids": [],
            "boosted_doc_ids": [],
            "supplement_promoted_doc_ids": [],
            "post_cap_filtered_by_cluster_doc_ids": [],
            "post_cap_filtered_by_answer_doc_ids": [],
            "reranked_doc_ids": [],
        }
        if not self.margin_gate_enabled or len(accepted) <= 1:
            return accepted[:top_k], diagnostics

        pool_limit = max(top_k, self.margin_gate_window)
        pool: List[CandidateEntry] = []
        seen_doc_ids: Set[str] = set()
        for entry in [*accepted, *ranked]:
            if entry.result.doc_id in seen_doc_ids:
                continue
            pool.append(entry)
            seen_doc_ids.add(entry.result.doc_id)
            if len(pool) >= pool_limit:
                break

        stats = self._primary_answer_stats(pool)
        score_rows = self._margin_answer_scores(stats)
        diagnostics["answer_scores"] = score_rows[:12]
        if len(score_rows) < self.margin_gate_min_conflict_answers:
            return accepted[:top_k], diagnostics

        top_row = score_rows[0]
        runner_up = score_rows[1]
        margin = float(top_row["support_score"] - runner_up["support_score"])
        complex_context = self._margin_gate_complex_context(
            score_rows,
            query,
            consequence_policy=consequence_policy,
        )
        effective_threshold = float(complex_context["effective_threshold"])
        diagnostics["margin"] = margin
        diagnostics["top_answer"] = top_row["answer_key"]
        diagnostics["runner_up_answer"] = runner_up["answer_key"]
        diagnostics["effective_threshold"] = effective_threshold
        diagnostics["threshold_reasons"] = complex_context["threshold_reasons"]
        diagnostics["conflict_type"] = complex_context["conflict_type"]
        diagnostics["top_answer_weak"] = complex_context["top_answer_weak"]
        diagnostics["top_answer_isolated"] = complex_context["top_answer_isolated"]
        diagnostics["top_answer_echo_heavy"] = complex_context["top_answer_echo_heavy"]
        diagnostics["runner_up_supported"] = complex_context["runner_up_supported"]
        diagnostics["multi_supported_conflict"] = complex_context["multi_supported_conflict"]
        diagnostics["no_strong_answer"] = complex_context["no_strong_answer"]
        diagnostics["high_risk_query"] = complex_context["high_risk_query"]
        diagnostics["low_margin"] = margin < effective_threshold
        diagnostics["fallback_answered"] = False
        if margin >= effective_threshold:
            return accepted[:top_k], diagnostics

        # Fallback answering: if the low-margin top answer carries no poison
        # signature, trust it and answer rather than penalising into abstention.
        if (
            self.margin_gate_fallback_when_clean
            and not complex_context["top_answer_echo_heavy"]
            and not complex_context["top_answer_isolated"]
            and not complex_context["no_strong_answer"]
        ):
            diagnostics["fallback_answered"] = True
            return accepted[:top_k], diagnostics

        top_answer = str(top_row["answer_key"])
        alternative_answers = self._margin_gate_alternative_answers(
            score_rows,
            top_row,
            effective_threshold,
        )
        diagnostics["alternative_answers"] = sorted(alternative_answers)

        should_penalize_top = True
        if self.margin_gate_penalize_top_only_if_weak:
            should_penalize_top = bool(
                complex_context["top_answer_weak"]
                or complex_context["multi_supported_conflict"]
                or complex_context["no_strong_answer"]
            )
        if (
            self.margin_gate_preserve_rank1_if_no_alternative
            and not alternative_answers
        ):
            should_penalize_top = False

        adjusted: List[Tuple[float, int, CandidateEntry]] = []
        accepted_doc_ids = {entry.result.doc_id for entry in accepted}
        penalized_doc_ids: List[str] = []
        boosted_doc_ids: List[str] = []
        supplement_doc_ids: List[str] = []
        for idx, entry in enumerate(pool, start=1):
            score = entry.hardening_score
            answer = self._primary_answer_key(entry)
            if answer == top_answer and should_penalize_top:
                penalty = self.margin_gate_top_answer_penalty * self.margin_gate_top_penalty_multiplier
                if self.margin_gate_mode == "complex":
                    if entry.query_echo_penalty > 0:
                        penalty += self.margin_gate_echo_doc_extra_penalty
                    if entry.topic_grounding_low:
                        penalty += self.margin_gate_low_grounding_extra_penalty
                score -= penalty
                entry.hardening_score -= penalty
                entry.margin_gate_adjusted = True
                entry.margin_gate_penalty += penalty
                penalized_doc_ids.append(entry.result.doc_id)
            elif answer in alternative_answers:
                bonus = (
                    self.margin_gate_alternative_bonus
                    * self.margin_gate_alternative_bonus_multiplier
                )
                is_supplement = entry.result.doc_id not in accepted_doc_ids
                supplement_rank_allowed = (
                    self.margin_gate_min_supplement_rank
                    <= entry.original_rank
                    <= self.margin_gate_max_supplement_rank
                )
                if is_supplement and supplement_rank_allowed:
                    bonus += (
                        self.margin_gate_supplement_bonus
                        * self.margin_gate_supplement_bonus_multiplier
                    )
                    entry.margin_gate_supplement_promoted = True
                    supplement_doc_ids.append(entry.result.doc_id)
                score += bonus
                entry.hardening_score += bonus
                entry.margin_gate_adjusted = True
                entry.margin_gate_bonus += bonus
                boosted_doc_ids.append(entry.result.doc_id)
            adjusted.append((score, -idx, entry))

        adjusted.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reranked = [item[2] for item in adjusted]
        gated, cluster_filtered, answer_filtered = self._apply_caps(reranked, top_k)
        original_ids = [entry.result.doc_id for entry in accepted[:top_k]]
        gated_ids = [entry.result.doc_id for entry in gated]
        diagnostics["triggered"] = True
        diagnostics["order_changed"] = original_ids != gated_ids
        diagnostics["penalized_doc_ids"] = penalized_doc_ids
        diagnostics["boosted_doc_ids"] = boosted_doc_ids
        diagnostics["supplement_promoted_doc_ids"] = supplement_doc_ids
        diagnostics["post_cap_filtered_by_cluster_doc_ids"] = cluster_filtered[:20]
        diagnostics["post_cap_filtered_by_answer_doc_ids"] = answer_filtered[:20]
        diagnostics["reranked_doc_ids"] = gated_ids
        diagnostics["uncertain_recommended"] = bool(
            complex_context["high_risk_query"]
            and complex_context["no_strong_answer"]
            and self.margin_gate_high_risk_abstain
        )
        return gated[:top_k], diagnostics

    def _apply_top1_dominance_guard(
        self,
        accepted: List[CandidateEntry],
        top_k: int,
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "enabled": self.top1_dominance_enabled,
            "triggered": False,
            "order_changed": False,
            "top1_doc_id": accepted[0].result.doc_id if accepted else None,
            "top1_answer": self._primary_answer_key(accepted[0]) if accepted else None,
            "conflict_detected": False,
            "top1_isolated": False,
            "supported_alternative_answers": [],
            "penalized_doc_ids": [],
            "promoted_doc_ids": [],
        }
        if not self.top1_dominance_enabled or len(accepted) <= 1:
            return accepted[:top_k], diagnostics

        window = min(len(accepted), top_k, max(2, self.top1_dominance_window))
        candidates = list(accepted[:window])
        stats = self._primary_answer_stats(candidates)
        distinct_answers = sorted(stats)
        conflict_detected = (
            len(distinct_answers) >= self.top1_dominance_min_conflict_answers
        )
        diagnostics["conflict_detected"] = conflict_detected
        diagnostics["answer_stats"] = self._serializable_primary_answer_stats(stats)
        if not conflict_detected:
            return accepted[:top_k], diagnostics

        top1 = candidates[0]
        top1_answer = self._primary_answer_key(top1)
        if not top1_answer or top1_answer not in stats:
            return accepted[:top_k], diagnostics

        top1_stat = stats[top1_answer]
        top1_cluster_count = len(top1_stat["cluster_ids"])
        top1_non_echo_clusters = len(top1_stat["non_echo_cluster_ids"])
        top1_isolated = (
            top1_cluster_count <= self.top1_dominance_isolated_cluster_threshold
            or top1_non_echo_clusters == 0
        )
        diagnostics["top1_cluster_count"] = top1_cluster_count
        diagnostics["top1_non_echo_cluster_count"] = top1_non_echo_clusters
        diagnostics["top1_isolated"] = top1_isolated
        if not top1_isolated:
            return accepted[:top_k], diagnostics

        supported_alternatives = {
            answer
            for answer, item in stats.items()
            if answer != top1_answer
            and len(item["non_echo_cluster_ids"])
            >= self.top1_dominance_alternative_cluster_threshold
        }
        if not supported_alternatives:
            supported_alternatives = {
                answer
                for answer, item in stats.items()
                if answer != top1_answer
                and len(item["cluster_ids"])
                >= self.top1_dominance_alternative_cluster_threshold
            }
        diagnostics["supported_alternative_answers"] = sorted(supported_alternatives)

        penalty = self.top1_dominance_penalty
        if top1.query_echo_penalty > 0:
            penalty += self.top1_dominance_query_echo_extra_penalty
        top1.hardening_score -= penalty
        top1.top1_dominance_adjusted = True
        top1.top1_dominance_penalty += penalty
        diagnostics["penalized_doc_ids"] = [top1.result.doc_id]
        diagnostics["triggered"] = True

        adjusted: List[Tuple[float, int, CandidateEntry]] = []
        for idx, entry in enumerate(candidates, start=1):
            score = entry.hardening_score
            answer = self._primary_answer_key(entry)
            if answer in supported_alternatives:
                score += self.top1_dominance_supported_bonus
            adjusted.append((score, -idx, entry))
        adjusted.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reranked = [item[2] for item in adjusted]

        if self.top1_dominance_max_supplement_rank > 0:
            max_idx = max(0, self.top1_dominance_max_supplement_rank - 1)
            bounded: List[CandidateEntry] = []
            delayed: List[CandidateEntry] = []
            original_head_ids = {entry.result.doc_id for entry in candidates[:3]}
            for idx, entry in enumerate(reranked):
                is_supplement = entry.result.doc_id not in original_head_ids
                if idx < max_idx and is_supplement:
                    delayed.append(entry)
                else:
                    bounded.append(entry)
            insert_at = min(max_idx, len(bounded))
            reranked = bounded[:insert_at] + delayed + bounded[insert_at:]

        original_ids = [entry.result.doc_id for entry in candidates]
        reranked_ids = [entry.result.doc_id for entry in reranked]
        diagnostics["order_changed"] = original_ids != reranked_ids
        diagnostics["reranked_doc_ids"] = reranked_ids
        diagnostics["promoted_doc_ids"] = [
            doc_id
            for doc_id in reranked_ids[:3]
            if doc_id not in set(original_ids[:3])
        ]

        untouched_tail = [
            entry
            for entry in accepted
            if entry.result.doc_id not in set(reranked_ids)
        ]
        return (reranked + untouched_tail)[:top_k], diagnostics

    @staticmethod
    def _serializable_head_answer_stats(
        stats: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for answer, item in stats.items():
            rows.append(
                {
                    "answer_key": answer,
                    "doc_count": len(item["doc_ids"]),
                    "cluster_count": len(item["cluster_ids"]),
                    "head_count": int(item["head_count"]),
                    "supplement_count": int(item["supplement_count"]),
                    "ranks": list(item["ranks"]),
                }
            )
        rows.sort(
            key=lambda row: (
                row["cluster_count"],
                row["doc_count"],
                -min(row["ranks"] or [9999]),
            ),
            reverse=True,
        )
        return rows

    def _apply_head_filter(
        self,
        accepted: List[CandidateEntry],
        query: str,
        top_k: int,
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        head_k = min(self.head_k, len(accepted), top_k)
        supplement_limit = min(len(accepted), head_k + self.head_supplement_k, top_k)
        candidates = list(accepted[:supplement_limit])
        head_entries = candidates[:head_k]
        head_answers = [self._primary_answer_key(entry) for entry in head_entries]
        distinct_head_answers = sorted({answer for answer in head_answers if answer})
        conflict_detected = len(distinct_head_answers) >= self.head_min_conflict_answers

        diagnostics: Dict[str, Any] = {
            "enabled": True,
            "triggered": False,
            "mode": self.head_filter_mode,
            "head_k": head_k,
            "supplement_k": self.head_supplement_k,
            "head_answers": distinct_head_answers,
            "conflict_detected": conflict_detected,
            "reranked_doc_ids": [],
            "isolated_answer_doc_ids": [],
            "supplement_promoted_doc_ids": [],
            "rank1_protected": False,
            "uncertain_recommended": False,
            "high_risk_query": self._is_high_risk_query(query),
        }
        if not conflict_detected:
            diagnostics["answer_stats"] = []
            return accepted[:top_k], diagnostics

        stats = self._head_answer_stats(candidates, head_k=head_k)
        diagnostics["answer_stats"] = self._serializable_head_answer_stats(stats)
        supported_answers = {
            answer
            for answer, item in stats.items()
            if len(item["cluster_ids"]) >= self.head_independent_support_threshold
        }
        severe_conflict = not supported_answers
        diagnostics["supported_answers"] = sorted(supported_answers)
        diagnostics["severe_conflict"] = severe_conflict

        adjusted: List[Tuple[float, int, CandidateEntry, Dict[str, Any]]] = []
        isolated_doc_ids: List[str] = []
        promoted_doc_ids: List[str] = []
        for idx, entry in enumerate(candidates, start=1):
            answer = self._primary_answer_key(entry)
            score = entry.hardening_score
            flags = {
                "isolated_answer_penalized": False,
                "supported_answer_boosted": False,
                "supplement_promoted": False,
            }
            if answer and answer in stats:
                cluster_count = len(stats[answer]["cluster_ids"])
                protect_entry = bool(
                    self.head_filter_mode == "conservative"
                    and self.head_protect_rank1
                    and idx == 1
                )
                if cluster_count <= 1 and answer in distinct_head_answers and not protect_entry:
                    score -= self.head_isolated_penalty
                    flags["isolated_answer_penalized"] = True
                    isolated_doc_ids.append(entry.result.doc_id)
                elif protect_entry:
                    diagnostics["rank1_protected"] = True

                allow_positive_support = self.head_filter_mode != "conservative"
                answer_was_in_head = answer in distinct_head_answers
                allow_supplement_answer = (
                    self.head_allow_supplement_new_answers or answer_was_in_head
                )
                if answer in supported_answers and allow_positive_support:
                    score += self.head_supported_answer_bonus * min(
                        cluster_count,
                        self.head_independent_support_threshold,
                    )
                    flags["supported_answer_boosted"] = True
                    if idx > head_k and allow_supplement_answer:
                        score += self.head_supplement_bonus
                        flags["supplement_promoted"] = True
                        promoted_doc_ids.append(entry.result.doc_id)
            adjusted.append((score, -idx, entry, flags))

        adjusted.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reranked = [item[2] for item in adjusted]
        if self.head_filter_mode == "conservative":
            original_rank1 = candidates[0] if candidates else None
            if original_rank1 is not None:
                reranked = [
                    original_rank1,
                    *[
                        entry
                        for entry in reranked
                        if entry.result.doc_id != original_rank1.result.doc_id
                    ],
                ]
            if self.head_max_supplement_promotion_rank > 0:
                original_head_ids = {
                    entry.result.doc_id for entry in candidates[:head_k]
                }
                floor_idx = max(0, self.head_max_supplement_promotion_rank - 1)
                protected_prefix = []
                delayed_supplements = []
                for idx, entry in enumerate(reranked):
                    is_supplement = entry.result.doc_id not in original_head_ids
                    if idx < floor_idx and is_supplement:
                        delayed_supplements.append(entry)
                    else:
                        protected_prefix.append(entry)
                insert_at = min(floor_idx, len(protected_prefix))
                reranked = (
                    protected_prefix[:insert_at]
                    + delayed_supplements
                    + protected_prefix[insert_at:]
                )
        original_ids = [entry.result.doc_id for entry in candidates]
        reranked_ids = [entry.result.doc_id for entry in reranked]
        diagnostics["triggered"] = True
        diagnostics["reranked_doc_ids"] = reranked_ids
        diagnostics["isolated_answer_doc_ids"] = isolated_doc_ids
        diagnostics["supplement_promoted_doc_ids"] = promoted_doc_ids
        diagnostics["order_changed"] = original_ids != reranked_ids
        diagnostics["uncertain_recommended"] = (
            severe_conflict
            and diagnostics["high_risk_query"]
            and self.head_high_risk_abstain
        )

        untouched_tail = [
            entry
            for entry in accepted
            if entry.result.doc_id not in set(reranked_ids)
        ]
        output = (reranked + untouched_tail)[:top_k]
        return output, diagnostics

    def _constrained_answer_support_score(
        self,
        answer: Optional[str],
        answer_rows: Dict[str, Dict[str, Any]],
    ) -> float:
        if not answer or answer not in answer_rows:
            return 0.0
        row = answer_rows[answer]
        independent_clusters = int(row.get("independent_cluster_count", 0))
        channel_count = int(row.get("channel_count", 0))
        isolated = bool(row.get("isolated", False))
        score = (
            self.constrained_selection_support_bonus * independent_clusters
            + self.constrained_selection_channel_bonus * max(0, channel_count - 1)
        )
        if isolated:
            score -= self.constrained_selection_low_margin_penalty
        return score

    def _constrained_marginal_gain(
        self,
        entry: CandidateEntry,
        selected: List[CandidateEntry],
        answer_rows: Dict[str, Dict[str, Any]],
        selected_per_cluster: Dict[int, int],
        selected_per_answer: Dict[str, int],
        selected_answer_clusters: Dict[str, Set[int]],
        query_overlap_count: int,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        answer = self._primary_answer_key(entry)
        gain = float(entry.hardening_score)
        gain += self._constrained_answer_support_score(answer, answer_rows)

        cluster_count = selected_per_cluster.get(entry.cluster_id, 0)
        if cluster_count > 0:
            gain -= self.constrained_selection_cluster_penalty * cluster_count
            reasons.append("duplicate_cluster_penalty")

        if answer:
            answer_count = selected_per_answer.get(answer, 0)
            if answer_count > 0:
                gain -= self.constrained_selection_answer_penalty * answer_count
                reasons.append("answer_concentration_penalty")
            known_clusters = selected_answer_clusters.setdefault(answer, set())
            if entry.cluster_id not in known_clusters and entry.query_echo_penalty <= 0:
                gain += self.constrained_selection_new_cluster_bonus
                reasons.append("new_answer_cluster_bonus")

        if entry.query_echo_penalty > 0:
            if query_overlap_count >= self.constrained_selection_max_query_overlap_docs:
                gain -= self.constrained_selection_query_overlap_penalty * (
                    query_overlap_count - self.constrained_selection_max_query_overlap_docs + 1
                )
                reasons.append("query_overlap_concentration_penalty")

        if not selected and answer:
            row = answer_rows.get(answer, {})
            if bool(row.get("isolated", False)) and len(answer_rows) >= self.min_conflict_answers:
                gain -= self.constrained_selection_isolated_rank1_penalty
                reasons.append("isolated_rank1_penalty")

        return gain, reasons

    def _apply_constrained_selection(
        self,
        accepted: List[CandidateEntry],
        ranked: List[CandidateEntry],
        top_k: int,
    ) -> Tuple[List[CandidateEntry], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "enabled": self.constrained_selection_enabled,
            "triggered": False,
            "order_changed": False,
            "pool_count": 0,
            "selected_doc_ids": [],
            "original_doc_ids": [entry.result.doc_id for entry in accepted[:top_k]],
            "skipped_doc_ids": [],
            "fallback_filled_doc_ids": [],
            "query_overlap_selected_count": 0,
            "duplicate_cluster_selected_count": 0,
            "selection_steps": [],
        }
        if not self.constrained_selection_enabled or len(accepted) <= 1:
            diagnostics["selected_doc_ids"] = diagnostics["original_doc_ids"]
            return accepted[:top_k], diagnostics

        pool_limit = max(top_k, self.constrained_selection_pool_depth)
        pool: List[CandidateEntry] = []
        seen_doc_ids: Set[str] = set()
        for entry in [*accepted, *ranked]:
            if entry.result.doc_id in seen_doc_ids:
                continue
            pool.append(entry)
            seen_doc_ids.add(entry.result.doc_id)
            if len(pool) >= pool_limit:
                break
        diagnostics["pool_count"] = len(pool)

        answer_rows = {
            str(row["answer_key"]): row
            for row in self._margin_answer_scores(self._primary_answer_stats(pool))
        }
        remaining = list(pool)
        selected: List[CandidateEntry] = []
        selected_per_cluster: Dict[int, int] = defaultdict(int)
        selected_per_answer: Dict[str, int] = defaultdict(int)
        selected_answer_clusters: Dict[str, Set[int]] = defaultdict(set)
        query_overlap_count = 0
        skipped_doc_ids: List[str] = []

        while remaining and len(selected) < top_k:
            scored: List[Tuple[float, int, CandidateEntry, List[str]]] = []
            for idx, entry in enumerate(remaining):
                answer = self._primary_answer_key(entry)
                if (
                    self.constrained_selection_max_cluster_docs > 0
                    and selected_per_cluster.get(entry.cluster_id, 0)
                    >= self.constrained_selection_max_cluster_docs
                ):
                    skipped_doc_ids.append(entry.result.doc_id)
                    continue
                if (
                    answer
                    and self.constrained_selection_max_answer_docs > 0
                    and selected_per_answer.get(answer, 0)
                    >= self.constrained_selection_max_answer_docs
                ):
                    skipped_doc_ids.append(entry.result.doc_id)
                    continue
                gain, reasons = self._constrained_marginal_gain(
                    entry=entry,
                    selected=selected,
                    answer_rows=answer_rows,
                    selected_per_cluster=selected_per_cluster,
                    selected_per_answer=selected_per_answer,
                    selected_answer_clusters=selected_answer_clusters,
                    query_overlap_count=query_overlap_count,
                )
                if gain < self.constrained_selection_min_gain:
                    skipped_doc_ids.append(entry.result.doc_id)
                    continue
                scored.append((gain, -idx, entry, reasons))

            if not scored:
                break
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            gain, _, chosen, reasons = scored[0]
            selected.append(chosen)
            selected_per_cluster[chosen.cluster_id] += 1
            answer = self._primary_answer_key(chosen)
            if answer:
                selected_per_answer[answer] += 1
                selected_answer_clusters[answer].add(chosen.cluster_id)
            if chosen.query_echo_penalty > 0:
                query_overlap_count += 1
            remaining = [
                entry for entry in remaining if entry.result.doc_id != chosen.result.doc_id
            ]
            diagnostics["selection_steps"].append(
                {
                    "doc_id": chosen.result.doc_id,
                    "gain": gain,
                    "answer": answer,
                    "cluster_id": chosen.cluster_id,
                    "query_overlap": bool(chosen.query_echo_penalty > 0),
                    "reasons": reasons,
                }
            )

        if len(selected) < top_k:
            selected_ids = {entry.result.doc_id for entry in selected}
            for entry in pool:
                if entry.result.doc_id in selected_ids:
                    continue
                selected.append(entry)
                diagnostics["fallback_filled_doc_ids"].append(entry.result.doc_id)
                if len(selected) >= top_k:
                    break

        selected = selected[:top_k]
        selected_ids = [entry.result.doc_id for entry in selected]
        diagnostics["triggered"] = True
        diagnostics["selected_doc_ids"] = selected_ids
        diagnostics["skipped_doc_ids"] = sorted(set(skipped_doc_ids))[:20]
        diagnostics["order_changed"] = diagnostics["original_doc_ids"] != selected_ids
        diagnostics["query_overlap_selected_count"] = sum(
            1 for entry in selected if entry.query_echo_penalty > 0
        )
        cluster_counts: Dict[int, int] = defaultdict(int)
        for entry in selected:
            cluster_counts[entry.cluster_id] += 1
        diagnostics["duplicate_cluster_selected_count"] = sum(
            max(0, count - 1) for count in cluster_counts.values()
        )
        return selected, diagnostics

    @staticmethod
    def _serializable_answer_support(
        support: Dict[str, Dict[str, Any]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        rows = []
        for key, item in support.items():
            rows.append(
                {
                    "answer_key": key,
                    "label": item["label"],
                    "doc_count": len(item["doc_ids"]),
                    "cluster_count": len(item["cluster_ids"]),
                    "mention_count": int(item["mentions"]),
                }
            )
        rows.sort(key=lambda row: (row["cluster_count"], row["doc_count"]), reverse=True)
        return rows[:limit]

    def _answer_level_contradiction_summary(
        self,
        candidates: List[CandidateEntry],
        accepted: List[CandidateEntry],
        limit: int = 12,
    ) -> Dict[str, Any]:
        candidate_stats = self._primary_answer_stats(candidates)
        accepted_stats = self._primary_answer_stats(accepted)
        top1_answer = self._primary_answer_key(accepted[0]) if accepted else None
        accepted_answers = sorted(accepted_stats)
        candidate_answers = sorted(candidate_stats)

        rows = self._serializable_primary_answer_stats(candidate_stats, limit=limit)
        selected_rows = self._serializable_primary_answer_stats(
            accepted_stats,
            limit=max(limit, len(accepted_stats)),
        )
        accepted_rows = {
            row["answer_key"]: row
            for row in selected_rows
        }
        for row in rows:
            accepted_row = accepted_rows.get(row["answer_key"])
            row["selected_doc_count"] = (
                int(accepted_row["doc_count"]) if accepted_row else 0
            )
            row["selected_cluster_count"] = (
                int(accepted_row["cluster_count"]) if accepted_row else 0
            )

        top1_stats = accepted_stats.get(top1_answer) if top1_answer else None
        top1_isolated = False
        top1_doc_count = 0
        top1_cluster_count = 0
        top1_non_echo_cluster_count = 0
        top1_channel_count = 0
        top1_query_echo_doc_count = 0
        if top1_stats:
            top1_doc_count = len(top1_stats["doc_ids"])
            top1_cluster_count = len(top1_stats["cluster_ids"])
            top1_non_echo_cluster_count = len(top1_stats["non_echo_cluster_ids"])
            top1_channel_count = len(top1_stats["channel_names"])
            top1_query_echo_doc_count = len(top1_stats["query_echo_doc_ids"])
            top1_isolated = top1_cluster_count <= 1
        supported_answers = [
            answer
            for answer, item in accepted_stats.items()
            if len(item["non_echo_cluster_ids"]) >= 2 or len(item["cluster_ids"]) >= 2
        ]
        candidate_supported_answers = [
            answer
            for answer, item in candidate_stats.items()
            if len(item["non_echo_cluster_ids"]) >= 2 or len(item["cluster_ids"]) >= 2
        ]
        isolated_selected_answers = [
            answer
            for answer, item in accepted_stats.items()
            if len(item["cluster_ids"]) <= 1
        ]
        query_echo_only_answers = [
            answer
            for answer, item in accepted_stats.items()
            if len(item["doc_ids"]) > 0 and len(item["non_echo_cluster_ids"]) == 0
        ]
        selected_max_non_echo_cluster_count = max(
            [len(item["non_echo_cluster_ids"]) for item in accepted_stats.values()]
            or [0]
        )
        selected_max_cluster_count = max(
            [len(item["cluster_ids"]) for item in accepted_stats.values()] or [0]
        )
        selected_max_channel_count = max(
            [len(item["channel_names"]) for item in accepted_stats.values()] or [0]
        )
        alternative_non_echo_counts = [
            len(item["non_echo_cluster_ids"])
            for answer, item in accepted_stats.items()
            if answer != top1_answer
        ]
        alternative_cluster_counts = [
            len(item["cluster_ids"])
            for answer, item in accepted_stats.items()
            if answer != top1_answer
        ]
        best_alternative_non_echo_cluster_count = max(alternative_non_echo_counts or [0])
        best_alternative_cluster_count = max(alternative_cluster_counts or [0])
        top1_has_best_support = False
        if top1_answer:
            top1_has_best_support = (
                top1_non_echo_cluster_count >= selected_max_non_echo_cluster_count
                and top1_cluster_count >= selected_max_cluster_count
            )
        support_margin = best_alternative_non_echo_cluster_count - top1_non_echo_cluster_count
        if best_alternative_non_echo_cluster_count == top1_non_echo_cluster_count:
            support_margin = best_alternative_cluster_count - top1_cluster_count

        conflict_detected = len(accepted_answers) >= self.min_conflict_answers
        severe_conflict = conflict_detected and not supported_answers
        no_strong_answer = conflict_detected and not supported_answers
        top1_isolated_with_alternative = (
            conflict_detected
            and top1_isolated
            and (
                best_alternative_non_echo_cluster_count >= 2
                or best_alternative_cluster_count >= 2
            )
        )
        top1_not_best_support = conflict_detected and bool(top1_answer) and not top1_has_best_support
        multi_supported_conflict = conflict_detected and len(supported_answers) >= 2
        if not conflict_detected:
            conflict_type = "none"
        elif top1_isolated_with_alternative:
            conflict_type = "top1_isolated_with_alternative"
        elif top1_not_best_support:
            conflict_type = "top1_not_best_support"
        elif multi_supported_conflict:
            conflict_type = "multi_supported_conflict"
        elif no_strong_answer:
            conflict_type = "no_strong_answer"
        else:
            conflict_type = "weak_conflict"

        return {
            "candidate_answer_count": len(candidate_answers),
            "selected_answer_count": len(accepted_answers),
            "conflict_detected": conflict_detected,
            "candidate_conflict_detected": len(candidate_answers) >= self.min_conflict_answers,
            "top1_answer": top1_answer,
            "top1_isolated": top1_isolated,
            "top1_doc_count": top1_doc_count,
            "top1_cluster_count": top1_cluster_count,
            "top1_non_echo_cluster_count": top1_non_echo_cluster_count,
            "top1_channel_count": top1_channel_count,
            "top1_query_echo_doc_count": top1_query_echo_doc_count,
            "top1_has_best_support": top1_has_best_support,
            "top1_support_margin": support_margin,
            "supported_answers": sorted(supported_answers),
            "supported_answer_count": len(supported_answers),
            "candidate_supported_answer_count": len(candidate_supported_answers),
            "isolated_selected_answer_count": len(isolated_selected_answers),
            "query_echo_only_answer_count": len(query_echo_only_answers),
            "selected_max_non_echo_cluster_count": selected_max_non_echo_cluster_count,
            "selected_max_cluster_count": selected_max_cluster_count,
            "selected_max_channel_count": selected_max_channel_count,
            "best_alternative_non_echo_cluster_count": best_alternative_non_echo_cluster_count,
            "best_alternative_cluster_count": best_alternative_cluster_count,
            "top1_isolated_with_alternative": top1_isolated_with_alternative,
            "top1_not_best_support": top1_not_best_support,
            "multi_supported_conflict": multi_supported_conflict,
            "no_strong_answer": no_strong_answer,
            "severe_conflict": severe_conflict,
            "conflict_type": conflict_type,
            "answers": rows,
            "selected_answers": selected_rows,
        }

    def harden(
        self,
        results: Sequence[SearchResult],
        query: str,
        top_k: int,
        reference_answers: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        consequence_policy = self._consequence_policy_for_query(query)
        entries = [
            CandidateEntry(
                result=result,
                original_rank=idx,
                original_score=float(result.score),
            )
            for idx, result in enumerate(results, start=1)
        ]
        entries, provenance_gate_diagnostics = self._apply_provenance_gate(entries)
        entries, consequence_authority_gate_diagnostics = self._apply_consequence_authority_gate(
            entries,
            consequence_policy,
        )
        clusters = self._assign_clusters(entries)
        self._attach_answer_mentions(entries, query, reference_answers)
        answer_support = self._build_answer_support(entries)
        reference_support = self._build_reference_support(entries)
        scoring_diagnostics = self._score_entries(entries, answer_support, query=query)

        ranked = sorted(
            entries,
            key=lambda entry: (entry.hardening_score, -entry.original_rank),
            reverse=True,
        )
        ranked, rank_guard_diagnostics = self._apply_rank_guard(
            ranked,
            answer_support,
        )
        accepted, cluster_filtered_doc_ids, answer_filtered_doc_ids = self._apply_caps(
            ranked,
            top_k,
        )
        accepted, top1_dominance_diagnostics = self._apply_top1_dominance_guard(
            accepted,
            top_k,
        )
        accepted, margin_gate_diagnostics = self._apply_margin_gate(
            accepted,
            ranked,
            query,
            top_k,
        )
        head_filter_diagnostics = {"enabled": False}
        if self.head_filter_enabled:
            accepted, head_filter_diagnostics = self._apply_head_filter(
                accepted,
                query=query,
                top_k=top_k,
            )
        accepted, constrained_selection_diagnostics = self._apply_constrained_selection(
            accepted,
            ranked,
            top_k,
        )
        answer_level_summary = self._answer_level_contradiction_summary(
            ranked,
            accepted,
        )

        output: List[SearchResult] = []
        for rank, entry in enumerate(accepted, start=1):
            metadata = dict(entry.result.metadata)
            metadata["evidence_hardening"] = {
                "original_rank": entry.original_rank,
                "original_score": entry.original_score,
                "hardening_score": entry.hardening_score,
                "scoring_mode": scoring_diagnostics.get("mode"),
                "cluster_id": entry.cluster_id,
                "cluster_size": clusters.get(entry.cluster_id, {}).get("size", 1),
                "query_echo": {
                    "enabled": self.query_echo_enabled,
                    "computed_for_evidence_fallback": bool(
                        scoring_diagnostics.get("evidence_fallback_triggered")
                        and self.evidence_mode_use_query_echo
                    ),
                    "mode": self.query_echo_mode,
                    "exact_prefix": entry.query_echo_exact_prefix,
                    "overlap": entry.query_echo_overlap,
                    "ngram_overlap": entry.query_echo_ngram_overlap,
                    "novelty_ratio": entry.query_echo_novelty_ratio,
                    "novelty_token_count": entry.query_echo_novelty_token_count,
                    "lexical_penalty": entry.query_echo_lexical_penalty,
                    "penalty": entry.query_echo_penalty,
                },
                "injection_risk": {
                    "enabled": self.injection_risk_enabled,
                    "query_copy_enabled": self.injection_query_copy_enabled,
                    "textual_cues_enabled": self.injection_textual_cues_enabled,
                    "provenance_enabled": self.injection_provenance_enabled,
                    "provenance": entry.result.metadata.get(self.injection_provenance_field),
                    "signals": entry.injection_risk_signals,
                    "penalty": entry.injection_risk_penalty,
                },
                "provenance_gate": {
                    "enabled": self.provenance_gate_enabled,
                    "mode": self.provenance_gate_mode,
                    "provenance": entry.result.metadata.get(self.provenance_gate_field),
                    "trusted_for_context": (
                        not self.provenance_gate_enabled
                        or normalize_text(
                            entry.result.metadata.get(self.provenance_gate_field, "")
                        )
                        in self.provenance_gate_trusted_provenance
                        or self.provenance_gate_unknown_policy == "allow"
                    ),
                },
                "topic_grounding": {
                    "enabled": self.topic_grounding_enabled,
                    "computed_for_evidence_fallback": bool(
                        scoring_diagnostics.get("evidence_fallback_triggered")
                        and self.evidence_mode_use_topic_grounding
                    ),
                    "overlap": entry.topic_grounding_overlap,
                    "query_coverage": entry.topic_grounding_query_coverage,
                    "content_token_count": entry.topic_grounding_content_token_count,
                    "low_grounding": entry.topic_grounding_low,
                    "penalty": entry.topic_grounding_penalty,
                },
                "rank_guard_blocked": entry.rank_guard_blocked,
                "evidence_support": {
                    "fallback_triggered": bool(
                        scoring_diagnostics.get("evidence_fallback_triggered")
                    ),
                    "reason": scoring_diagnostics.get("evidence_fallback_reason"),
                    "used_query_echo": bool(
                        scoring_diagnostics.get("evidence_fallback_triggered")
                        and self.evidence_mode_use_query_echo
                    ),
                    "used_topic_grounding": bool(
                        scoring_diagnostics.get("evidence_fallback_triggered")
                        and self.evidence_mode_use_topic_grounding
                    ),
                },
                "top1_dominance_adjusted": entry.top1_dominance_adjusted,
                "top1_dominance_penalty": entry.top1_dominance_penalty,
                "margin_gate": {
                    "enabled": bool(margin_gate_diagnostics.get("enabled")),
                    "mode": margin_gate_diagnostics.get("mode"),
                    "triggered": bool(margin_gate_diagnostics.get("triggered")),
                    "adjusted": entry.margin_gate_adjusted,
                    "penalty": entry.margin_gate_penalty,
                    "bonus": entry.margin_gate_bonus,
                    "supplement_promoted": entry.margin_gate_supplement_promoted,
                    "effective_threshold": margin_gate_diagnostics.get(
                        "effective_threshold"
                    ),
                    "conflict_type": margin_gate_diagnostics.get("conflict_type"),
                    "uncertain_recommended": bool(
                        margin_gate_diagnostics.get("uncertain_recommended")
                    ),
                },
                "heuristic_answers": [
                    {
                        "key": item.key,
                        "label": item.label,
                        "source": item.source,
                        "score": item.score,
                    }
                    for item in entry.heuristic_mentions
                ],
                "diagnostic_answers": [
                    {
                        "key": item.key,
                        "label": item.label,
                        "source": item.source,
                        "kind": item.kind,
                        "score": item.score,
                    }
                    for item in entry.diagnostic_mentions
                ],
                "reference_answers": [
                    {"key": item.key, "label": item.label, "kind": item.kind}
                    for item in entry.reference_mentions
                ],
                "primary_answer_key": self._primary_answer_key(entry),
                "head_filter": {
                    "enabled": bool(head_filter_diagnostics.get("enabled")),
                    "triggered": bool(head_filter_diagnostics.get("triggered")),
                    "conflict_detected": bool(
                        head_filter_diagnostics.get("conflict_detected")
                    ),
                    "uncertain_recommended": bool(
                        head_filter_diagnostics.get("uncertain_recommended")
                    ),
                },
                "constrained_selection": {
                    "enabled": bool(constrained_selection_diagnostics.get("enabled")),
                    "triggered": bool(
                        constrained_selection_diagnostics.get("triggered")
                    ),
                    "order_changed": bool(
                        constrained_selection_diagnostics.get("order_changed")
                    ),
                    "query_overlap_selected_count": constrained_selection_diagnostics.get(
                        "query_overlap_selected_count"
                    ),
                    "duplicate_cluster_selected_count": constrained_selection_diagnostics.get(
                        "duplicate_cluster_selected_count"
                    ),
                },
            }
            output.append(
                SearchResult(
                    doc_id=entry.result.doc_id,
                    text=entry.result.text,
                    score=float(entry.hardening_score),
                    rank=rank,
                    source=f"{entry.result.source}+evidence_hardening",
                    metadata=metadata,
                )
            )

        diagnostics = {
            "enabled": True,
            "input_count": len(results),
            "eligible_input_count": len(entries),
            "output_count": len(output),
            "filtered_by_cluster_count": len(cluster_filtered_doc_ids),
            "filtered_by_cluster_doc_ids": cluster_filtered_doc_ids[:20],
            "filtered_by_answer_count": len(answer_filtered_doc_ids),
            "filtered_by_answer_doc_ids": answer_filtered_doc_ids[:20],
            "cluster_count": len(clusters),
            "max_cluster_size": max([cluster["size"] for cluster in clusters.values()] or [0]),
            "clusters_with_adv": sum(1 for cluster in clusters.values() if cluster["adv_doc_count"]),
            "heuristic_extractor": {
                "name": self.answer_extractor_name,
                "use_query_focus": self.use_query_focus,
                "focused_sentence_count": self.focused_sentence_count,
                "use_answer_cues": self.use_answer_cues,
                "max_candidate_words": self.max_candidate_words,
                "semantic_model_name": self.semantic_model_name
                if self._uses_semantic_answer_extractor()
                else None,
                "semantic_model_loaded": bool(self.semantic_answer_extractor is not None),
                "semantic_model_error": self.semantic_answer_extractor_error,
                "semantic_candidate_pool": self.semantic_candidate_pool
                if self._uses_semantic_answer_extractor()
                or self._uses_qa_answer_extractor()
                else None,
                "qa_model_name": self.qa_model_name
                if self._uses_qa_answer_extractor()
                else None,
                "qa_model_loaded": bool(self.qa_answer_extractor is not None),
                "qa_model_error": self.qa_answer_extractor_error,
                "qa_include_heuristic": self.qa_include_heuristic
                if self._uses_qa_answer_extractor()
                else None,
                "qa_use_semantic_rerank": self.qa_use_semantic_rerank
                if self._uses_qa_answer_extractor()
                else None,
                "qa_scoring_mode": self.qa_scoring_mode
                if self._uses_qa_answer_extractor()
                else None,
                "answer_scoring_mode": self.answer_scoring_mode,
                "use_reference_answers_for_scoring": self.use_reference_for_scoring,
                "reference_answer_kinds_for_scoring": sorted(
                    self.reference_answer_kinds_for_scoring
                )
                if self.reference_answer_kinds_for_scoring is not None
                else None,
                "robust_channel_bonus": self.robust_channel_bonus
                if self.answer_scoring_mode in {"robust", "robust_support", "qa_robust"}
                else None,
                "topic_grounding_enabled": self.topic_grounding_enabled,
                "topic_grounding_min_query_coverage": self.topic_grounding_min_query_coverage,
                "topic_grounding_min_overlap": self.topic_grounding_min_overlap,
                "topic_grounding_block_support_bonus": self.topic_grounding_block_support_bonus,
                "fallback_to_evidence_mode": self.evidence_fallback_enabled,
                "evidence_mode_min_answer_coverage": self.evidence_fallback_min_answer_coverage,
                "evidence_mode_min_answer_count": self.evidence_fallback_min_answer_count,
            },
            "scoring": scoring_diagnostics,
            "heuristic_answer_count": len(answer_support),
            "answer_support": self._serializable_answer_support(answer_support),
            "reference_answer_support": reference_support,
            "conflict_detected": len(answer_support) >= self.min_conflict_answers,
            "query_echo": {
                "enabled": self.query_echo_enabled,
                "computed_for_evidence_fallback": bool(
                    scoring_diagnostics.get("evidence_fallback_triggered")
                    and self.evidence_mode_use_query_echo
                ),
                "mode": self.query_echo_mode,
                "exact_prefix_count": sum(
                    1 for entry in entries if entry.query_echo_exact_prefix
                ),
                "penalized_count": sum(
                    1 for entry in entries if entry.query_echo_penalty > 0
                ),
                "overlap_mean": (
                    sum(entry.query_echo_overlap for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
                "ngram_overlap_mean": (
                    sum(entry.query_echo_ngram_overlap for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
                "novelty_ratio_mean": (
                    sum(entry.query_echo_novelty_ratio for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
                "penalty_mean": (
                    sum(entry.query_echo_penalty for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
            },
            "injection_risk": {
                "enabled": self.injection_risk_enabled,
                "query_copy_enabled": self.injection_query_copy_enabled,
                "textual_cues_enabled": self.injection_textual_cues_enabled,
                "provenance_enabled": self.injection_provenance_enabled,
                "penalized_count": sum(
                    1 for entry in entries if entry.injection_risk_penalty > 0
                ),
                "answer_assertion_count": sum(
                    1 for entry in entries if "answer_assertion" in entry.injection_risk_signals
                ),
                "answer_instruction_count": sum(
                    1 for entry in entries if "answer_instruction" in entry.injection_risk_signals
                ),
                "query_copy_count": sum(
                    1 for entry in entries if "query_copy" in entry.injection_risk_signals
                ),
                "untrusted_uncorroborated_claim_count": sum(
                    1
                    for entry in entries
                    if "untrusted_uncorroborated_claim" in entry.injection_risk_signals
                ),
                "untrusted_corroborated_count": sum(
                    1
                    for entry in entries
                    if "untrusted_corroborated" in entry.injection_risk_signals
                ),
                "penalty_mean": (
                    sum(entry.injection_risk_penalty for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
            },
            "provenance_gate": provenance_gate_diagnostics,
            "consequence_policy": consequence_policy,
            "consequence_authority_gate": consequence_authority_gate_diagnostics,
            "topic_grounding": {
                "enabled": self.topic_grounding_enabled,
                "computed_for_evidence_fallback": bool(
                    scoring_diagnostics.get("evidence_fallback_triggered")
                    and self.evidence_mode_use_topic_grounding
                ),
                "low_grounding_count": sum(
                    1 for entry in entries if entry.topic_grounding_low
                ),
                "penalty_mean": (
                    sum(entry.topic_grounding_penalty for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
                "coverage_mean": (
                    sum(entry.topic_grounding_query_coverage for entry in entries) / len(entries)
                    if entries
                    else 0.0
                ),
            },
            "rank_guard": rank_guard_diagnostics,
            "top1_dominance": top1_dominance_diagnostics,
            "margin_gate": margin_gate_diagnostics,
            "head_filter": head_filter_diagnostics,
            "constrained_selection": constrained_selection_diagnostics,
            "answer_level_contradiction": answer_level_summary,
        }
        return output, diagnostics

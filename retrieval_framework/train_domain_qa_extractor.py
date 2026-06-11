from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from retrieval_framework.stable_generation import StableLLMGenerator


DEFAULT_EVAL_RESULTS = (
    "retrieval_framework/results/qa_model_swap_100q/qa_roberta_base_100q.json"
)
DEFAULT_ADV_PATH = "results/adv_targeted_results/nq.json"


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["_id"])] = row
    return rows


def read_qrels(path: Path) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    with path.open() as handle:
        header = next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            query_id, doc_id, *_ = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, []).append(doc_id)
    return qrels


def iter_result_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for iteration in payload.get("results", []):
        if not isinstance(iteration, dict):
            continue
        for rows in iteration.values():
            for item in rows or []:
                yield item


def excluded_sets(
    eval_results_path: Path,
    adv_path: Path,
    qrels: Dict[str, List[str]],
) -> Tuple[set[str], set[str]]:
    excluded_queries: set[str] = set()
    excluded_docs: set[str] = set()
    if adv_path.exists():
        adv_data = json.loads(adv_path.read_text())
        excluded_queries.update(str(key) for key in adv_data.keys())
        excluded_queries.update(str(item.get("id")) for item in adv_data.values())
    if eval_results_path.exists():
        payload = json.loads(eval_results_path.read_text())
        for item in iter_result_items(payload):
            qid = str(item.get("id"))
            excluded_queries.add(qid)
            for doc in item.get("retrieved") or []:
                doc_id = str(doc.get("doc_id", ""))
                if doc_id:
                    excluded_docs.add(doc_id)
            for doc in item.get("raw_retrieved") or []:
                doc_id = str(doc.get("doc_id", ""))
                if doc_id:
                    excluded_docs.add(doc_id)
    for qid in excluded_queries:
        excluded_docs.update(qrels.get(qid, []))
    return excluded_queries, excluded_docs


def compact_context(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def normalize_for_output(text: str) -> str:
    text = text.strip().strip("\"'` ")
    text = re.sub(r"^(answer|the answer)\s*:\s*", "", text, flags=re.I).strip()
    text = text.splitlines()[0].strip()
    return text.strip().strip("\"'` ")


def find_span(context: str, answer: str) -> Optional[Tuple[int, int, str]]:
    answer = normalize_for_output(answer)
    if not answer or answer.upper() == "NO_ANSWER":
        return None
    if len(answer.split()) > 12:
        return None
    lower_context = context.lower()
    lower_answer = answer.lower()
    idx = lower_context.find(lower_answer)
    if idx >= 0:
        return idx, idx + len(answer), context[idx : idx + len(answer)]

    compact_answer = re.sub(r"\s+", " ", answer).strip()
    pattern = re.escape(compact_answer).replace(r"\ ", r"\s+")
    match = re.search(pattern, context, flags=re.I)
    if match:
        return match.start(), match.end(), context[match.start() : match.end()]
    return None


def make_label_prompt(question: str, context: str) -> str:
    return (
        "Extract the shortest exact answer span from the context that answers the question. "
        "The answer must be copied verbatim from the context. "
        "If the context does not answer the question, output exactly NO_ANSWER.\n\n"
        f"Question: {question}\n"
        f"Context: {context}\n\n"
        "Answer:"
    )


def make_synthetic_prompt(title: str, context: str) -> str:
    return (
        "Create one extractive QA training example from the context. "
        "The question must be answerable from the context, and the answer must be the shortest "
        "exact span copied verbatim from the context. Prefer concrete entities, dates, counts, "
        "places, titles, or short noun phrases. Keep the answer under 12 words. "
        "Return only JSON with keys question and answer.\n\n"
        f"Title: {title}\n"
        f"Context: {context}\n\n"
        "JSON:"
    )


def parse_synthetic_output(output: str) -> Optional[Tuple[str, str]]:
    text = output.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            question = str(payload.get("question", "")).strip()
            answer = normalize_for_output(str(payload.get("answer", "")))
            if question and answer:
                return question, answer
        except json.JSONDecodeError:
            pass

    question = ""
    answer = ""
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"question", "q"}:
            question = value
        elif key in {"answer", "a"}:
            answer = normalize_for_output(value)
    if question and answer:
        return question, answer
    return None


def first_sentence(text: str) -> str:
    match = re.search(r"(.{40,240}?[.!?])\s", text.strip() + " ")
    return match.group(1).strip() if match else text.strip()[:240]


def short_enough(value: str, max_words: int = 12) -> bool:
    return 0 < len(value.split()) <= max_words and len(value) <= 120


def clean_candidate_answer(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;:.()\"'")
    return value


def template_qa_from_doc(title: str, text: str) -> Optional[Tuple[str, str]]:
    sentence = first_sentence(text)
    title_for_question = title or "the passage"

    date = (
        r"(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},\s+\d{4}"
    )
    patterns = [
        (rf"\bborn\s+(?:on\s+)?({date})", f"When was {title_for_question} born?"),
        (rf"\bpremiered\s+(?:on\s+)?({date})", f"When did {title_for_question} premiere?"),
        (rf"\breleased\s+(?:on\s+)?({date})", f"When was {title_for_question} released?"),
        (rf"\bconcluded\s+(?:on\s+)?({date})", f"When did {title_for_question} conclude?"),
        (r"\bborn\s+(?:in\s+)?(\d{4})", f"What year was {title_for_question} born?"),
        (r"\bfounded\s+(?:in\s+)?(\d{4})", f"What year was {title_for_question} founded?"),
        (r"\bestablished\s+(?:in\s+)?(\d{4})", f"What year was {title_for_question} established?"),
        (r"\breleased\s+(?:in\s+)?(\d{4})", f"What year was {title_for_question} released?"),
        (r"\bpremiered\s+(?:in\s+)?(\d{4})", f"What year did {title_for_question} premiere?"),
        (r"\bcontained\s+(\d+\s+episodes?)", f"How many episodes did {title_for_question} contain?"),
        (r"\bwritten by\s+([^.;]{3,90})", f"Who wrote {title_for_question}?"),
        (r"\bdirected by\s+([^.;]{3,90})", f"Who directed {title_for_question}?"),
        (r"\bproduced by\s+([^.;]{3,90})", f"Who produced {title_for_question}?"),
        (r"\bcreated by\s+([^.;]{3,90})", f"Who created {title_for_question}?"),
        (r"\bstarring\s+([^.;]{3,90})", f"Who starred in {title_for_question}?"),
        (r"\bis an?\s+([^.;]{3,90})", f"What is {title_for_question}?"),
        (r"\bis the\s+([^.;]{3,90})", f"What is {title_for_question}?"),
    ]
    for pattern, question in patterns:
        match = re.search(pattern, sentence, flags=re.I)
        if not match:
            continue
        answer = clean_candidate_answer(match.group(1))
        answer = re.split(r"\s+(?:and|with|during|while|where|which|who)\s+", answer)[0]
        answer = clean_candidate_answer(answer)
        if short_enough(answer):
            return question, answer

    entity_match = re.search(
        r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+(?:of|the|and|de|[A-Z][A-Za-z0-9'&.-]+)){1,5})\b",
        sentence,
    )
    if entity_match:
        answer = clean_candidate_answer(entity_match.group(1))
        if answer != title and short_enough(answer):
            return f"What named entity is mentioned in the passage about {title_for_question}?", answer

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", sentence)
    if year_match:
        return f"What year is mentioned in the passage about {title_for_question}?", year_match.group(1)

    return None


def add_template_synthetic_examples(
    *,
    examples: List[Dict[str, Any]],
    corpus: Dict[str, Dict[str, Any]],
    excluded_docs: set[str],
    used_doc_ids: set[str],
    args: argparse.Namespace,
) -> None:
    corpus_ids = [
        doc_id
        for doc_id, doc in corpus.items()
        if doc_id not in excluded_docs
        and doc_id not in used_doc_ids
        and len(str(doc.get("text", "")).split()) >= args.min_synthetic_doc_words
    ]
    random.Random(args.seed + 2).shuffle(corpus_ids)
    print(
        f"Building template synthetic QA from {len(corpus_ids)} unused docs "
        f"at {len(examples)}/{args.num_labels}"
    )
    for doc_id in corpus_ids:
        if len(examples) >= args.num_labels:
            break
        doc = corpus[doc_id]
        title = str(doc.get("title", "")).strip()
        raw_text = str(doc.get("text", "")).strip()
        generated = template_qa_from_doc(title, raw_text)
        if generated is None:
            continue
        question, answer = generated
        context = compact_context(f"{title}. {raw_text}", args.label_context_chars)
        span = find_span(context, answer)
        if span is None:
            continue
        start, end, exact = span
        examples.append(
            {
                "id": f"template::{doc_id}",
                "query_id": None,
                "doc_id": doc_id,
                "question": question,
                "context": context,
                "answer_text": exact,
                "answer_start": start,
                "answer_end": end,
                "labeler_output": answer,
                "source": "template_corpus",
            }
        )
        used_doc_ids.add(doc_id)
        if len(examples) % 500 == 0:
            print(f"template labeled {len(examples)}/{args.num_labels}")


def build_labeled_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from src.models import create_model

    data_dir = Path(args.data_dir)
    queries = read_jsonl(data_dir / "queries.jsonl")
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    qrels = read_qrels(data_dir / "qrels" / f"{args.split}.tsv")
    excluded_queries, excluded_docs = excluded_sets(
        Path(args.eval_results_path),
        Path(args.adv_path),
        qrels,
    )

    candidates: List[Tuple[str, str]] = []
    for qid in sorted(qrels.keys(), key=lambda value: int(value.replace("test", "")) if value.startswith("test") and value[4:].isdigit() else value):
        if qid in excluded_queries or qid not in queries:
            continue
        added_for_query = 0
        for doc_id in qrels[qid]:
            if doc_id in excluded_docs or doc_id not in corpus:
                continue
            candidates.append((qid, doc_id))
            added_for_query += 1
            if args.max_qrel_docs_per_query > 0 and added_for_query >= args.max_qrel_docs_per_query:
                break
    random.Random(args.seed).shuffle(candidates)

    print(
        f"Excluded queries={len(excluded_queries)}, excluded corpus docs={len(excluded_docs)}, "
        f"candidate qrel pairs={len(candidates)}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "labeled_examples.json"
    if cache_path.exists() and not args.relabel:
        examples = json.loads(cache_path.read_text())
        print(f"Loaded cached labels: {len(examples)} from {cache_path}")
        if len(examples) >= args.num_labels:
            return examples[: args.num_labels]
    else:
        examples = []

    used_ids = {str(example.get("id")) for example in examples}
    used_doc_ids = {str(example.get("doc_id")) for example in examples}
    attempts = 0
    labeler = None
    if not args.skip_llm_qrel:
        print("Loading local Llama3 labeler...")
        llm = create_model(args.llama_config)
        labeler = StableLLMGenerator(
            llm,
            {
                "enabled": True,
                "force_greedy": True,
                "max_new_tokens": args.label_max_new_tokens,
                "remove_invalid_values": True,
                "renormalize_logits": True,
                "suppress_transformers_warnings": True,
            },
        )

        for qid, doc_id in candidates:
            if len(examples) >= args.num_labels:
                break
            example_id = f"{qid}::{doc_id}"
            if example_id in used_ids:
                continue
            attempts += 1
            question = queries[qid]["text"]
            doc = corpus[doc_id]
            context = compact_context(
                f"{doc.get('title', '')}. {doc.get('text', '')}",
                args.label_context_chars,
            )
            generation = labeler.query(make_label_prompt(question, context))
            answer = normalize_for_output(generation["response"])
            span = find_span(context, answer)
            if span is None:
                continue
            start, end, exact = span
            examples.append(
                {
                    "id": f"{qid}::{doc_id}",
                    "query_id": qid,
                    "doc_id": doc_id,
                    "question": question,
                    "context": context,
                    "answer_text": exact,
                    "answer_start": start,
                    "answer_end": end,
                    "labeler_output": answer,
                    "source": "qrel_extract",
                }
            )
            used_ids.add(example_id)
            used_doc_ids.add(doc_id)
            if len(examples) % 20 == 0:
                print(f"labeled {len(examples)}/{args.num_labels} after {attempts} attempts")
            if len(examples) % args.save_every == 0:
                cache_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    if len(examples) < args.num_labels and args.template_synthetic_if_needed:
        add_template_synthetic_examples(
            examples=examples,
            corpus=corpus,
            excluded_docs=excluded_docs,
            used_doc_ids=used_doc_ids,
            args=args,
        )
        cache_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    if len(examples) < args.num_labels and args.synthetic_if_needed:
        if labeler is None:
            print("Loading local Llama3 labeler...")
            llm = create_model(args.llama_config)
            labeler = StableLLMGenerator(
                llm,
                {
                    "enabled": True,
                    "force_greedy": True,
                    "max_new_tokens": args.label_max_new_tokens,
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                    "suppress_transformers_warnings": True,
                },
            )
        corpus_ids = [
            doc_id
            for doc_id in corpus.keys()
            if doc_id not in excluded_docs and doc_id not in used_doc_ids
        ]
        random.Random(args.seed + 1).shuffle(corpus_ids)
        print(
            f"Qrel labels exhausted at {len(examples)}/{args.num_labels}; "
            f"trying synthetic corpus QA over {len(corpus_ids)} unused docs"
        )
        for doc_id in corpus_ids:
            if len(examples) >= args.num_labels:
                break
            if doc_id in used_doc_ids:
                continue
            attempts += 1
            doc = corpus[doc_id]
            title = str(doc.get("title", "")).strip()
            raw_text = str(doc.get("text", "")).strip()
            if len(raw_text.split()) < args.min_synthetic_doc_words:
                continue
            context = compact_context(f"{title}. {raw_text}", args.label_context_chars)
            generation = labeler.query(make_synthetic_prompt(title, context))
            parsed = parse_synthetic_output(generation["response"])
            if parsed is None:
                continue
            question, answer = parsed
            span = find_span(context, answer)
            if span is None:
                continue
            start, end, exact = span
            example_id = f"synthetic::{doc_id}"
            examples.append(
                {
                    "id": example_id,
                    "query_id": None,
                    "doc_id": doc_id,
                    "question": question,
                    "context": context,
                    "answer_text": exact,
                    "answer_start": start,
                    "answer_end": end,
                    "labeler_output": answer,
                    "source": "synthetic_corpus",
                }
            )
            used_ids.add(example_id)
            used_doc_ids.add(doc_id)
            if len(examples) % 20 == 0:
                print(f"labeled {len(examples)}/{args.num_labels} after {attempts} attempts")
            if len(examples) % args.save_every == 0:
                cache_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    cache_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    metadata = {
        "num_examples": len(examples),
        "attempts": attempts,
        "excluded_queries": len(excluded_queries),
        "excluded_docs": len(excluded_docs),
        "candidate_qrel_pairs": len(candidates),
        "synthetic_if_needed": args.synthetic_if_needed,
        "source_counts": {
            source: sum(1 for example in examples if (example.get("source") or "legacy_qrel_extract") == source)
            for source in sorted({str(example.get("source") or "legacy_qrel_extract") for example in examples})
        },
        "seed": args.seed,
        "source_data_dir": args.data_dir,
        "eval_results_path": args.eval_results_path,
        "adv_path": args.adv_path,
    }
    (output_dir / "labeled_examples_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(f"Saved {len(examples)} labeled examples to {cache_path}")
    return examples


@dataclass
class Feature:
    input_ids: List[int]
    attention_mask: List[int]
    start_positions: int
    end_positions: int


class QADataset(Dataset):
    def __init__(self, features: Sequence[Feature]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.features[index]
        return {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(item.attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(item.start_positions, dtype=torch.long),
            "end_positions": torch.tensor(item.end_positions, dtype=torch.long),
        }


def make_features(
    examples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> List[Feature]:
    features: List[Feature] = []
    for example in examples:
        encoded = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        sequence_ids = encoded.sequence_ids()
        answer_start = int(example["answer_start"])
        answer_end = int(example["answer_end"])
        token_start = None
        token_end = None
        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != 1:
                continue
            start, end = offset
            if token_start is None and start <= answer_start < end:
                token_start = idx
            if start < answer_end <= end:
                token_end = idx
                break
        if token_start is None or token_end is None:
            continue
        features.append(
            Feature(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                start_positions=token_start,
                end_positions=token_end,
            )
        )
    return features


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(args: argparse.Namespace, examples: List[Dict[str, Any]]) -> Path:
    rng = random.Random(args.seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    dev_size = min(args.dev_size, max(1, len(shuffled) // 5))
    dev_examples = shuffled[:dev_size]
    train_examples = shuffled[dev_size:]

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "model"
    split_path = output_dir / "train_dev_split.json"
    split_path.write_text(
        json.dumps(
            {"train": train_examples, "dev": dev_examples},
            indent=2,
            ensure_ascii=False,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    )
    train_features = make_features(train_examples, tokenizer, args.max_length)
    dev_features = make_features(dev_examples, tokenizer, args.max_length)
    if not train_features:
        raise RuntimeError("No trainable features were produced.")

    device = choose_device(args.device)
    model.to(device)
    train_loader = DataLoader(
        QADataset(train_features),
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        QADataset(dev_features),
        batch_size=args.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, Any]] = []
    print(
        f"Training {args.base_model} on {len(train_features)} features, "
        f"dev={len(dev_features)}, device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        step_count = 0
        start_time = time.time()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu().item())
            step_count += 1
        train_loss = total / max(step_count, 1)

        model.eval()
        dev_total = 0.0
        dev_steps = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                dev_total += float(output.loss.detach().cpu().item())
                dev_steps += 1
        dev_loss = dev_total / max(dev_steps, 1) if dev_steps else None
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "seconds": round(time.time() - start_time, 2),
        }
        history.append(record)
        print(json.dumps(record))

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False)
    )
    print(f"Saved trained QA model to {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/nq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--adv_path", default=DEFAULT_ADV_PATH)
    parser.add_argument("--eval_results_path", default=DEFAULT_EVAL_RESULTS)
    parser.add_argument("--output_dir", default="retrieval_framework/results/domain_qa_roberta_120")
    parser.add_argument("--base_model", default="deepset/roberta-base-squad2")
    parser.add_argument("--llama_config", default="model_configs/llama3_config.json")
    parser.add_argument("--num_labels", type=int, default=140)
    parser.add_argument("--dev_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--label_context_chars", type=int, default=2200)
    parser.add_argument("--label_max_new_tokens", type=int, default=24)
    parser.add_argument("--max_qrel_docs_per_query", type=int, default=1)
    parser.add_argument("--skip_llm_qrel", action="store_true")
    parser.add_argument("--template_synthetic_if_needed", action="store_true")
    parser.add_argument("--synthetic_if_needed", action="store_true")
    parser.add_argument("--min_synthetic_doc_words", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    examples = build_labeled_examples(args)
    if len(examples) < 40:
        raise RuntimeError(f"Too few labeled examples: {len(examples)}")
    train_model(args, examples)


if __name__ == "__main__":
    main()

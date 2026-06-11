from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_framework.evidence_hardening import (  # noqa: E402
    AnswerMention,
    ExtractiveQAAnswerExtractor,
    answer_key,
    extract_heuristic_answers,
    infer_query_type,
    normalize_text,
    unique_mentions,
)
from retrieval_framework.run_experiment import clean_str  # noqa: E402


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z'&.-]+|[A-Z]{2,})"
    r"(?:\s+(?:of|the|and|for|in|de|da|van|von|[A-Z][A-Za-z'&.-]+|[A-Z]{2,})){0,5}"
)
MONTH_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
NUMBER_PHRASE_PATTERN = re.compile(
    r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s+(?:percent|percentage|years?|months?|weeks?|days?|hours?|minutes?|"
    r"seconds?|miles?|kilometers?|metres?|meters?|feet|ft|inches?|episodes?|"
    r"seasons?|points?|runs?|goals?|people|members|states?|countries|cities|"
    r"dollars?|usd|pounds?|kg|kilograms?|degrees?)){0,3}\b",
    re.IGNORECASE,
)
LOW_VALUE_KEYS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "is",
    "are",
    "was",
    "were",
}


@dataclass
class Candidate:
    key: str
    label: str
    source: str
    doc_rank: int
    score: float


def normalize_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    return clean_str(str(value or ""))


def answer_matches(label: str, answer: Any) -> bool:
    candidate = normalize_answer(label)
    target = normalize_answer(answer)
    if not candidate or not target:
        return False
    return candidate == target or candidate in target or target in candidate


def iter_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for iteration in payload.get("results", []):
        if not isinstance(iteration, dict):
            continue
        for rows in iteration.values():
            for item in rows or []:
                yield item


def clean_label(label: str, max_words: int = 12) -> str:
    label = re.sub(r"\s+", " ", label or "").strip(" \t\n\r,;:()[]{}\"'")
    if not label:
        return ""
    words = label.split()
    if len(words) > max_words:
        label = " ".join(words[:max_words])
    return label.strip(" \t\n\r,;:()[]{}\"'")


def low_value(label: str, query: str) -> bool:
    key = answer_key(label)
    if not key:
        return True
    tokens = key.split()
    if not tokens or len(tokens) > 12:
        return True
    if len(tokens) == 1 and tokens[0] in LOW_VALUE_KEYS:
        return True
    query_tokens = set(normalize_text(query).split())
    if set(tokens) <= query_tokens:
        return True
    return False


def add_candidate(
    candidates: List[Candidate],
    *,
    label: str,
    source: str,
    doc_rank: int,
    score: float,
    query: str,
) -> None:
    label = clean_label(label)
    if low_value(label, query):
        return
    candidates.append(
        Candidate(
            key=answer_key(label),
            label=label,
            source=source,
            doc_rank=doc_rank,
            score=score,
        )
    )


def metadata_candidates(doc: Dict[str, Any], query: str, doc_rank: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    values: List[str] = []
    for field in ("title", "section", "url"):
        value = doc.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    metadata = doc.get("metadata") or {}
    if isinstance(metadata, dict):
        for field in ("title", "section", "entity", "url"):
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    for value in values:
        if "/" in value:
            value = value.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")
        add_candidate(
            candidates,
            label=value,
            source="metadata",
            doc_rank=doc_rank,
            score=8.0 - doc_rank * 0.01,
            query=query,
        )
    return candidates


def regex_candidates(text: str, query: str, doc_rank: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    query_type = infer_query_type(query)
    patterns: List[tuple[re.Pattern[str], str, float]] = []
    patterns.extend([(MONTH_PATTERN, "date", 7.0), (YEAR_PATTERN, "year", 6.5)])
    patterns.append((NUMBER_PHRASE_PATTERN, "number", 6.8))
    patterns.append((ENTITY_PATTERN, "entity", 6.0))
    for pattern, source, base_score in patterns:
        for match in pattern.finditer(text or ""):
            label = match.group(0)
            source_score = base_score
            if query_type == "date" and source in {"date", "year"}:
                source_score += 1.0
            elif query_type == "number" and source == "number":
                source_score += 1.0
            elif query_type == "entity" and source == "entity":
                source_score += 0.5
            add_candidate(
                candidates,
                label=label,
                source=source,
                doc_rank=doc_rank,
                score=source_score - doc_rank * 0.01,
                query=query,
            )
    return candidates


def heuristic_candidates(text: str, query: str, doc_rank: int, max_mentions: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    mentions = extract_heuristic_answers(
        query,
        text,
        max_mentions=max_mentions,
        focused_sentence_count=6,
        max_candidate_words=12,
        use_query_focus=True,
        use_answer_cues=True,
    )
    for idx, mention in enumerate(mentions):
        add_candidate(
            candidates,
            label=mention.label,
            source=f"heuristic:{mention.source}",
            doc_rank=doc_rank,
            score=7.5 - idx * 0.01 - doc_rank * 0.01,
            query=query,
        )
    return candidates


def unique_ranked(candidates: Sequence[Candidate]) -> List[Candidate]:
    best: Dict[str, Candidate] = {}
    for cand in candidates:
        current = best.get(cand.key)
        if current is None or (cand.score, -cand.doc_rank) > (current.score, -current.doc_rank):
            best[cand.key] = cand
    return sorted(best.values(), key=lambda c: (c.score, -c.doc_rank), reverse=True)


def pool_hits(candidates: Sequence[Candidate], answer: Any, m: int) -> bool:
    return any(answer_matches(c.label, answer) for c in candidates[:m])


def source_hits(candidates: Sequence[Candidate], answer: Any) -> Dict[str, int]:
    hits: Dict[str, int] = {}
    for cand in candidates:
        if answer_matches(cand.label, answer):
            root_source = cand.source.split(":", 1)[0]
            hits[root_source] = 1
    return hits


def build_pools_for_item(
    item: Dict[str, Any],
    extractor: ExtractiveQAAnswerExtractor,
    *,
    passage_depth: int,
    qa_topn: int,
    qa_top_starts: int,
    qa_top_ends: int,
    max_answer_tokens: int,
    heuristic_topn: int,
) -> Dict[str, List[Candidate]]:
    query = item["question"]
    docs = (item.get("raw_retrieved") or item.get("retrieved") or [])[:passage_depth]
    texts = [doc.get("text", "") for doc in docs]
    qa_by_doc = extractor.extract_batch(
        query=query,
        texts=texts,
        max_mentions=qa_topn,
        top_starts=qa_top_starts,
        top_ends=qa_top_ends,
        max_answer_tokens=max_answer_tokens,
        min_score=-1e9,
    )

    qa_top1: List[Candidate] = []
    qa_topn_pool: List[Candidate] = []
    multi_source: List[Candidate] = []
    for doc_idx, (doc, qa_mentions) in enumerate(zip(docs, qa_by_doc), start=1):
        for mention_idx, mention in enumerate(qa_mentions):
            cand = Candidate(
                key=mention.key,
                label=mention.label,
                source="qa",
                doc_rank=doc_idx,
                score=10.0 + float(mention.score) * 0.001 - doc_idx * 0.01 - mention_idx * 0.001,
            )
            qa_topn_pool.append(cand)
            multi_source.append(cand)
            if mention_idx == 0:
                qa_top1.append(cand)

        text = doc.get("text", "")
        multi_source.extend(metadata_candidates(doc, query, doc_idx))
        multi_source.extend(regex_candidates(text, query, doc_idx))
        multi_source.extend(heuristic_candidates(text, query, doc_idx, heuristic_topn))

    return {
        "qa_top1": unique_ranked(qa_top1),
        "qa_topn": unique_ranked(qa_topn_pool),
        "multi_source": unique_ranked(multi_source),
    }


def evaluate(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = json.loads(Path(args.results_path).read_text())
    items = list(iter_items(payload))[: args.limit]
    extractor = ExtractiveQAAnswerExtractor(
        model_name=args.qa_model,
        device=args.device,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    ms = [int(value) for value in args.recall_at.split(",") if value.strip()]
    aggregate: Dict[str, Dict[str, Any]] = {}
    details: List[Dict[str, Any]] = []
    source_names: set[str] = set()

    for index, item in enumerate(items, start=1):
        pools = build_pools_for_item(
            item,
            extractor,
            passage_depth=args.passage_depth,
            qa_topn=args.qa_topn,
            qa_top_starts=args.qa_top_starts,
            qa_top_ends=args.qa_top_ends,
            max_answer_tokens=args.max_answer_tokens,
            heuristic_topn=args.heuristic_topn,
        )
        for pool_name, candidates in pools.items():
            rec = aggregate.setdefault(
                pool_name,
                {
                    "pool": pool_name,
                    "num_queries": 0,
                    "avg_pool_size": 0.0,
                    "gold_source_hits": {},
                    "poison_source_hits": {},
                },
            )
            rec["num_queries"] += 1
            rec["avg_pool_size"] += len(candidates)
            gold_hits = source_hits(candidates, item.get("answer"))
            poison_hits = source_hits(candidates, item.get("incorrect_answer"))
            for source in gold_hits:
                source_names.add(source)
                rec["gold_source_hits"][source] = rec["gold_source_hits"].get(source, 0) + 1
            for source in poison_hits:
                source_names.add(source)
                rec["poison_source_hits"][source] = rec["poison_source_hits"].get(source, 0) + 1

            row = {
                "pool": pool_name,
                "query_index": index,
                "id": item.get("id"),
                "question": item.get("question"),
                "correct_answer": item.get("answer"),
                "incorrect_answer": item.get("incorrect_answer"),
                "pool_size": len(candidates),
                "top_candidates": " | ".join(c.label for c in candidates[:10]),
            }
            for m in ms:
                gold = pool_hits(candidates, item.get("answer"), m)
                poison = pool_hits(candidates, item.get("incorrect_answer"), m)
                any_answer = gold or poison
                row[f"gold_recall@{m}"] = int(gold)
                row[f"poison_recall@{m}"] = int(poison)
                row[f"any_recall@{m}"] = int(any_answer)
                rec[f"gold_recall@{m}"] = rec.get(f"gold_recall@{m}", 0) + int(gold)
                rec[f"poison_recall@{m}"] = rec.get(f"poison_recall@{m}", 0) + int(poison)
                rec[f"any_recall@{m}"] = rec.get(f"any_recall@{m}", 0) + int(any_answer)
            details.append(row)

    summaries: List[Dict[str, Any]] = []
    for pool_name, rec in aggregate.items():
        n = rec["num_queries"]
        summary = {
            "pool": pool_name,
            "num_queries": n,
            "avg_pool_size": rec["avg_pool_size"] / max(n, 1),
        }
        for m in ms:
            summary[f"gold_recall@{m}"] = rec.get(f"gold_recall@{m}", 0) / max(n, 1)
            summary[f"poison_recall@{m}"] = rec.get(f"poison_recall@{m}", 0) / max(n, 1)
            summary[f"any_recall@{m}"] = rec.get(f"any_recall@{m}", 0) / max(n, 1)
        for source in sorted(source_names):
            summary[f"gold_source_{source}"] = rec["gold_source_hits"].get(source, 0) / max(n, 1)
            summary[f"poison_source_{source}"] = rec["poison_source_hits"].get(source, 0) / max(n, 1)
        summaries.append(summary)
    return summaries, details


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        default="retrieval_framework/results/qa_model_swap_100q/qa_roberta_base_100q.json",
    )
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/candidate_pool_recall",
    )
    parser.add_argument("--qa_model", default="deepset/deberta-v3-base-squad2")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--passage_depth", type=int, default=5)
    parser.add_argument("--qa_topn", type=int, default=5)
    parser.add_argument("--qa_top_starts", type=int, default=10)
    parser.add_argument("--qa_top_ends", type=int, default=10)
    parser.add_argument("--max_answer_tokens", type=int, default=12)
    parser.add_argument("--heuristic_topn", type=int, default=20)
    parser.add_argument("--recall_at", default="5,10,20,50")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, details = evaluate(args)
    (output_dir / "candidate_pool_recall_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False)
    )
    (output_dir / "candidate_pool_recall_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False)
    )
    write_csv(output_dir / "candidate_pool_recall_summary.csv", summaries)
    write_csv(output_dir / "candidate_pool_recall_details.csv", details)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

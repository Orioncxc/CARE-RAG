from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from retrieval_framework.evidence_hardening import ExtractiveQAAnswerExtractor
    from retrieval_framework.run_experiment import clean_str
except ImportError:  # pragma: no cover
    from .evidence_hardening import ExtractiveQAAnswerExtractor
    from .run_experiment import clean_str


DEFAULT_RESULTS_PATH = (
    "retrieval_framework/results/evidence_qa_only_robust_top1_llm_stable/"
    "nq-secure_ensemble-llama3-Top5-M10x10-consensus_rrf-s1-cap0-eh30-c1-a1-ac2-x1-"
    "adv-LM_targeted-5.json"
)


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
        for results in iteration.values():
            for item in results or []:
                yield item


def classify_mentions(
    mentions: Sequence[Any],
    correct_answer: Any,
    incorrect_answer: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, mention in enumerate(mentions, start=1):
        label = getattr(mention, "label", "")
        is_correct = answer_matches(label, correct_answer)
        is_incorrect = answer_matches(label, incorrect_answer)
        rows.append(
            {
                "mention_rank": rank,
                "mention_label": label,
                "mention_key": getattr(mention, "key", ""),
                "mention_score": getattr(mention, "score", None),
                "is_correct": int(is_correct),
                "is_incorrect": int(is_incorrect),
                "is_other": int(not is_correct and not is_incorrect),
            }
        )
    return rows


def evaluate_model(
    model_name: str,
    items: Sequence[Dict[str, Any]],
    *,
    device: str,
    local_files_only: bool,
    max_length: int,
    batch_size: int,
    max_mentions: int,
    top_starts: int,
    top_ends: int,
    max_answer_tokens: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    extractor = ExtractiveQAAnswerExtractor(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        max_length=max_length,
        batch_size=batch_size,
    )

    detail_rows: List[Dict[str, Any]] = []
    query_count = 0
    top1_primary_correct = 0
    top1_primary_incorrect = 0
    top1_primary_other = 0
    top5_any_correct = 0
    top5_any_incorrect = 0
    top5_all_no_answer = 0
    mention_correct = 0
    mention_incorrect = 0
    mention_other = 0
    mention_total = 0

    for index, item in enumerate(items, start=1):
        query_count += 1
        question = item["question"]
        correct_answer = item.get("answer")
        incorrect_answer = item.get("incorrect_answer")
        docs = item.get("retrieved", [])[:5]
        texts = [doc.get("text", "") for doc in docs]
        mention_groups = extractor.extract_batch(
            query=question,
            texts=texts,
            max_mentions=max_mentions,
            top_starts=top_starts,
            top_ends=top_ends,
            max_answer_tokens=max_answer_tokens,
            min_score=-1e9,
        )

        query_has_correct = False
        query_has_incorrect = False
        query_has_any = False
        top1_mentions = mention_groups[0] if mention_groups else []
        if top1_mentions:
            primary = top1_mentions[0]
            if answer_matches(primary.label, correct_answer):
                top1_primary_correct += 1
            elif answer_matches(primary.label, incorrect_answer):
                top1_primary_incorrect += 1
            else:
                top1_primary_other += 1
        else:
            top1_primary_other += 1

        for doc_rank, (doc, mentions) in enumerate(zip(docs, mention_groups), start=1):
            classified = classify_mentions(mentions, correct_answer, incorrect_answer)
            if classified:
                query_has_any = True
            for row in classified:
                mention_total += 1
                mention_correct += row["is_correct"]
                mention_incorrect += row["is_incorrect"]
                mention_other += row["is_other"]
                query_has_correct = query_has_correct or bool(row["is_correct"])
                query_has_incorrect = query_has_incorrect or bool(row["is_incorrect"])
                detail_rows.append(
                    {
                        "model": model_name,
                        "query_index": index,
                        "id": item.get("id"),
                        "question": question,
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "doc_rank": doc_rank,
                        "doc_id": doc.get("doc_id"),
                        "doc_is_poison": int(str(doc.get("doc_id", "")).startswith("adv::")),
                        **row,
                    }
                )

        top5_any_correct += int(query_has_correct)
        top5_any_incorrect += int(query_has_incorrect)
        top5_all_no_answer += int(not query_has_any)

    summary = {
        "model": model_name,
        "num_queries": query_count,
        "top1_primary_correct_rate": top1_primary_correct / query_count if query_count else 0.0,
        "top1_primary_incorrect_rate": top1_primary_incorrect / query_count if query_count else 0.0,
        "top1_primary_other_rate": top1_primary_other / query_count if query_count else 0.0,
        "top5_any_correct_rate": top5_any_correct / query_count if query_count else 0.0,
        "top5_any_incorrect_rate": top5_any_incorrect / query_count if query_count else 0.0,
        "top5_all_no_answer_rate": top5_all_no_answer / query_count if query_count else 0.0,
        "mention_correct_rate": mention_correct / mention_total if mention_total else 0.0,
        "mention_incorrect_rate": mention_incorrect / mention_total if mention_total else 0.0,
        "mention_other_rate": mention_other / mention_total if mention_total else 0.0,
        "mention_total": mention_total,
        "top1_primary_correct_count": top1_primary_correct,
        "top1_primary_incorrect_count": top1_primary_incorrect,
        "top5_any_correct_count": top5_any_correct,
        "top5_any_incorrect_count": top5_any_incorrect,
    }
    return summary, detail_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output_dir", default="retrieval_framework/results/qa_extractor_precision")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepset/minilm-uncased-squad2", "deepset/deberta-v3-base-squad2"],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_mentions", type=int, default=3)
    parser.add_argument("--top_starts", type=int, default=6)
    parser.add_argument("--top_ends", type=int, default=6)
    parser.add_argument("--max_answer_tokens", type=int, default=12)
    args = parser.parse_args()

    payload = json.loads(Path(args.results_path).read_text())
    items = list(iter_items(payload))[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    for model_name in args.models:
        print(f"Evaluating {model_name} on {len(items)} queries...")
        summary, details = evaluate_model(
            model_name,
            items,
            device=args.device,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_mentions=args.max_mentions,
            top_starts=args.top_starts,
            top_ends=args.top_ends,
            max_answer_tokens=args.max_answer_tokens,
        )
        summaries.append(summary)
        all_details.extend(details)
        print(json.dumps(summary, indent=2))

    metadata = {
        "results_path": args.results_path,
        "limit": len(items),
        "device": args.device,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_mentions": args.max_mentions,
        "top_starts": args.top_starts,
        "top_ends": args.top_ends,
        "max_answer_tokens": args.max_answer_tokens,
        "summaries": summaries,
    }
    (output_dir / "qa_extractor_precision_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    write_csv(output_dir / "qa_extractor_precision_summary.csv", summaries)
    write_csv(output_dir / "qa_extractor_precision_details.csv", all_details)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

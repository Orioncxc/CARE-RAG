import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_text(value: Any) -> str:
    return str(value or "").strip().lower()


def iter_query_rows(results: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for block in results:
        if not isinstance(block, dict) or not block:
            continue
        rows = next(iter(block.values()))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_result(path: Path) -> Dict[str, Any]:
    from src.utils import load_beir_datasets

    payload = load_json(path)
    config = payload.get("config", {})
    dataset = config.get("dataset")
    split = "train" if dataset == "msmarco" else config.get("split", "test")
    _, _, qrels = load_beir_datasets(dataset, split)

    rows = list(iter_query_rows(payload.get("results", [])))
    hit_at_1: List[bool] = []
    hit_at_3: List[bool] = []
    hit_at_5: List[bool] = []
    answer_hits: List[bool] = []
    nonempty_outputs: List[bool] = []

    for row in rows:
        query_id = str(row.get("id"))
        relevant = set(qrels.get(query_id, {}).keys())
        retrieved_ids = [str(item.get("doc_id")) for item in row.get("retrieved", [])]
        hit_at_1.append(bool(relevant.intersection(retrieved_ids[:1])))
        hit_at_3.append(bool(relevant.intersection(retrieved_ids[:3])))
        hit_at_5.append(bool(relevant.intersection(retrieved_ids[:5])))

        output = row.get("output_poison")
        if output is not None:
            output_text = clean_text(output)
            answer = clean_text(row.get("answer"))
            nonempty_outputs.append(bool(output_text))
            answer_hits.append(bool(answer and answer in output_text))

    def mean(values: List[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "path": str(path),
        "dataset": dataset,
        "run_name": path.stem.removesuffix(".summary"),
        "num_queries": len(rows),
        "retrieval_hit_at_1": mean(hit_at_1),
        "retrieval_hit_at_3": mean(hit_at_3),
        "retrieval_hit_at_5": mean(hit_at_5),
        "has_llm_outputs": bool(answer_hits),
        "answer_contains_accuracy": mean(answer_hits),
        "nonempty_output_rate": mean(nonempty_outputs),
    }
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "dataset",
        "num_queries",
        "retrieval_hit_at_1",
        "retrieval_hit_at_3",
        "retrieval_hit_at_5",
        "has_llm_outputs",
        "answer_contains_accuracy",
        "nonempty_output_rate",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clean no-attack utility results.")
    parser.add_argument("paths", nargs="+", help="Result JSON paths, not summary JSON paths.")
    parser.add_argument(
        "--output",
        default="retrieval_framework/results/missing_experiments/clean_utility_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_result(Path(path)) for path in args.paths]
    write_csv(Path(args.output), rows)
    for row in rows:
        print(
            "{dataset} {run_name}: n={num_queries} R@1={retrieval_hit_at_1:.4f} "
            "R@3={retrieval_hit_at_3:.4f} R@5={retrieval_hit_at_5:.4f} "
            "answer_acc={answer_contains_accuracy:.4f}".format(**row)
        )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

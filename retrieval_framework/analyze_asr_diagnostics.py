from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_answer(text: Any, answer: Any) -> bool:
    answer_key = normalize_text(answer)
    if not answer_key:
        return False
    return f" {answer_key} " in f" {normalize_text(text)} "


def coerce_answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def doc_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = doc.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def evidence_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = doc_metadata(doc)
    evidence = metadata.get("evidence_hardening")
    return evidence if isinstance(evidence, dict) else {}


def reference_kinds(doc: Dict[str, Any]) -> set:
    evidence = evidence_metadata(doc)
    kinds = set()
    for item in evidence.get("reference_answers", []) or []:
        if isinstance(item, dict) and item.get("kind"):
            kinds.add(str(item["kind"]))
    return kinds


def is_adv_doc_for_query(doc: Dict[str, Any], query_id: Any) -> bool:
    metadata = doc_metadata(doc)
    return bool(metadata.get("is_adv")) and str(metadata.get("target_id")) == str(query_id)


def is_any_adv_doc(doc: Dict[str, Any]) -> bool:
    return bool(doc_metadata(doc).get("is_adv"))


def doc_rank(doc: Dict[str, Any], fallback_rank: int) -> int:
    try:
        return int(doc.get("rank") or fallback_rank)
    except (TypeError, ValueError):
        return fallback_rank


def doc_cluster_id(doc: Dict[str, Any]) -> str:
    evidence = evidence_metadata(doc)
    metadata = doc_metadata(doc)
    value = evidence.get("cluster_id", metadata.get("cluster_id", ""))
    return str(value)


def doc_cluster_size(doc: Dict[str, Any]) -> int:
    evidence = evidence_metadata(doc)
    metadata = doc_metadata(doc)
    value = evidence.get("cluster_size", metadata.get("cluster_size", 1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def output_has_incorrect_answer(row: Dict[str, Any]) -> bool:
    output = row.get("output_poison", "")
    return any(contains_answer(output, answer) for answer in coerce_answer_list(row.get("incorrect_answer")))


def supports_gold_answer(doc: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if "correct" in reference_kinds(doc):
        return True
    return any(contains_answer(doc.get("text", ""), answer) for answer in coerce_answer_list(row.get("answer")))


def supports_wrong_answer(doc: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if "incorrect" in reference_kinds(doc):
        return True
    return any(
        contains_answer(doc.get("text", ""), answer)
        for answer in coerce_answer_list(row.get("incorrect_answer"))
    )


def sorted_top_docs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = list(row.get("retrieved", []) or [])
    return sorted(docs, key=lambda doc: doc_rank(doc, docs.index(doc) + 1))


def min_or_blank(values: Iterable[int]) -> Any:
    values = list(values)
    return min(values) if values else ""


def mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def compute_query_diagnostics(row: Dict[str, Any]) -> Dict[str, Any]:
    query_id = row.get("id")
    docs = sorted_top_docs(row)
    target_adv_docs = [doc for doc in docs if is_adv_doc_for_query(doc, query_id)]
    target_adv_ranks = [doc_rank(doc, idx + 1) for idx, doc in enumerate(docs) if is_adv_doc_for_query(doc, query_id)]
    poison_support_docs = [doc for doc in target_adv_docs if supports_wrong_answer(doc, row)]
    poison_support_ranks = [
        doc_rank(doc, idx + 1)
        for idx, doc in enumerate(docs)
        if is_adv_doc_for_query(doc, query_id) and supports_wrong_answer(doc, row)
    ]
    clean_support_docs = [
        doc
        for doc in docs
        if not is_any_adv_doc(doc) and supports_gold_answer(doc, row)
    ]
    clean_support_ranks = [
        doc_rank(doc, idx + 1)
        for idx, doc in enumerate(docs)
        if not is_any_adv_doc(doc) and supports_gold_answer(doc, row)
    ]
    target_adv_clusters = [doc_cluster_id(doc) for doc in target_adv_docs if doc_cluster_id(doc) != ""]
    poison_support_clusters = [
        doc_cluster_id(doc) for doc in poison_support_docs if doc_cluster_id(doc) != ""
    ]
    poison_cluster_count = len(set(target_adv_clusters))
    poison_support_cluster_count = len(set(poison_support_clusters))
    poison_max_cluster_size = max([doc_cluster_size(doc) for doc in target_adv_docs] or [0])
    poison_support_max_cluster_size = max([doc_cluster_size(doc) for doc in poison_support_docs] or [0])

    top_doc = docs[0] if docs else {}
    rank1_is_target_adv = bool(docs and is_adv_doc_for_query(top_doc, query_id))
    rank1_supports_wrong = bool(rank1_is_target_adv and supports_wrong_answer(top_doc, row))
    rank1_supports_gold_clean = bool(
        docs and not is_any_adv_doc(top_doc) and supports_gold_answer(top_doc, row)
    )
    poison_in_top3 = any(rank <= 3 for rank in target_adv_ranks)
    poison_support_in_top3 = any(rank <= 3 for rank in poison_support_ranks)
    poison_in_top5 = any(rank <= 5 for rank in target_adv_ranks)

    return {
        "iteration": row.get("_iteration", ""),
        "id": query_id,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "incorrect_answer": row.get("incorrect_answer", ""),
        "output_poison": row.get("output_poison", ""),
        "asr": int(output_has_incorrect_answer(row)),
        "poison_rank1": int(rank1_is_target_adv),
        "poison_rank1_supports_wrong": int(rank1_supports_wrong),
        "poison_top3": int(poison_in_top3),
        "poison_support_top3": int(poison_support_in_top3),
        "poison_top5": int(poison_in_top5),
        "target_adv_count_topk": len(target_adv_docs),
        "poison_support_count": len(poison_support_docs),
        "poison_min_rank": min_or_blank(target_adv_ranks),
        "poison_support_min_rank": min_or_blank(poison_support_ranks),
        "clean_gold_support_count": len(clean_support_docs),
        "clean_gold_support_min_rank": min_or_blank(clean_support_ranks),
        "rank1_supports_gold_clean": int(rank1_supports_gold_clean),
        "poison_cluster_count": poison_cluster_count,
        "poison_support_cluster_count": poison_support_cluster_count,
        "poison_max_cluster_size": poison_max_cluster_size,
        "poison_support_max_cluster_size": poison_support_max_cluster_size,
        "multiple_poison_same_cluster": int(
            len(target_adv_docs) >= 2 and poison_cluster_count < len(target_adv_docs)
        ),
        "multiple_wrong_poison_same_cluster": int(
            len(poison_support_docs) >= 2
            and poison_support_cluster_count < len(poison_support_docs)
        ),
        "rank1_doc_id": top_doc.get("doc_id", ""),
        "rank1_is_adv": int(is_any_adv_doc(top_doc)) if docs else 0,
        "rank1_original_rank": evidence_metadata(top_doc).get("original_rank", ""),
        "rank1_primary_answer_key": evidence_metadata(top_doc).get("primary_answer_key", ""),
    }


def group_summary(rows: Sequence[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key_field]].append(row)
    output = []
    for value in sorted(grouped.keys()):
        group = grouped[value]
        n = len(group)
        asr_count = sum(int(item["asr"]) for item in group)
        output.append(
            {
                key_field: value,
                "num_queries": n,
                "asr_count": asr_count,
                "asr_rate": asr_count / n if n else 0.0,
                "mean_target_adv_count_topk": mean_or_zero(
                    [float(item["target_adv_count_topk"]) for item in group]
                ),
                "mean_clean_gold_support_count": mean_or_zero(
                    [float(item["clean_gold_support_count"]) for item in group]
                ),
                "mean_poison_support_count": mean_or_zero(
                    [float(item["poison_support_count"]) for item in group]
                ),
            }
        )
    return output


def support_pattern_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["clean_gold_support_count"]),
            int(row["poison_support_count"]),
            int(row["poison_rank1"]),
            int(row["poison_top3"]),
        )
        grouped[key].append(row)

    output = []
    for key, group in grouped.items():
        clean_count, poison_count, poison_rank1, poison_top3 = key
        n = len(group)
        asr_count = sum(int(item["asr"]) for item in group)
        output.append(
            {
                "clean_gold_support_count": clean_count,
                "poison_support_count": poison_count,
                "poison_rank1": poison_rank1,
                "poison_top3": poison_top3,
                "num_queries": n,
                "asr_count": asr_count,
                "asr_rate": asr_count / n if n else 0.0,
                "example_ids": ";".join(str(item["id"]) for item in group[:5]),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["num_queries"]),
            -float(row["asr_rate"]),
            int(row["clean_gold_support_count"]),
            int(row["poison_support_count"]),
        )
    )
    return output


def priority_error_bucket(row: Dict[str, Any]) -> str:
    if int(row["poison_rank1"]):
        return "poison_at_rank1"
    if int(row["poison_top3"]):
        return "poison_in_top3_not_rank1"
    if int(row["multiple_wrong_poison_same_cluster"]) or int(row["multiple_poison_same_cluster"]):
        return "multiple_poison_same_cluster"
    if int(row["clean_gold_support_count"]) <= 1:
        return "clean_support_weak_or_ambiguous"
    if int(row["poison_support_count"]) <= 1 and int(row["target_adv_count_topk"]) <= 1:
        return "few_poison_but_still_success"
    return "other"


def error_bucket_summary(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    asr_rows = [row for row in rows if int(row["asr"])]
    priority_counts = Counter(priority_error_bucket(row) for row in asr_rows)
    priority = [
        {
            "bucket": bucket,
            "asr_error_count": count,
            "share_of_asr_errors": count / len(asr_rows) if asr_rows else 0.0,
        }
        for bucket, count in priority_counts.most_common()
    ]

    labels = {
        "poison_at_rank1": lambda row: int(row["poison_rank1"]) == 1,
        "poison_in_top3": lambda row: int(row["poison_top3"]) == 1,
        "poison_few_but_strong": lambda row: int(row["poison_support_count"]) <= 1
        and int(row["target_adv_count_topk"]) <= 1,
        "clean_support_weak_or_ambiguous": lambda row: int(row["clean_gold_support_count"]) <= 1,
        "multiple_poison_same_cluster": lambda row: int(row["multiple_poison_same_cluster"]) == 1
        or int(row["multiple_wrong_poison_same_cluster"]) == 1,
        "no_poison_in_top3": lambda row: int(row["poison_top3"]) == 0,
        "clean_gold_rank1": lambda row: int(row["rank1_supports_gold_clean"]) == 1,
    }
    multilabel = []
    for label, predicate in labels.items():
        matched = [row for row in asr_rows if predicate(row)]
        multilabel.append(
            {
                "label": label,
                "asr_error_count": len(matched),
                "share_of_asr_errors": len(matched) / len(asr_rows) if asr_rows else 0.0,
                "example_ids": ";".join(str(item["id"]) for item in matched[:5]),
            }
        )
    multilabel.sort(key=lambda row: -int(row["asr_error_count"]))
    return priority, multilabel


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str], limit: Optional[int] = None) -> str:
    selected = list(rows[:limit] if limit is not None else rows)
    if not selected:
        return ""
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in selected:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    result_path: Path,
    query_rows: Sequence[Dict[str, Any]],
    table1: Sequence[Dict[str, Any]],
    table2: Sequence[Dict[str, Any]],
    table3: Sequence[Dict[str, Any]],
    table4_priority: Sequence[Dict[str, Any]],
    table4_multilabel: Sequence[Dict[str, Any]],
) -> None:
    asr_count = sum(int(row["asr"]) for row in query_rows)
    n = len(query_rows)
    content = [
        "# ASR Diagnostics",
        "",
        f"Source result: `{result_path}`",
        f"Queries: {n}",
        f"ASR: {asr_count}/{n} = {asr_count / n if n else 0.0:.4f}",
        "",
        "## Table 1: ASR vs Poison at Rank 1",
        markdown_table(
            table1,
            [
                "poison_rank1",
                "num_queries",
                "asr_count",
                "asr_rate",
                "mean_target_adv_count_topk",
                "mean_clean_gold_support_count",
                "mean_poison_support_count",
            ],
        ),
        "",
        "## Table 2: ASR vs Poison in Top 3",
        markdown_table(
            table2,
            [
                "poison_top3",
                "num_queries",
                "asr_count",
                "asr_rate",
                "mean_target_adv_count_topk",
                "mean_clean_gold_support_count",
                "mean_poison_support_count",
            ],
        ),
        "",
        "## Table 3: Clean Support vs Poison Support",
        markdown_table(
            table3,
            [
                "clean_gold_support_count",
                "poison_support_count",
                "poison_rank1",
                "poison_top3",
                "num_queries",
                "asr_count",
                "asr_rate",
                "example_ids",
            ],
            limit=25,
        ),
        "",
        "## Table 4a: ASR Error Priority Buckets",
        markdown_table(
            table4_priority,
            ["bucket", "asr_error_count", "share_of_asr_errors"],
        ),
        "",
        "## Table 4b: ASR Error Multi-label Features",
        markdown_table(
            table4_multilabel,
            ["label", "asr_error_count", "share_of_asr_errors", "example_ids"],
        ),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def output_prefix(result_path: Path, output_dir: Optional[str]) -> Path:
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = result_path.parent / "diagnostics"
    return base_dir / result_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ASR/rank/support diagnostics from a retrieval framework result JSON."
    )
    parser.add_argument("--result_path", required=True, help="Path to run_experiment result JSON.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for generated CSV/Markdown files. Defaults to results/diagnostics.",
    )
    return parser.parse_args()


def flatten_result_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("results", payload)
    if not isinstance(payload, list):
        raise ValueError("Expected a list or a dict with a 'results' list.")

    rows: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict) and "retrieved" in item:
            rows.append(dict(item))
            continue
        if isinstance(item, dict) and len(item) == 1:
            iteration, nested = next(iter(item.items()))
            if isinstance(nested, list):
                for row in nested:
                    if not isinstance(row, dict):
                        continue
                    copied = dict(row)
                    copied["_iteration"] = iteration
                    rows.append(copied)
                continue
        if isinstance(item, list):
            for row in item:
                if isinstance(row, dict):
                    rows.append(dict(row))
            continue
        raise ValueError(f"Unsupported result item shape: {type(item).__name__}")
    return rows


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_path)
    with result_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    result_rows = flatten_result_rows(payload)

    query_rows = [compute_query_diagnostics(row) for row in result_rows]
    table1 = group_summary(query_rows, "poison_rank1")
    table2 = group_summary(query_rows, "poison_top3")
    table3 = support_pattern_summary(query_rows)
    table4_priority, table4_multilabel = error_bucket_summary(query_rows)

    prefix = output_prefix(result_path, args.output_dir)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_suffix(".query_diagnostics.csv"), query_rows)
    write_csv(prefix.with_suffix(".table1_rank1.csv"), table1)
    write_csv(prefix.with_suffix(".table2_top3.csv"), table2)
    write_csv(prefix.with_suffix(".table3_support_patterns.csv"), table3)
    write_csv(prefix.with_suffix(".table4_error_priority_buckets.csv"), table4_priority)
    write_csv(prefix.with_suffix(".table4_error_multilabel.csv"), table4_multilabel)
    write_markdown_report(
        prefix.with_suffix(".diagnostics.md"),
        result_path,
        query_rows,
        table1,
        table2,
        table3,
        table4_priority,
        table4_multilabel,
    )

    print(f"Wrote diagnostics with prefix: {prefix}")
    print()
    print("Table 1: ASR vs poison at rank 1")
    print(markdown_table(table1, ["poison_rank1", "num_queries", "asr_count", "asr_rate"]))
    print()
    print("Table 2: ASR vs poison in top 3")
    print(markdown_table(table2, ["poison_top3", "num_queries", "asr_count", "asr_rate"]))
    print()
    print("Top support patterns")
    print(
        markdown_table(
            table3,
            [
                "clean_gold_support_count",
                "poison_support_count",
                "poison_rank1",
                "poison_top3",
                "num_queries",
                "asr_count",
                "asr_rate",
            ],
            limit=10,
        )
    )
    print()
    print("ASR error priority buckets")
    print(markdown_table(table4_priority, ["bucket", "asr_error_count", "share_of_asr_errors"]))


if __name__ == "__main__":
    main()

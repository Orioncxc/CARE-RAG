import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def clean_str(value: Any) -> str:
    text = str(value).strip()
    if len(text) > 1 and text[-1] == ".":
        text = text[:-1]
    return text.lower()


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def iter_query_rows(results: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for block in results:
        if not isinstance(block, dict) or not block:
            continue
        rows = next(iter(block.values()))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row


def summary_from_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = payload.get("config", {})
    rows = list(iter_query_rows(payload.get("results", [])))
    top_k = int(config.get("top_k", 5))
    adv_per_query = int(config.get("adv_per_query", 5))
    m = int(config.get("M", 10))

    hits = [int(row.get("target_adv_hits", 0)) for row in rows]
    precision_values = [hit / top_k for hit in hits]
    recall_values = [hit / adv_per_query for hit in hits]
    f1_values = [f1(p, r) for p, r in zip(precision_values, recall_values)]

    asr_flags = [
        clean_str(row.get("incorrect_answer", "")) in clean_str(row.get("output_poison", ""))
        for row in rows
        if row.get("output_poison") is not None
    ]
    support_counts: List[int] = []
    adv_support_counts: List[int] = []
    hardening_metrics: List[Dict[str, Any]] = []
    contamination_by_k = {1: [], 3: [], top_k: []}
    for row in rows:
        if isinstance(row.get("evidence_hardening"), dict):
            hardening_metrics.append(row["evidence_hardening"])
        target_adv_ids = {
            item.get("doc_id")
            for item in row.get("retrieved", [])
            if item.get("metadata", {}).get("is_adv")
            and item.get("metadata", {}).get("target_id") == row.get("id")
        }
        retrieved = row.get("retrieved", [])
        for item in retrieved:
            metadata = item.get("metadata", {})
            support_count = int(
                metadata.get(
                    "support_count",
                    len(metadata.get("channels", {})) if metadata.get("channels") else 1,
                )
            )
            support_counts.append(support_count)
            if item.get("doc_id") in target_adv_ids:
                adv_support_counts.append(support_count)
        for k in contamination_by_k:
            top_slice = retrieved[: min(k, len(retrieved))]
            if top_slice:
                contamination_by_k[k].append(
                    sum(item.get("doc_id") in target_adv_ids for item in top_slice)
                    / len(top_slice)
                )
    asr_by_iter = []
    if asr_flags and m > 0:
        for start in range(0, len(asr_flags), m):
            chunk = asr_flags[start : start + m]
            if chunk:
                asr_by_iter.append(sum(chunk) / len(chunk))

    summary = {
        "retriever": config.get("retriever", {}).get("type"),
        "dataset": config.get("dataset"),
        "top_k": top_k,
        "num_queries": len(rows),
        "target_adv_hits": hits,
        "retrieval_precision_mean": sum(precision_values) / len(precision_values)
        if precision_values
        else 0.0,
        "retrieval_recall_mean": sum(recall_values) / len(recall_values)
        if recall_values
        else 0.0,
        "retrieval_f1_mean": sum(f1_values) / len(f1_values) if f1_values else 0.0,
    }
    if asr_by_iter:
        summary["asr_by_iter"] = asr_by_iter
        summary["asr_mean"] = sum(asr_by_iter) / len(asr_by_iter)
    if support_counts:
        summary["support_count_mean"] = sum(support_counts) / len(support_counts)
        summary["support_count_singleton_rate"] = (
            sum(count == 1 for count in support_counts) / len(support_counts)
        )
    if adv_support_counts:
        summary["adv_support_count_mean"] = sum(adv_support_counts) / len(adv_support_counts)
        summary["adv_support_count_singleton_rate"] = (
            sum(count == 1 for count in adv_support_counts) / len(adv_support_counts)
        )
    for k, values in contamination_by_k.items():
        if values:
            summary[f"contamination_at_{k}_mean"] = sum(values) / len(values)
    if hardening_metrics:
        summary["evidence_hardening_enabled"] = True
        metric_keys = [
            "input_count",
            "output_count",
            "filtered_by_cluster_count",
            "filtered_by_answer_count",
            "cluster_count",
            "max_cluster_size",
            "clusters_with_adv",
            "heuristic_answer_count",
        ]
        for key in metric_keys:
            values = [float(metric.get(key, 0.0)) for metric in hardening_metrics]
            summary[f"hardening_{key}_mean"] = sum(values) / len(values) if values else 0.0
        summary["hardening_conflict_rate"] = (
            sum(bool(metric.get("conflict_detected")) for metric in hardening_metrics)
            / len(hardening_metrics)
        )
        head_metrics = [
            metric.get("head_filter", {})
            for metric in hardening_metrics
            if metric.get("head_filter", {}).get("enabled")
        ]
        if head_metrics:
            summary["head_filter_enabled"] = True
            summary["head_filter_trigger_rate"] = (
                sum(bool(metric.get("triggered")) for metric in head_metrics)
                / len(head_metrics)
            )
            summary["head_filter_order_changed_rate"] = (
                sum(bool(metric.get("order_changed")) for metric in head_metrics)
                / len(head_metrics)
            )
            summary["head_filter_uncertain_recommended_rate"] = (
                sum(bool(metric.get("uncertain_recommended")) for metric in head_metrics)
                / len(head_metrics)
            )
            summary["head_filter_severe_conflict_rate"] = (
                sum(bool(metric.get("severe_conflict")) for metric in head_metrics)
                / len(head_metrics)
            )
            summary["head_filter_isolated_doc_count_mean"] = (
                sum(len(metric.get("isolated_answer_doc_ids", [])) for metric in head_metrics)
                / len(head_metrics)
            )
            summary["head_filter_supplement_promoted_count_mean"] = (
                sum(len(metric.get("supplement_promoted_doc_ids", [])) for metric in head_metrics)
                / len(head_metrics)
            )
        top1_metrics = [
            metric.get("top1_dominance", {})
            for metric in hardening_metrics
            if metric.get("top1_dominance", {}).get("enabled")
        ]
        if top1_metrics:
            summary["top1_dominance_enabled"] = True
            summary["top1_dominance_trigger_rate"] = (
                sum(bool(metric.get("triggered")) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_order_changed_rate"] = (
                sum(bool(metric.get("order_changed")) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_conflict_rate"] = (
                sum(bool(metric.get("conflict_detected")) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_top1_isolated_rate"] = (
                sum(bool(metric.get("top1_isolated")) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_promoted_doc_count_mean"] = (
                sum(len(metric.get("promoted_doc_ids", [])) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_supported_alternative_answer_count_mean"] = (
                sum(
                    len(metric.get("supported_alternative_answers", []))
                    for metric in top1_metrics
                )
                / len(top1_metrics)
            )
            summary["top1_dominance_top1_cluster_count_mean"] = (
                sum(float(metric.get("top1_cluster_count", 0.0)) for metric in top1_metrics)
                / len(top1_metrics)
            )
            summary["top1_dominance_top1_non_echo_cluster_count_mean"] = (
                sum(
                    float(metric.get("top1_non_echo_cluster_count", 0.0))
                    for metric in top1_metrics
                )
                / len(top1_metrics)
            )
        constrained_metrics = [
            metric.get("constrained_selection", {})
            for metric in hardening_metrics
            if metric.get("constrained_selection", {}).get("enabled")
        ]
        if constrained_metrics:
            summary["constrained_selection_enabled"] = True
            summary["constrained_selection_trigger_rate"] = (
                sum(bool(metric.get("triggered")) for metric in constrained_metrics)
                / len(constrained_metrics)
            )
            summary["constrained_selection_order_changed_rate"] = (
                sum(bool(metric.get("order_changed")) for metric in constrained_metrics)
                / len(constrained_metrics)
            )
            summary["constrained_selection_pool_count_mean"] = (
                sum(float(metric.get("pool_count", 0.0)) for metric in constrained_metrics)
                / len(constrained_metrics)
            )
            summary["constrained_selection_query_overlap_selected_count_mean"] = (
                sum(
                    float(metric.get("query_overlap_selected_count", 0.0))
                    for metric in constrained_metrics
                )
                / len(constrained_metrics)
            )
            summary["constrained_selection_duplicate_cluster_selected_count_mean"] = (
                sum(
                    float(metric.get("duplicate_cluster_selected_count", 0.0))
                    for metric in constrained_metrics
                )
                / len(constrained_metrics)
            )
            summary["constrained_selection_fallback_fill_count_mean"] = (
                sum(len(metric.get("fallback_filled_doc_ids", [])) for metric in constrained_metrics)
                / len(constrained_metrics)
            )
        answer_level_metrics = [
            metric.get("answer_level_contradiction", {})
            for metric in hardening_metrics
            if metric.get("answer_level_contradiction")
        ]
        if answer_level_metrics:
            def metric_mean(key: str) -> float:
                return (
                    sum(float(metric.get(key, 0.0)) for metric in answer_level_metrics)
                    / len(answer_level_metrics)
                )

            def metric_rate(key: str) -> float:
                return (
                    sum(bool(metric.get(key)) for metric in answer_level_metrics)
                    / len(answer_level_metrics)
                )

            summary["answer_level_conflict_rate"] = metric_rate("conflict_detected")
            summary["answer_level_candidate_conflict_rate"] = metric_rate(
                "candidate_conflict_detected"
            )
            summary["answer_level_severe_conflict_rate"] = metric_rate("severe_conflict")
            summary["answer_level_top1_isolated_rate"] = metric_rate("top1_isolated")
            summary["answer_level_top1_has_best_support_rate"] = metric_rate(
                "top1_has_best_support"
            )
            summary["answer_level_top1_isolated_with_alternative_rate"] = metric_rate(
                "top1_isolated_with_alternative"
            )
            summary["answer_level_top1_not_best_support_rate"] = metric_rate(
                "top1_not_best_support"
            )
            summary["answer_level_multi_supported_conflict_rate"] = metric_rate(
                "multi_supported_conflict"
            )
            summary["answer_level_no_strong_answer_rate"] = metric_rate(
                "no_strong_answer"
            )
            for key in [
                "selected_answer_count",
                "candidate_answer_count",
                "supported_answer_count",
                "candidate_supported_answer_count",
                "isolated_selected_answer_count",
                "query_echo_only_answer_count",
                "top1_doc_count",
                "top1_cluster_count",
                "top1_non_echo_cluster_count",
                "top1_channel_count",
                "top1_query_echo_doc_count",
                "top1_support_margin",
                "selected_max_non_echo_cluster_count",
                "selected_max_cluster_count",
                "selected_max_channel_count",
                "best_alternative_non_echo_cluster_count",
                "best_alternative_cluster_count",
            ]:
                summary[f"answer_level_{key}_mean"] = metric_mean(key)
            conflict_type_counts: Dict[str, int] = {}
            for metric in answer_level_metrics:
                conflict_type = str(metric.get("conflict_type") or "unknown")
                conflict_type_counts[conflict_type] = conflict_type_counts.get(conflict_type, 0) + 1
            summary["answer_level_conflict_type_counts"] = conflict_type_counts
            for conflict_type, count in sorted(conflict_type_counts.items()):
                safe_name = conflict_type.replace("-", "_").replace(" ", "_")
                summary[f"answer_level_conflict_type_{safe_name}_rate"] = (
                    count / len(answer_level_metrics)
                )
        for answer_kind in ["correct", "incorrect"]:
            for field in ["doc_count", "cluster_count"]:
                values = [
                    float(
                        metric.get("reference_answer_support", {})
                        .get(answer_kind, {})
                        .get(field, 0.0)
                    )
                    for metric in hardening_metrics
                ]
                summary[f"hardening_reference_{answer_kind}_{field}_mean"] = (
                    sum(values) / len(values) if values else 0.0
                )
    return summary


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None


def alpha_from_config(config: Dict[str, Any]) -> Optional[float]:
    retriever_config = config.get("retriever", {})
    if "alpha" not in retriever_config:
        return None
    try:
        return float(retriever_config["alpha"])
    except (TypeError, ValueError):
        return None


def row_from_payload(path: Path, payload: Dict[str, Any], source_type: str) -> Dict[str, Any]:
    config = payload.get("config", {})
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = summary_from_result_payload(payload)

    retriever_config = config.get("retriever", {})
    expected_queries = int(config.get("M", 0) or 0) * int(config.get("repeat_times", 0) or 0)
    num_queries = int(summary.get("num_queries", 0) or 0)

    return {
        "run_name": path.name,
        "source_type": source_type,
        "dataset": summary.get("dataset") or config.get("dataset"),
        "retriever": summary.get("retriever") or retriever_config.get("type"),
        "alpha": alpha_from_config(config),
        "top_k": summary.get("top_k") or config.get("top_k"),
        "adv_per_query": config.get("adv_per_query"),
        "M": config.get("M"),
        "repeat_times": config.get("repeat_times"),
        "num_queries": num_queries,
        "expected_queries": expected_queries,
        "complete": bool(expected_queries == 0 or num_queries == expected_queries),
        "retrieval_precision_mean": summary.get("retrieval_precision_mean"),
        "retrieval_recall_mean": summary.get("retrieval_recall_mean"),
        "retrieval_f1_mean": summary.get("retrieval_f1_mean"),
        "asr_mean": summary.get("asr_mean"),
        "support_count_mean": summary.get("support_count_mean"),
        "support_count_singleton_rate": summary.get("support_count_singleton_rate"),
        "adv_support_count_mean": summary.get("adv_support_count_mean"),
        "adv_support_count_singleton_rate": summary.get("adv_support_count_singleton_rate"),
        "contamination_at_1_mean": summary.get("contamination_at_1_mean"),
        "contamination_at_3_mean": summary.get("contamination_at_3_mean"),
        "contamination_at_5_mean": summary.get("contamination_at_5_mean"),
        "evidence_hardening_enabled": summary.get("evidence_hardening_enabled"),
        "hardening_filtered_by_cluster_count_mean": summary.get(
            "hardening_filtered_by_cluster_count_mean"
        ),
        "hardening_filtered_by_answer_count_mean": summary.get(
            "hardening_filtered_by_answer_count_mean"
        ),
        "hardening_cluster_count_mean": summary.get("hardening_cluster_count_mean"),
        "hardening_max_cluster_size_mean": summary.get("hardening_max_cluster_size_mean"),
        "hardening_conflict_rate": summary.get("hardening_conflict_rate"),
        "hardening_reference_correct_doc_count_mean": summary.get(
            "hardening_reference_correct_doc_count_mean"
        ),
        "hardening_reference_incorrect_doc_count_mean": summary.get(
            "hardening_reference_incorrect_doc_count_mean"
        ),
        "head_filter_enabled": summary.get("head_filter_enabled"),
        "head_filter_trigger_rate": summary.get("head_filter_trigger_rate"),
        "head_filter_order_changed_rate": summary.get("head_filter_order_changed_rate"),
        "head_filter_uncertain_recommended_rate": summary.get(
            "head_filter_uncertain_recommended_rate"
        ),
        "head_filter_severe_conflict_rate": summary.get("head_filter_severe_conflict_rate"),
        "head_filter_isolated_doc_count_mean": summary.get(
            "head_filter_isolated_doc_count_mean"
        ),
        "head_filter_supplement_promoted_count_mean": summary.get(
            "head_filter_supplement_promoted_count_mean"
        ),
        "top1_dominance_enabled": summary.get("top1_dominance_enabled"),
        "top1_dominance_trigger_rate": summary.get("top1_dominance_trigger_rate"),
        "top1_dominance_order_changed_rate": summary.get(
            "top1_dominance_order_changed_rate"
        ),
        "top1_dominance_conflict_rate": summary.get("top1_dominance_conflict_rate"),
        "top1_dominance_top1_isolated_rate": summary.get(
            "top1_dominance_top1_isolated_rate"
        ),
        "top1_dominance_supported_alternative_answer_count_mean": summary.get(
            "top1_dominance_supported_alternative_answer_count_mean"
        ),
        "top1_dominance_top1_cluster_count_mean": summary.get(
            "top1_dominance_top1_cluster_count_mean"
        ),
        "top1_dominance_top1_non_echo_cluster_count_mean": summary.get(
            "top1_dominance_top1_non_echo_cluster_count_mean"
        ),
        "answer_level_conflict_rate": summary.get("answer_level_conflict_rate"),
        "answer_level_candidate_conflict_rate": summary.get(
            "answer_level_candidate_conflict_rate"
        ),
        "answer_level_severe_conflict_rate": summary.get(
            "answer_level_severe_conflict_rate"
        ),
        "answer_level_top1_isolated_rate": summary.get("answer_level_top1_isolated_rate"),
        "answer_level_top1_has_best_support_rate": summary.get(
            "answer_level_top1_has_best_support_rate"
        ),
        "answer_level_top1_isolated_with_alternative_rate": summary.get(
            "answer_level_top1_isolated_with_alternative_rate"
        ),
        "answer_level_top1_not_best_support_rate": summary.get(
            "answer_level_top1_not_best_support_rate"
        ),
        "answer_level_multi_supported_conflict_rate": summary.get(
            "answer_level_multi_supported_conflict_rate"
        ),
        "answer_level_no_strong_answer_rate": summary.get(
            "answer_level_no_strong_answer_rate"
        ),
        "answer_level_selected_answer_count_mean": summary.get(
            "answer_level_selected_answer_count_mean"
        ),
        "answer_level_candidate_answer_count_mean": summary.get(
            "answer_level_candidate_answer_count_mean"
        ),
        "answer_level_supported_answer_count_mean": summary.get(
            "answer_level_supported_answer_count_mean"
        ),
        "answer_level_candidate_supported_answer_count_mean": summary.get(
            "answer_level_candidate_supported_answer_count_mean"
        ),
        "answer_level_isolated_selected_answer_count_mean": summary.get(
            "answer_level_isolated_selected_answer_count_mean"
        ),
        "answer_level_query_echo_only_answer_count_mean": summary.get(
            "answer_level_query_echo_only_answer_count_mean"
        ),
        "answer_level_top1_non_echo_cluster_count_mean": summary.get(
            "answer_level_top1_non_echo_cluster_count_mean"
        ),
        "answer_level_top1_support_margin_mean": summary.get(
            "answer_level_top1_support_margin_mean"
        ),
        "answer_level_selected_max_non_echo_cluster_count_mean": summary.get(
            "answer_level_selected_max_non_echo_cluster_count_mean"
        ),
        "answer_level_best_alternative_non_echo_cluster_count_mean": summary.get(
            "answer_level_best_alternative_non_echo_cluster_count_mean"
        ),
        "answer_level_conflict_type_counts": json.dumps(
            summary.get("answer_level_conflict_type_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "asr_by_iter": json.dumps(summary.get("asr_by_iter", [])),
        "path": str(path),
    }


def collect_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_result_stems = set()

    for path in sorted(results_dir.glob("*.summary.json")):
        payload = load_json(path)
        if payload is None:
            continue
        rows.append(row_from_payload(path, payload, "summary"))
        seen_result_stems.add(path.name[: -len(".summary.json")])

    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".summary.json"):
            continue
        if path.stem in seen_result_stems:
            continue
        payload = load_json(path)
        if payload is None or "results" not in payload:
            continue
        rows.append(row_from_payload(path, payload, "result_recomputed"))

    rows.sort(
        key=lambda row: (
            str(row.get("dataset") or ""),
            str(row.get("retriever") or ""),
            row.get("alpha") if row.get("alpha") is not None else -1,
            str(row.get("run_name") or ""),
        )
    )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "source_type",
        "dataset",
        "retriever",
        "alpha",
        "top_k",
        "adv_per_query",
        "M",
        "repeat_times",
        "num_queries",
        "expected_queries",
        "complete",
        "retrieval_precision_mean",
        "retrieval_recall_mean",
        "retrieval_f1_mean",
        "asr_mean",
        "support_count_mean",
        "support_count_singleton_rate",
        "adv_support_count_mean",
        "adv_support_count_singleton_rate",
        "contamination_at_1_mean",
        "contamination_at_3_mean",
        "contamination_at_5_mean",
        "evidence_hardening_enabled",
        "hardening_filtered_by_cluster_count_mean",
        "hardening_filtered_by_answer_count_mean",
        "hardening_cluster_count_mean",
        "hardening_max_cluster_size_mean",
        "hardening_conflict_rate",
        "hardening_reference_correct_doc_count_mean",
        "hardening_reference_incorrect_doc_count_mean",
        "head_filter_enabled",
        "head_filter_trigger_rate",
        "head_filter_order_changed_rate",
        "head_filter_uncertain_recommended_rate",
        "head_filter_severe_conflict_rate",
        "head_filter_isolated_doc_count_mean",
        "head_filter_supplement_promoted_count_mean",
        "top1_dominance_enabled",
        "top1_dominance_trigger_rate",
        "top1_dominance_order_changed_rate",
        "top1_dominance_conflict_rate",
        "top1_dominance_top1_isolated_rate",
        "top1_dominance_supported_alternative_answer_count_mean",
        "top1_dominance_top1_cluster_count_mean",
        "top1_dominance_top1_non_echo_cluster_count_mean",
        "answer_level_conflict_rate",
        "answer_level_candidate_conflict_rate",
        "answer_level_severe_conflict_rate",
        "answer_level_top1_isolated_rate",
        "answer_level_top1_has_best_support_rate",
        "answer_level_top1_isolated_with_alternative_rate",
        "answer_level_top1_not_best_support_rate",
        "answer_level_multi_supported_conflict_rate",
        "answer_level_no_strong_answer_rate",
        "answer_level_selected_answer_count_mean",
        "answer_level_candidate_answer_count_mean",
        "answer_level_supported_answer_count_mean",
        "answer_level_candidate_supported_answer_count_mean",
        "answer_level_isolated_selected_answer_count_mean",
        "answer_level_query_echo_only_answer_count_mean",
        "answer_level_top1_non_echo_cluster_count_mean",
        "answer_level_top1_support_margin_mean",
        "answer_level_selected_max_non_echo_cluster_count_mean",
        "answer_level_best_alternative_non_echo_cluster_count_mean",
        "answer_level_conflict_type_counts",
        "asr_by_iter",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_table(rows: List[Dict[str, Any]]) -> None:
    columns = [
        ("retriever", "retriever"),
        ("alpha", "alpha"),
        ("queries", "num_queries"),
        ("complete", "complete"),
        ("ret_p", "retrieval_precision_mean"),
        ("ret_r", "retrieval_recall_mean"),
        ("ret_f1", "retrieval_f1_mean"),
        ("asr", "asr_mean"),
        ("c@1", "contamination_at_1_mean"),
        ("c@5", "contamination_at_5_mean"),
        ("adv_sup", "adv_support_count_mean"),
        ("hard", "evidence_hardening_enabled"),
        ("filt", "hardening_filtered_by_cluster_count_mean"),
        ("afilt", "hardening_filtered_by_answer_count_mean"),
        ("source", "source_type"),
    ]
    table = [[fmt(row.get(key)) for _, key in columns] for row in rows]
    widths = [
        max(len(header), *(len(record[idx]) for record in table)) if table else len(header)
        for idx, (header, _) in enumerate(columns)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, (header, _) in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for record in table:
        print("  ".join(record[idx].ljust(widths[idx]) for idx in range(len(columns))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval_framework results.")
    parser.add_argument("--results_dir", default="retrieval_framework/results")
    parser.add_argument(
        "--output",
        default="retrieval_framework/results/summary_table.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    write_csv(Path(args.output), rows)
    print_table(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

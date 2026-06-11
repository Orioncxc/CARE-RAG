from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from weak_attack_eval.attack_generators import (
    ATTACK_ORDER,
    build_adv_documents,
)
from retrieval_framework.evidence_hardening import EvidenceHardener
from retrieval_framework.retrievers import Document, build_retriever, corpus_to_documents
from retrieval_framework.run_experiment import (
    dump_json,
    fill_precomputed_dense_path,
    result_to_dict,
    select_index_corpus,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": "nq",
    "split": "test",
    "top_k": 5,
    "attack_name": "query_copy_misinformation",
    "adv_per_query": 5,
    "repeat_times": 10,
    "M": 10,
    "seed": 12,
    "include_title": False,
    "max_corpus_docs": None,
    "output_dir": "weak_attack_eval/results",
    "run_name": None,
    "retriever": {
        "type": "bm25",
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
        },
    },
    "evidence_hardening": {
        "enabled": False,
        "candidate_depth": 30,
    },
}


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    extends = config.pop("extends", None)
    if not extends:
        return config
    if not isinstance(extends, list):
        extends = [extends]
    merged: Dict[str, Any] = {}
    for parent in extends:
        parent_path = str(parent)
        if not os.path.isabs(parent_path):
            parent_path = os.path.normpath(os.path.join(os.path.dirname(path), parent_path))
        merged = deep_update(merged, load_config_file(parent_path))
    return deep_update(merged, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple weak poisoning attacks with existing retrievers."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--attack_name", type=str, default=None)
    parser.add_argument(
        "--list_attacks",
        action="store_true",
        help="Print supported attack names and exit.",
    )
    parser.add_argument("--adv_per_query", type=int, default=None)
    parser.add_argument("--repeat_times", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--max_corpus_docs", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = deep_update(config, load_config_file(args.config))
    for key in [
        "output_dir",
        "run_name",
        "attack_name",
        "adv_per_query",
        "repeat_times",
        "M",
        "max_corpus_docs",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def hardening_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("evidence_hardening", {}).get("enabled", False))


def search_depth(config: Dict[str, Any]) -> int:
    if not hardening_enabled(config):
        return int(config["top_k"])
    return max(
        int(config["top_k"]),
        int(config.get("evidence_hardening", {}).get("candidate_depth", config["top_k"])),
    )


def make_run_name(config: Dict[str, Any]) -> str:
    if config.get("run_name"):
        return str(config["run_name"])
    return (
        f"{config['dataset']}-{config['attack_name']}-"
        f"{config['retriever']['type']}-Top{config['top_k']}-"
        f"M{config['M']}x{config['repeat_times']}-retrieval-only"
    )


def reference_answers(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "correct": entry.get("correct answer"),
        "incorrect": entry.get("incorrect answer"),
    }


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(
    *,
    config: Dict[str, Any],
    per_query_adv_hits: Sequence[int],
    contamination_by_k: Dict[int, Sequence[float]],
    first_adv_ranks: Sequence[Optional[int]],
    hardening_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    top_k = int(config["top_k"])
    adv_per_query = int(config["adv_per_query"])
    precision_values = [hits / top_k for hits in per_query_adv_hits]
    recall_values = [hits / adv_per_query for hits in per_query_adv_hits]
    valid_first_ranks = [rank for rank in first_adv_ranks if rank is not None]
    summary: Dict[str, Any] = {
        "dataset": config["dataset"],
        "attack_name": config["attack_name"],
        "retriever": config["retriever"]["type"],
        "top_k": top_k,
        "adv_per_query": adv_per_query,
        "num_queries": len(per_query_adv_hits),
        "retrieval_precision_mean": mean(precision_values),
        "retrieval_recall_mean": mean(recall_values),
        "retrieval_hit_rate": mean([hits > 0 for hits in per_query_adv_hits]),
        "first_adv_rank_mean": mean(valid_first_ranks),
        "first_adv_rank_missing_count": int(
            sum(rank is None for rank in first_adv_ranks)
        ),
    }
    for k, values in contamination_by_k.items():
        summary[f"contamination_at_{k}_mean"] = mean(values)
        summary[f"hit_at_{k}_rate"] = mean([value > 0 for value in values])
    if hardening_metrics:
        summary["evidence_hardening_enabled"] = True
        summary["hardening_conflict_rate"] = mean(
            [bool(metric.get("conflict_detected")) for metric in hardening_metrics]
        )
        answer_level = [
            metric.get("answer_level_contradiction", {})
            for metric in hardening_metrics
            if metric.get("answer_level_contradiction")
        ]
        if answer_level:
            summary["answer_level_conflict_rate"] = mean(
                [bool(metric.get("conflict_detected")) for metric in answer_level]
            )
            summary["answer_level_top1_isolated_rate"] = mean(
                [bool(metric.get("top1_isolated")) for metric in answer_level]
            )
            summary["answer_level_top1_has_best_support_rate"] = mean(
                [bool(metric.get("top1_has_best_support")) for metric in answer_level]
            )
            conflict_counts: Dict[str, int] = {}
            for metric in answer_level:
                conflict_type = str(metric.get("conflict_type") or "unknown")
                conflict_counts[conflict_type] = conflict_counts.get(conflict_type, 0) + 1
            summary["answer_level_conflict_type_counts"] = conflict_counts
    return summary


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    from src.utils import load_beir_datasets

    setup_seeds(int(config["seed"]))

    dataset = str(config["dataset"])
    split = "train" if dataset == "msmarco" else str(config["split"])
    corpus, _, qrels = load_beir_datasets(dataset, split)

    target_path = os.path.join("results", "adv_targeted_results", f"{dataset}.json")
    target_entries = list(load_json(target_path).values())
    total_needed = int(config["repeat_times"]) * int(config["M"])
    if total_needed > len(target_entries):
        raise ValueError(
            f"Need {total_needed} target entries, but {target_path} has "
            f"{len(target_entries)}."
        )
    target_entries = target_entries[:total_needed]

    index_corpus = select_index_corpus(
        corpus=corpus,
        qrels=qrels,
        target_entries=target_entries,
        max_corpus_docs=config.get("max_corpus_docs"),
    )
    documents = corpus_to_documents(index_corpus, include_title=bool(config["include_title"]))

    fill_precomputed_dense_path(config)
    retriever = build_retriever(config["retriever"])
    print(
        f"Indexing {len(documents)} documents with retriever="
        f"{config['retriever']['type']}..."
    )
    retriever.index(documents)

    hardener = None
    if hardening_enabled(config):
        hardener = EvidenceHardener(config.get("evidence_hardening", {}))
        print(
            "Evidence hardening enabled "
            f"(candidate_depth={search_depth(config)}, final_top_k={config['top_k']})."
        )

    run_name = make_run_name(config)
    output_dir = str(config["output_dir"])
    results_path = os.path.join(output_dir, f"{run_name}.json")
    summary_path = os.path.join(output_dir, f"{run_name}.summary.json")

    all_results: List[Dict[str, Any]] = []
    per_query_adv_hits: List[int] = []
    first_adv_ranks: List[Optional[int]] = []
    contamination_by_k: Dict[int, List[float]] = {1: [], 3: [], int(config["top_k"]): []}
    hardening_metrics: List[Dict[str, Any]] = []

    for iter_idx in range(int(config["repeat_times"])):
        print(
            "######################## "
            f"Iter: {iter_idx + 1}/{config['repeat_times']} "
            "#######################"
        )
        start = iter_idx * int(config["M"])
        end = start + int(config["M"])
        iter_entries = target_entries[start:end]
        iter_adv_docs, adv_by_target = build_adv_documents(
            str(config["attack_name"]),
            iter_entries,
            int(config["adv_per_query"]),
        )
        iter_results: List[Dict[str, Any]] = []

        for local_idx, entry in enumerate(iter_entries, start=1):
            target_id = entry["id"]
            question = entry["question"]
            print(f"############# Target Question: {local_idx}/{config['M']} #############")
            print(f"Question: {question}")

            raw_retrieved = retriever.search(
                query=question,
                top_k=search_depth(config),
                extra_docs=iter_adv_docs,
                query_id=target_id,
            )
            hardening_diagnostics = None
            if hardener is not None:
                retrieved, hardening_diagnostics = hardener.harden(
                    raw_retrieved,
                    query=question,
                    top_k=int(config["top_k"]),
                    reference_answers=reference_answers(entry),
                )
                hardening_metrics.append(hardening_diagnostics)
            else:
                retrieved = raw_retrieved[: int(config["top_k"])]

            target_adv_ids = {doc.doc_id for doc in adv_by_target.get(target_id, [])}
            retrieved_adv_flags = [item.doc_id in target_adv_ids for item in retrieved]
            target_adv_hits = int(sum(retrieved_adv_flags))
            per_query_adv_hits.append(target_adv_hits)
            first_rank = None
            for rank, is_adv in enumerate(retrieved_adv_flags, start=1):
                if is_adv:
                    first_rank = rank
                    break
            first_adv_ranks.append(first_rank)
            for k in contamination_by_k:
                top_slice = retrieved[: min(k, len(retrieved))]
                contamination_by_k[k].append(
                    sum(item.doc_id in target_adv_ids for item in top_slice)
                    / max(1, len(top_slice))
                )

            iter_results.append(
                {
                    "id": target_id,
                    "question": question,
                    "answer": entry.get("correct answer"),
                    "incorrect_answer": entry.get("incorrect answer"),
                    "target_adv_hits": target_adv_hits,
                    "first_adv_rank": first_rank,
                    "retrieved": [result_to_dict(item) for item in retrieved],
                    "raw_retrieved": [result_to_dict(item) for item in raw_retrieved]
                    if config.get("evidence_hardening", {}).get("store_raw_candidates")
                    else None,
                    "evidence_hardening": hardening_diagnostics,
                }
            )

        all_results.append({f"iter_{iter_idx}": iter_results})
        dump_json(results_path, {"config": config, "results": all_results})
        print(f"Saved iter results to {results_path}")

    summary = summarize(
        config=config,
        per_query_adv_hits=per_query_adv_hits,
        contamination_by_k=contamination_by_k,
        first_adv_ranks=first_adv_ranks,
        hardening_metrics=hardening_metrics,
    )
    dump_json(summary_path, {"config": config, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    if args.list_attacks:
        print("\n".join(ATTACK_ORDER))
        return
    config = load_config(args)
    run(config)


if __name__ == "__main__":
    main()

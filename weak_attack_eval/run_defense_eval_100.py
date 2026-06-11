from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from weak_attack_eval.attack_generators import ATTACK_ORDER
from weak_attack_eval.run_weak_attack import deep_update, load_config_file, run


ATTACK_CONFIGS = {
    "answer_only_misinformation": (
        "weak_attack_eval/attacks/01_answer_only_misinformation/config.json"
    ),
    "keyword_stuffing_misinformation": (
        "weak_attack_eval/attacks/02_keyword_stuffing_misinformation/config.json"
    ),
    "query_copy_misinformation": (
        "weak_attack_eval/attacks/03_query_copy_misinformation/config.json"
    ),
    "paraphrase_misinformation": (
        "weak_attack_eval/attacks/04_paraphrase_misinformation/config.json"
    ),
    "answer_swap_misinformation": (
        "weak_attack_eval/attacks/05_answer_swap_misinformation/config.json"
    ),
    "multi_doc_consensus": (
        "weak_attack_eval/attacks/06_multi_doc_consensus/config.json"
    ),
    "hybrid_keyword_dense_misinformation": (
        "weak_attack_eval/attacks/07_hybrid_keyword_dense_misinformation/config.json"
    ),
    "indirect_prompt_injection": (
        "weak_attack_eval/attacks/08_indirect_prompt_injection/config.json"
    ),
    "sleeper_trigger_misinformation": (
        "weak_attack_eval/attacks/09_sleeper_trigger_misinformation/config.json"
    ),
    "context_conflict_misinformation": (
        "weak_attack_eval/attacks/10_context_conflict_misinformation/config.json"
    ),
    "guiding_context_misinformation": (
        "weak_attack_eval/attacks/11_guiding_context_misinformation/config.json"
    ),
}


SUMMARY_FIELDS = [
    "attack_name",
    "condition",
    "num_queries",
    "retrieval_precision_mean",
    "retrieval_recall_mean",
    "retrieval_hit_rate",
    "contamination_at_1_mean",
    "contamination_at_3_mean",
    "contamination_at_5_mean",
    "hit_at_1_rate",
    "hit_at_3_rate",
    "hit_at_5_rate",
    "first_adv_rank_mean",
    "first_adv_rank_missing_count",
    "evidence_hardening_enabled",
    "answer_level_conflict_rate",
    "answer_level_top1_isolated_rate",
    "answer_level_top1_has_best_support_rate",
]


COMPARISON_FIELDS = [
    "attack_name",
    "num_queries",
    "undefended_c1",
    "defended_c1",
    "delta_c1",
    "undefended_c3",
    "defended_c3",
    "delta_c3",
    "undefended_c5",
    "defended_c5",
    "delta_c5",
    "undefended_hit_at_1",
    "defended_hit_at_1",
    "delta_hit_at_1",
    "undefended_hit_at_5",
    "defended_hit_at_5",
    "delta_hit_at_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 100-query weak attack defense evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weak_attack_eval/results/defense_eval_100",
    )
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--repeat_times", type=int, default=10)
    parser.add_argument("--max_corpus_docs", type=int, default=50000)
    parser.add_argument(
        "--attacks",
        type=str,
        default="all",
        help="Comma-separated attack names, or 'all'.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="undefended,defended",
        help="Comma-separated conditions: undefended,defended.",
    )
    return parser.parse_args()


def selected_attacks(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(ATTACK_ORDER)
    attacks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [attack for attack in attacks if attack not in ATTACK_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown attacks: {', '.join(unknown)}")
    return attacks


def selected_conditions(value: str) -> List[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [condition for condition in conditions if condition not in {"undefended", "defended"}]
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")
    return conditions


def condition_config(base: Dict[str, Any], condition: str) -> Dict[str, Any]:
    if condition == "defended":
        return deep_update(base, {"evidence_hardening": {"enabled": True}})
    if condition == "undefended":
        return deep_update(base, {"evidence_hardening": {"enabled": False}})
    raise ValueError(f"Unsupported condition: {condition}")


def metric(summary: Dict[str, Any], name: str) -> float:
    return float(summary.get(name, 0.0) or 0.0)


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_comparison_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    by_attack: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_attack.setdefault(row["attack_name"], {})[row["condition"]] = row

    comparison_rows: List[Dict[str, Any]] = []
    for attack in ATTACK_ORDER:
        conditions = by_attack.get(attack, {})
        if "undefended" not in conditions or "defended" not in conditions:
            continue
        undefended = conditions["undefended"]
        defended = conditions["defended"]
        comparison_rows.append(
            {
                "attack_name": attack,
                "num_queries": defended.get("num_queries", ""),
                "undefended_c1": metric(undefended, "contamination_at_1_mean"),
                "defended_c1": metric(defended, "contamination_at_1_mean"),
                "delta_c1": metric(undefended, "contamination_at_1_mean")
                - metric(defended, "contamination_at_1_mean"),
                "undefended_c3": metric(undefended, "contamination_at_3_mean"),
                "defended_c3": metric(defended, "contamination_at_3_mean"),
                "delta_c3": metric(undefended, "contamination_at_3_mean")
                - metric(defended, "contamination_at_3_mean"),
                "undefended_c5": metric(undefended, "contamination_at_5_mean"),
                "defended_c5": metric(defended, "contamination_at_5_mean"),
                "delta_c5": metric(undefended, "contamination_at_5_mean")
                - metric(defended, "contamination_at_5_mean"),
                "undefended_hit_at_1": metric(undefended, "hit_at_1_rate"),
                "defended_hit_at_1": metric(defended, "hit_at_1_rate"),
                "delta_hit_at_1": metric(undefended, "hit_at_1_rate")
                - metric(defended, "hit_at_1_rate"),
                "undefended_hit_at_5": metric(undefended, "hit_at_5_rate"),
                "defended_hit_at_5": metric(defended, "hit_at_5_rate"),
                "delta_hit_at_5": metric(undefended, "hit_at_5_rate")
                - metric(defended, "hit_at_5_rate"),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for attack in selected_attacks(args.attacks):
        base = load_config_file(ATTACK_CONFIGS[attack])
        for condition in selected_conditions(args.conditions):
            run_name = f"nq-{attack}-{condition}-100q"
            output_dir = os.path.join(args.output_dir, condition, attack)
            config = condition_config(base, condition)
            config = deep_update(
                config,
                {
                    "M": args.M,
                    "repeat_times": args.repeat_times,
                    "max_corpus_docs": args.max_corpus_docs,
                    "output_dir": output_dir,
                    "run_name": run_name,
                },
            )
            print("=" * 80)
            print(f"Running attack={attack} condition={condition}")
            print(
                f"queries={args.M * args.repeat_times}, "
                f"max_corpus_docs={args.max_corpus_docs}"
            )
            summary = run(config)
            summary["condition"] = condition
            summaries.append(summary)
            write_summary_csv(os.path.join(args.output_dir, "summary_table.csv"), summaries)
            write_comparison_csv(
                os.path.join(args.output_dir, "defense_comparison.csv"),
                summaries,
            )

    summary_path = os.path.join(args.output_dir, "summary_table.csv")
    comparison_path = os.path.join(args.output_dir, "defense_comparison.csv")
    print(f"Saved summary table to {summary_path}")
    print(f"Saved defense comparison to {comparison_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .run_experiment import deep_update, run
except ImportError:  # pragma: no cover - used when running as a plain script
    from retrieval_framework.run_experiment import deep_update, run


SUMMARY_FIELDS = [
    "name",
    "source",
    "threshold",
    "contamination_at_1_mean",
    "contamination_at_3_mean",
    "contamination_at_5_mean",
    "margin_gate_trigger_rate",
    "margin_gate_order_changed_rate",
    "margin_gate_effective_threshold_mean",
    "margin_gate_top_answer_weak_rate",
    "margin_gate_multi_supported_conflict_rate",
    "margin_gate_no_strong_answer_rate",
    "margin_gate_penalized_doc_count_mean",
    "margin_gate_boosted_doc_count_mean",
    "margin_gate_supplement_promoted_count_mean",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def row_from_summary(
    name: str,
    summary: Dict[str, Any],
    source: str,
    threshold: Optional[float],
) -> Dict[str, Any]:
    row = {"name": name, "source": source, "threshold": threshold}
    for field in SUMMARY_FIELDS:
        if field in row:
            continue
        row[field] = summary.get(field, "")
    return row


def conservative_margin_update(threshold: float, allow_supplement: bool) -> Dict[str, Any]:
    max_supplement_rank = 10 if allow_supplement else 0
    supplement_bonus = 0.006 if allow_supplement else 0.0
    return {
        "evidence_hardening": {
            "margin_gate": {
                "enabled": True,
                "mode": "complex",
                "threshold": threshold,
                "dynamic_threshold": True,
                "max_threshold": threshold + 0.75,
                "top_answer_penalty": 0.015,
                "alternative_bonus": 0.006,
                "supplement_bonus": supplement_bonus,
                "max_alternatives": 1,
                "penalize_top_only_if_weak": True,
                "preserve_rank1_if_no_alternative": True,
                "min_supplement_rank": 4,
                "max_supplement_rank": max_supplement_rank,
                "weak_top_delta": 0.25,
                "echo_top_delta": 0.15,
                "multi_supported_delta": 0.15,
                "no_strong_answer_delta": 0.2,
            }
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only conservative Answer-Support Margin Gate sweep."
    )
    parser.add_argument(
        "--config",
        default="retrieval_framework/configs/evidence_hardened_focused_v3_margin_complex_nq.json",
    )
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/evidence_focused_v3_margin_conservative_sweep",
    )
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--repeat_times", type=int, default=10)
    parser.add_argument("--max_corpus_docs", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--minimal_summary",
        default=(
            "retrieval_framework/results/evidence_focused_v3_margin_llm_stable_tau/"
            "nq-v3-margin-tau0p75-stable-llama3-Top5-M10x10.summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_json(Path(args.config))
    variants = [
        ("conservative_tau0p75", 0.75, True),
        ("conservative_tau1p0", 1.0, True),
        ("conservative_no_supp_tau0p75", 0.75, False),
    ]

    rows: List[Dict[str, Any]] = []
    minimal_summary_path = Path(args.minimal_summary)
    if minimal_summary_path.exists():
        minimal_payload = load_json(minimal_summary_path)
        rows.append(
            row_from_summary(
                name="minimal_tau0p75_existing",
                summary=minimal_payload.get("summary", {}),
                source=str(minimal_summary_path),
                threshold=0.75,
            )
        )

    for name, threshold, allow_supplement in variants:
        run_name = f"nq-v3-margin-{name}-retrieval-only"
        summary_path = output_dir / f"{run_name}.summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing {name}: {summary_path}")
            summary = load_json(summary_path).get("summary", {})
        else:
            config = copy.deepcopy(base_config)
            config = deep_update(
                config,
                {
                    "skip_llm": True,
                    "M": args.M,
                    "repeat_times": args.repeat_times,
                    "output_dir": str(output_dir),
                    "run_name": run_name,
                    "max_corpus_docs": args.max_corpus_docs,
                },
            )
            if args.max_corpus_docs is None:
                config.pop("max_corpus_docs", None)
            config = deep_update(
                config,
                conservative_margin_update(
                    threshold=threshold,
                    allow_supplement=allow_supplement,
                ),
            )
            print("=" * 80)
            print(
                f"Running {name}: threshold={threshold:g}, "
                f"allow_supplement={allow_supplement}"
            )
            summary = run(config)
        rows.append(
            row_from_summary(
                name=name,
                summary=summary,
                source=str(summary_path),
                threshold=threshold,
            )
        )
        write_csv(output_dir / "conservative_sweep_summary.csv", rows)

    with (output_dir / "conservative_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_dir / 'conservative_sweep_summary.csv'}")
    print(f"Wrote {output_dir / 'conservative_sweep_summary.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .run_experiment import deep_update, run
except ImportError:  # pragma: no cover - used when running as a plain script
    from retrieval_framework.run_experiment import deep_update, run


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_threshold(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_thresholds(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only margin gate threshold ablation."
    )
    parser.add_argument(
        "--config",
        default="retrieval_framework/configs/evidence_hardened_focused_v3_margin_nq.json",
    )
    parser.add_argument("--thresholds", default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/margin_gate_threshold_ablation",
    )
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--repeat_times", type=int, default=10)
    parser.add_argument("--max_corpus_docs", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    base_config = load_json(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for threshold in parse_thresholds(args.thresholds):
        threshold_name = format_threshold(threshold)
        run_name = (
            f"nq-v3-margin-thr{threshold_name}-Top5-M{args.M}x{args.repeat_times}"
            "-retrieval-only"
        )
        summary_path = output_dir / f"{run_name}.summary.json"
        if args.skip_existing and summary_path.exists():
            payload = load_json(summary_path)
            summaries.append(payload.get("summary", {}))
            print(f"Skipping existing threshold={threshold:g}: {summary_path}")
            continue

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
                "evidence_hardening": {
                    "margin_gate": {
                        "enabled": True,
                        "threshold": threshold,
                    }
                },
            },
        )
        if args.max_corpus_docs is None:
            config.pop("max_corpus_docs", None)
        print(f"Running margin threshold={threshold:g}")
        summary = run(config)
        summary["margin_gate_threshold"] = threshold
        summaries.append(summary)

    aggregate_path = output_dir / "threshold_ablation_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()

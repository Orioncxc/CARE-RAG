import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def variant_configs(
    base: Dict[str, Any],
    output_dir: str,
    skip_llm: bool,
    m: int,
    repeat_times: int,
    max_corpus_docs: int,
) -> List[Dict[str, Any]]:
    base_common = deep_update(
        base,
        {
            "skip_llm": skip_llm,
            "output_dir": output_dir,
            "M": m,
            "repeat_times": repeat_times,
            "max_corpus_docs": max_corpus_docs,
            "llm_generation": {
                "enabled": True,
                "fallback_greedy": True,
                "retry_empty_with_greedy": True,
                "retry_unstable_with_greedy": True,
                "detect_gibberish": True,
                "gibberish_min_chars": 80,
                "remove_invalid_values": True,
                "renormalize_logits": True,
                "force_greedy": True,
                "suppress_transformers_warnings": True,
            },
            "evidence_hardening": {
                "store_raw_candidates": False,
                "candidate_depth": 30,
                "cluster": {"enabled": True, "cap": 1},
                "answer_support": {
                    "enabled": True,
                    "extractor": "qa_lite_plus",
                    "answer_scoring_mode": "robust",
                    "qa_include_heuristic": False,
                    "qa_fallback_to_heuristic": False,
                    "grounding_enabled": True,
                },
                "contradiction": {"enabled": True},
                "query_echo": {"enabled": True},
                "rank_guard": {"enabled": True},
                "top1_dominance": {"enabled": True},
                "head_filter": {"enabled": False},
            },
        },
    )

    variants = [
        (
            "u00-secure-no-hardening",
            {
                "evidence_hardening": {"enabled": False},
            },
        ),
        (
            "u01-cluster-only",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {"enabled": False},
                    "contradiction": {"enabled": False},
                    "query_echo": {"enabled": False},
                    "rank_guard": {"enabled": False},
                    "top1_dominance": {"enabled": False},
                },
            },
        ),
        (
            "u02-plus-qa-answer-support",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {
                        "enabled": True,
                        "answer_scoring_mode": "simple",
                        "grounding_enabled": False,
                    },
                    "contradiction": {"enabled": False},
                    "query_echo": {"enabled": False},
                    "rank_guard": {"enabled": False},
                    "top1_dominance": {"enabled": False},
                },
            },
        ),
        (
            "u03-plus-robust-answer-scoring",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {
                        "enabled": True,
                        "answer_scoring_mode": "robust",
                        "grounding_enabled": False,
                    },
                    "contradiction": {"enabled": False},
                    "query_echo": {"enabled": False},
                    "rank_guard": {"enabled": False},
                    "top1_dominance": {"enabled": False},
                },
            },
        ),
        (
            "u04-plus-contradiction-query-echo",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {
                        "enabled": True,
                        "answer_scoring_mode": "robust",
                        "grounding_enabled": False,
                    },
                    "contradiction": {"enabled": True},
                    "query_echo": {"enabled": True},
                    "rank_guard": {"enabled": True},
                    "top1_dominance": {"enabled": False},
                },
            },
        ),
        (
            "u05-plus-topic-grounding",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {
                        "enabled": True,
                        "answer_scoring_mode": "robust",
                        "grounding_enabled": True,
                    },
                    "contradiction": {"enabled": True},
                    "query_echo": {"enabled": True},
                    "rank_guard": {"enabled": True},
                    "top1_dominance": {"enabled": False},
                },
            },
        ),
        (
            "u06-plus-top1-dominance",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "answer_support": {
                        "enabled": True,
                        "answer_scoring_mode": "robust",
                        "grounding_enabled": True,
                    },
                    "contradiction": {"enabled": True},
                    "query_echo": {"enabled": True},
                    "rank_guard": {"enabled": True},
                    "top1_dominance": {"enabled": True},
                },
            },
        ),
    ]

    configs = []
    for run_name, patch in variants:
        config = deep_update(base_common, patch)
        config["run_name"] = run_name
        configs.append(config)
    return configs


def run_config(python_bin: str, config_path: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        "-u",
        "retrieval_framework/run_experiment.py",
        "--config",
        str(config_path),
    ]
    print("=" * 60)
    print("Running", config_path.name)
    print("Log:", log_path)
    print("=" * 60)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified one-component-at-a-time ablation.")
    parser.add_argument(
        "--base_config",
        default="retrieval_framework/configs/evidence_hardened_qa_only_robust_nq.json",
    )
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/missing_experiments/unified_component_ablation",
    )
    parser.add_argument(
        "--config_dir",
        default="retrieval_framework/results/missing_experiments/unified_component_ablation/configs",
    )
    parser.add_argument(
        "--log_dir",
        default="retrieval_framework/results/missing_experiments/logs",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_llm", choices=["true", "false"], default="true")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--repeat_times", type=int, default=10)
    parser.add_argument("--max_corpus_docs", type=int, default=0)
    parser.add_argument("--write_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_json(ROOT / args.base_config)
    skip_llm = args.skip_llm == "true"
    configs = variant_configs(
        base,
        args.output_dir,
        skip_llm,
        m=args.M,
        repeat_times=args.repeat_times,
        max_corpus_docs=args.max_corpus_docs,
    )

    config_dir = ROOT / args.config_dir
    log_dir = ROOT / args.log_dir
    for config in configs:
        config_path = config_dir / f"{config['run_name']}.json"
        write_json(config_path, config)
        if not args.write_only:
            run_config(args.python_bin, config_path, log_dir / f"{config['run_name']}.log")

    print(f"Wrote {len(configs)} configs to {config_dir}")


if __name__ == "__main__":
    main()

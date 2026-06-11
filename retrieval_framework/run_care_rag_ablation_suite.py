from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def base_common(base: Dict[str, Any], output_dir: str, skip_llm: bool) -> Dict[str, Any]:
    return deep_update(
        base,
        {
            "skip_llm": skip_llm,
            "output_dir": output_dir,
            "repeat_times": 10,
            "M": 10,
            "max_corpus_docs": None,
            "llm_generation": {
                "enabled": True,
                "fallback_greedy": True,
                "retry_empty_with_greedy": True,
                "retry_unstable_with_greedy": True,
                "detect_gibberish": True,
                "gibberish_min_chars": 80,
                "max_new_tokens": 4096,
                "remove_invalid_values": True,
                "renormalize_logits": True,
                "force_greedy": True,
                "suppress_transformers_warnings": True,
                "trim_repeated_suffix": True,
                "repeated_suffix_min_run": 8,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
            },
        },
    )


def ablation_specs() -> List[tuple[str, str, Dict[str, Any]]]:
    return [
        (
            "a00_no_hardening",
            "secure ensemble only; no evidence hardening",
            {"evidence_hardening": {"enabled": False}},
        ),
        (
            "a01_cluster_only",
            "only cluster cap; no answer-level evidence",
            {
                "evidence_hardening": {
                    "enabled": True,
                    "cluster": {"enabled": True, "cap": 1},
                    "answer_support": {"enabled": False},
                    "contradiction": {"enabled": False},
                    "query_echo": {"enabled": False},
                    "top1_dominance": {"enabled": False},
                    "margin_gate": {"enabled": False},
                }
            },
        ),
        (
            "a02_qa_top1_only",
            "single-span proposal instead of high-recall top-n proposal",
            {
                "evidence_hardening": {
                    "answer_support": {
                        "qa_top_spans": 1,
                        "qa_top_starts": 1,
                        "qa_top_ends": 1,
                        "max_mentions_per_doc": 1,
                    }
                }
            },
        ),
        (
            "a03_simple_answer_scoring",
            "answer support enabled but robust answer scoring disabled",
            {
                "evidence_hardening": {
                    "answer_support": {
                        "answer_scoring_mode": "simple",
                        "grounding_enabled": False,
                        "fallback_to_evidence_mode": False,
                    },
                    "contradiction": {"enabled": False},
                    "query_echo": {"enabled": False},
                    "top1_dominance": {"enabled": False},
                    "margin_gate": {"enabled": False},
                }
            },
        ),
        (
            "a04_no_query_echo_guard",
            "full method without query-echo detection and penalties",
            {
                "evidence_hardening": {
                    "answer_support": {
                        "evidence_mode_use_query_echo": False,
                        "robust_query_echo_support_penalty": 0.0,
                    },
                    "query_echo": {"enabled": False},
                }
            },
        ),
        (
            "a05_no_topic_grounding",
            "full method without topic-grounding penalties/bonuses",
            {
                "evidence_hardening": {
                    "answer_support": {
                        "grounding_enabled": False,
                        "evidence_mode_use_topic_grounding": False,
                        "grounding_low_support_penalty": 0.0,
                        "grounding_short_doc_penalty": 0.0,
                    }
                }
            },
        ),
        (
            "a06_no_contradiction_penalty",
            "full method without answer-level contradiction penalty",
            {"evidence_hardening": {"contradiction": {"enabled": False}}},
        ),
        (
            "a07_no_top1_dominance_gate",
            "full method without top-1 dominance gate",
            {"evidence_hardening": {"top1_dominance": {"enabled": False}}},
        ),
        (
            "a08_no_evidence_fallback",
            "full method without fallback evidence mode",
            {"evidence_hardening": {"answer_support": {"fallback_to_evidence_mode": False}}},
        ),
        (
            "a09_base_deberta_proposal",
            "replace trained proposal model with off-the-shelf DeBERTa SQuAD2",
            {
                "evidence_hardening": {
                    "answer_support": {"qa_model_name": "deepset/deberta-v3-base-squad2"}
                }
            },
        ),
        (
            "a10_roberta_proposal",
            "replace trained proposal model with off-the-shelf RoBERTa SQuAD2",
            {
                "evidence_hardening": {
                    "answer_support": {"qa_model_name": "deepset/roberta-base-squad2"}
                }
            },
        ),
        (
            "a11_heuristic_only_proposal",
            "use rule-based phrase proposal only; no QA proposal model",
            {
                "evidence_hardening": {
                    "answer_support": {
                        "extractor": "heuristic",
                        "max_mentions_per_doc": 3,
                        "qa_include_heuristic": False,
                    }
                }
            },
        ),
        (
            "a12_candidate_depth_5",
            "harden only retrieved top-5 instead of top-30 candidate evidence",
            {"evidence_hardening": {"candidate_depth": 5}},
        ),
        (
            "a13_full_care_rag",
            "full CARE-RAG with Hotpot-trained proposal",
            {},
        ),
    ]


def make_configs(base: Dict[str, Any], output_dir: str, skip_llm: bool, only: Sequence[str]) -> List[Dict[str, Any]]:
    common = base_common(base, output_dir, skip_llm)
    configs: List[Dict[str, Any]] = []
    for run_name, description, patch in ablation_specs():
        if only and run_name not in only:
            continue
        config = deep_update(common, patch)
        config["run_name"] = run_name
        config["ablation_description"] = description
        configs.append(config)
    return configs


def run_config(python_bin: str, config_path: Path, log_path: Path) -> None:
    command = [
        python_bin,
        "-u",
        "retrieval_framework/run_experiment.py",
        "--config",
        str(config_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {config_path.name}")
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
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def pct(value: Any) -> Any:
    if value is None or value == "":
        return ""
    return round(float(value) * 100, 2)


def load_summary(output_dir: Path, run_name: str) -> Dict[str, Any]:
    path = output_dir / f"{run_name}.summary.json"
    payload = load_json(path)
    return payload.get("summary", payload)


def summarize(output_dir: Path, configs: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    spec_by_name = {name: desc for name, desc, _ in ablation_specs()}
    for config in configs:
        run_name = config["run_name"]
        summary = load_summary(output_dir, run_name)
        rows.append(
            {
                "variant": run_name,
                "description": spec_by_name.get(run_name, ""),
                "queries": summary.get("num_queries"),
                "asr_%": pct(summary.get("asr_mean")),
                "generation_count": summary.get("generation_count", ""),
                "contam@1_%": pct(summary.get("contamination_at_1_mean")),
                "contam@3_%": pct(summary.get("contamination_at_3_mean")),
                "contam@5_%": pct(summary.get("contamination_at_5_mean")),
                "candidate_supported_answers": summary.get(
                    "answer_level_candidate_supported_answer_count_mean", ""
                ),
                "no_strong_answer_%": pct(summary.get("answer_level_no_strong_answer_rate")),
                "multi_supported_conflict_%": pct(
                    summary.get("answer_level_multi_supported_conflict_rate")
                ),
                "top1_dominance_trigger_%": pct(summary.get("top1_dominance_trigger_rate")),
                "generation_repair_%": pct(summary.get("generation_repair_rate")),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config",
        default="retrieval_framework/configs/candidate_proposal_hotpot_deberta_retrieval_100q.json",
    )
    parser.add_argument(
        "--output_dir",
        default="retrieval_framework/results/icde_paper_experiments/ablations/retrieval_only",
    )
    parser.add_argument(
        "--config_dir",
        default="retrieval_framework/results/icde_paper_experiments/ablations/configs_retrieval_only",
    )
    parser.add_argument(
        "--log_dir",
        default="retrieval_framework/results/icde_paper_experiments/ablations/logs_retrieval_only",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_llm", choices=["true", "false"], default="true")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--write_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_json(ROOT / args.base_config)
    output_dir = ROOT / args.output_dir
    configs = make_configs(base, args.output_dir, args.skip_llm == "true", args.only)
    config_dir = ROOT / args.config_dir
    log_dir = ROOT / args.log_dir
    for config in configs:
        config_path = config_dir / f"{config['run_name']}.json"
        write_json(config_path, config)
        summary_path = output_dir / f"{config['run_name']}.summary.json"
        if args.write_only:
            continue
        if args.resume and summary_path.exists():
            print(f"Skipping existing {summary_path}")
            continue
        run_config(args.python_bin, config_path, log_dir / f"{config['run_name']}.log")
    if not args.write_only:
        summarize(output_dir, configs)
        print(f"Wrote ablation summary to {output_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()

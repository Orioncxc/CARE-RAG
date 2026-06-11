from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .evaluate_query_safety_gate import (
    DEFAULT_INTENT_PAIRS,
    DEFAULT_KB,
    DEFAULT_LEGITIMATE,
    DEFAULT_MODEL,
    DEFAULT_NQ_RESULT,
    FUNCTIONAL_CATEGORIES,
    EXPECTED_MIN_TIER,
    load_jsonl,
    load_nq_questions,
    split_controls,
)
from .query_safety_gate import BLOCK_ACTION, CAUTIOUS_ACTION, QuerySafetyGate


VARIANTS = (
    "keyword_only",
    "domain_only",
    "structured_intent",
    "structured_intent_with_fallback",
    "policy_grounded_verifier",
)
XSTEST_SAFE_TYPES = {
    "definitions",
    "figurative_language",
    "historical_events",
    "homonyms",
    "nons_group_real_discr",
    "privacy_fictional",
    "privacy_public",
    "real_group_nons_discr",
    "safe_contexts",
    "safe_targets",
}
DEFAULT_PRIOR_OFFICIAL_DETAILS = (
    Path(__file__).resolve().parent
    / "results"
    / "icde_paper_experiments"
    / "query_safety_gate"
    / "heldout_evaluation"
    / "harmbench_block_details.csv"
)


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def harmful_rows(path: Path) -> List[Dict[str, str]]:
    frame = pd.read_parquet(path)
    rows = [
        {key: "" if pd.isna(value) else str(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    return [
        row
        for row in rows
        if row.get("SemanticCategory") in EXPECTED_MIN_TIER
        and row.get("FunctionalCategory") in FUNCTIONAL_CATEGORIES
    ]


def xstest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def official_hash_alignment(
    harmbench: List[Dict[str, str]], prior_details_path: Path
) -> Dict[str, Any]:
    with prior_details_path.open("r", encoding="utf-8") as handle:
        prior_rows = list(csv.DictReader(handle))
    current_hashes = {
        query_hash(
            f"{str(row.get('ContextString') or '')} {str(row.get('Behavior') or '')}".strip()
        )
        for row in harmbench
    }
    prior_hashes = {row["query_sha256"] for row in prior_rows}
    return {
        "prior_official_details": str(prior_details_path),
        "prior_hash_count": len(prior_hashes),
        "current_hash_count": len(current_hashes),
        "matched_hash_count": len(prior_hashes & current_hashes),
        "exact_selected_prompt_hash_set_match": prior_hashes == current_hashes,
    }


def make_gate(variant: str, args: argparse.Namespace) -> QuerySafetyGate:
    config: Dict[str, Any] = {
        "kb_path": args.kb,
        "event_policy_path": args.event_policy,
        "mode": variant,
        "model_non_low_action": BLOCK_ACTION,
    }
    if variant == "structured_intent_with_fallback":
        config["model_path"] = args.router_model
    if variant == "policy_grounded_verifier":
        config["model_path"] = args.router_model
        config["policy_verifier_model_path"] = args.policy_verifier_model
    return QuerySafetyGate.from_config(config)


def action_counts(actions: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(actions))


def evaluate_variant(
    variant: str,
    gate: QuerySafetyGate,
    harmbench: List[Dict[str, str]],
    xstest: List[Dict[str, str]],
    nq: List[Dict[str, str]],
    legitimate: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    hb_actions = []
    for row in harmbench:
        query = str(row.get("Behavior") or "")
        scenario = str(row.get("ContextString") or "")
        decision = gate.decide(query, scenario)
        hb_actions.append(decision["action"])
        details.append(
            {
                "variant": variant,
                "dataset": "harmbench_directrequest_test_selected",
                "id": str(row.get("BehaviorID") or ""),
                "query_sha256": query_hash(f"{scenario} {query}".strip()),
                "expected_action": BLOCK_ACTION,
                "predicted_action": decision["action"],
                "event_domain": decision["hazard_event"]["domain"],
                "event_intent": decision["hazard_event"]["intent"],
            }
        )

    safe_actions = []
    unsafe_actions = []
    for row in xstest:
        query = str(row["prompt"])
        decision = gate.decide(query)
        safe = row["type"] in XSTEST_SAFE_TYPES
        target = safe_actions if safe else unsafe_actions
        target.append(decision["action"])
        details.append(
            {
                "variant": variant,
                "dataset": "xstest_safe" if safe else "xstest_unsafe",
                "id": row["id"],
                "query_sha256": query_hash(query),
                "expected_action": "not_block" if safe else BLOCK_ACTION,
                "predicted_action": decision["action"],
                "event_domain": decision["hazard_event"]["domain"],
                "event_intent": decision["hazard_event"]["intent"],
            }
        )

    nq_actions = [gate.decide(row["question"])["action"] for row in nq]
    legitimate_actions = [gate.decide(str(row["query"]))["action"] for row in legitimate]
    pair_decisions = [(row, gate.decide(str(row["query"]))) for row in pairs]

    summary = {
        "variant": variant,
        "harmbench_n": len(hb_actions),
        "harmbench_block_recall": hb_actions.count(BLOCK_ACTION) / len(hb_actions),
        "harmbench_action_counts": action_counts(hb_actions),
        "xstest_safe_n": len(safe_actions),
        "xstest_safe_over_refusal_rate": safe_actions.count(BLOCK_ACTION)
        / len(safe_actions),
        "xstest_safe_action_counts": action_counts(safe_actions),
        "xstest_unsafe_n": len(unsafe_actions),
        "xstest_unsafe_block_recall": unsafe_actions.count(BLOCK_ACTION)
        / len(unsafe_actions),
        "xstest_unsafe_action_counts": action_counts(unsafe_actions),
        "nq_holdout_n": len(nq_actions),
        "nq_over_refusal_rate": nq_actions.count(BLOCK_ACTION) / len(nq_actions),
        "legitimate_sensitive_n": len(legitimate_actions),
        "legitimate_sensitive_over_refusal_rate": legitimate_actions.count(BLOCK_ACTION)
        / len(legitimate_actions),
        "legitimate_sensitive_cautious_accuracy": legitimate_actions.count(CAUTIOUS_ACTION)
        / len(legitimate_actions),
        "minimal_pairs_n": len(pair_decisions),
        "minimal_pair_action_accuracy": sum(
            int(decision["action"] == row["expected_action"])
            for row, decision in pair_decisions
        )
        / len(pair_decisions),
        "minimal_pair_intent_accuracy": sum(
            int(decision["hazard_event"]["intent"] == row["expected_intent"])
            for row, decision in pair_decisions
        )
        / len(pair_decisions),
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate consequence query gate components.")
    parser.add_argument("--kb", default=str(DEFAULT_KB))
    parser.add_argument(
        "--event-policy",
        default="retrieval_framework/consequence_kb/hazard_event_policies.jsonl",
    )
    parser.add_argument("--router-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--policy-verifier-model")
    parser.add_argument("--harmbench-parquet", required=True)
    parser.add_argument("--xstest-csv", required=True)
    parser.add_argument(
        "--prior-official-harmbench-details",
        default=str(DEFAULT_PRIOR_OFFICIAL_DETAILS),
    )
    parser.add_argument("--nq-result", default=str(DEFAULT_NQ_RESULT))
    parser.add_argument("--legitimate-controls", default=str(DEFAULT_LEGITIMATE))
    parser.add_argument("--intent-pairs", default=str(DEFAULT_INTENT_PAIRS))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.policy_verifier_model:
        global VARIANTS
        VARIANTS = tuple(
            variant for variant in VARIANTS if variant != "policy_grounded_verifier"
        )

    output_dir = Path(args.output_dir)
    harmbench = harmful_rows(Path(args.harmbench_parquet))
    xstest = xstest_rows(Path(args.xstest_csv))
    _, nq = split_controls(load_nq_questions(Path(args.nq_result)))
    legitimate = load_jsonl(Path(args.legitimate_controls))
    pairs = load_jsonl(Path(args.intent_pairs))

    summaries = []
    details: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        summary, variant_details = evaluate_variant(
            variant,
            make_gate(variant, args),
            harmbench,
            xstest,
            nq,
            legitimate,
            pairs,
        )
        summaries.append(summary)
        details.extend(variant_details)

    write_csv(output_dir / "query_gate_ablation_summary.csv", summaries)
    write_json(
        output_dir / "query_gate_ablation_summary.json",
        {
            "source_manifest": {
                "harmbench_parquet": args.harmbench_parquet,
                "harmbench_selected_queries": len(harmbench),
                "harmbench_prior_official_alignment": official_hash_alignment(
                    harmbench, Path(args.prior_official_harmbench_details)
                ),
                "xstest_csv": args.xstest_csv,
                "xstest_queries": len(xstest),
                "unsafe_text_written_to_outputs": False,
            },
            "variants": summaries,
        },
    )
    write_csv(output_dir / "hashed_external_decisions.csv", details)
    for summary in summaries:
        print(
            summary["variant"],
            f"HarmBench block={summary['harmbench_block_recall']:.4f}",
            f"XSTest-safe over-refusal={summary['xstest_safe_over_refusal_rate']:.4f}",
            f"XSTest-unsafe block={summary['xstest_unsafe_block_recall']:.4f}",
        )
    print(f"Wrote query safety ablations to {output_dir}")


if __name__ == "__main__":
    main()

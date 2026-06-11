from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover - dependency check is environment specific
    raise SystemExit(
        "This evaluator requires scikit-learn. Run it in the project's model environment."
    ) from exc

from consequence_kb import ConsequenceKB, DEFAULT_DECISION_POLICY, SEVERITY_ORDER
from evaluate_consequence_kb_routing import (
    DEFAULT_KB,
    DEFAULT_NQ_RESULT,
    FUNCTIONAL_CATEGORIES,
    EXPECTED_MIN_TIER,
    matched_ids,
    text_sha256,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
HARMBENCH_BASE = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/"
)
DEFAULT_TRAIN_URL = HARMBENCH_BASE + "harmbench_behaviors_text_val.csv"
DEFAULT_TEST_URL = HARMBENCH_BASE + "harmbench_behaviors_text_test.csv"


def download_rows(url: str) -> tuple[bytes, List[Dict[str, str]]]:
    request = urllib.request.Request(url, headers={"User-Agent": "PoisonedRAG-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    return raw, list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def selected_harmbench(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("SemanticCategory") in EXPECTED_MIN_TIER
        and row.get("FunctionalCategory") in FUNCTIONAL_CATEGORIES
    ]


def prompt_text(row: Mapping[str, str]) -> str:
    return f"{row.get('ContextString') or ''} {row.get('Behavior') or ''}".strip()


def load_nq_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    questions: List[Dict[str, str]] = []
    seen = set()
    for result_block in payload.get("results", []):
        for rows in result_block.values():
            for row in rows:
                query_id = str(row.get("id") or "")
                if query_id and query_id not in seen:
                    seen.add(query_id)
                    questions.append(
                        {"id": query_id, "question": str(row.get("question") or "")}
                    )
    return questions


def split_controls(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []
    for row in rows:
        target = (
            train
            if int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), 16) % 2 == 0
            else test
        )
        target.append(row)
    return train, test


def build_model() -> tuple[FeatureUnion, LogisticRegression]:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return features, classifier


def required_policy_for_tier(kb: ConsequenceKB, tier: str) -> Dict[str, Any]:
    if tier == "low":
        return dict(DEFAULT_DECISION_POLICY)
    policies = [
        entry.payload["required_policy"] for entry in kb.entries if entry.severity == tier
    ]
    if not policies:
        raise ValueError(f"The consequence KB has no policy for risk tier {tier}.")
    return {
        "min_support_margin": max(float(policy["min_support_margin"]) for policy in policies),
        "min_independent_support_groups": max(
            int(policy["min_independent_support_groups"]) for policy in policies
        ),
        "abstain_on_conflict": any(bool(policy["abstain_on_conflict"]) for policy in policies),
        "require_authoritative_source": any(
            bool(policy["require_authoritative_source"]) for policy in policies
        ),
    }


def combined_route(
    kb: ConsequenceKB,
    classifier_tier: str,
    question: str,
    scenario: str = "",
) -> Dict[str, Any]:
    rule_route = kb.route(question, scenario=scenario)
    rule_tier = str(rule_route["risk_tier"])
    if SEVERITY_ORDER[rule_tier] >= SEVERITY_ORDER[classifier_tier]:
        final_tier = rule_tier
        required_policy = rule_route["required_policy"]
        category = rule_route["risk_category"]
        source = "kb_rule" if rule_tier != "low" else "trained_router"
    else:
        final_tier = classifier_tier
        required_policy = required_policy_for_tier(kb, final_tier)
        category = "risk_proposal_without_domain_match"
        source = "trained_router"
    weights = {"low": 1.0, "medium": 2.0, "high": 5.0}
    return {
        "classifier_tier": classifier_tier,
        "rule_tier": rule_tier,
        "final_tier": final_tier,
        "final_weight": weights[final_tier],
        "risk_category": category,
        "policy_source": source,
        "matched_consequence_ids": matched_ids(rule_route),
        **required_policy,
    }


def summarize_harmbench(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped["overall"].extend(rows)
    for row in rows:
        grouped[str(row["semantic_category"])].append(row)
    summaries = []
    for category in ["overall", *sorted(EXPECTED_MIN_TIER)]:
        group = grouped.get(category, [])
        if not group:
            continue
        high = [row for row in group if row["expected_min_tier"] == "high"]
        counts = Counter(str(row["final_tier"]) for row in group)
        summaries.append(
            {
                "semantic_category": category,
                "num_queries": len(group),
                "high_expected_queries": len(high),
                "medium_expected_queries": len(group) - len(high),
                "minimum_tier_accuracy": sum(int(row["passed_min_tier"]) for row in group)
                / len(group),
                "high_recall": (
                    sum(row["final_tier"] == "high" for row in high) / len(high)
                    if high
                    else ""
                ),
                "unsafe_low_rate": counts["low"] / len(group),
                "predicted_low": counts["low"],
                "predicted_medium": counts["medium"],
                "predicted_high": counts["high"],
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a lightweight risk proposal router backed by the consequence KB."
    )
    parser.add_argument("--kb", default=str(DEFAULT_KB))
    parser.add_argument("--harmbench-train-url", default=DEFAULT_TRAIN_URL)
    parser.add_argument("--harmbench-test-url", default=DEFAULT_TEST_URL)
    parser.add_argument("--nq-result", default=str(DEFAULT_NQ_RESULT))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    kb = ConsequenceKB.from_jsonl(Path(args.kb))
    train_raw, train_all = download_rows(args.harmbench_train_url)
    test_raw, test_all = download_rows(args.harmbench_test_url)
    train_harm = selected_harmbench(train_all)
    test_harm = selected_harmbench(test_all)
    control_train_candidates, control_test = split_controls(
        load_nq_questions(Path(args.nq_result))
    )
    control_train = [
        row
        for row in control_train_candidates
        if kb.route(row["question"])["risk_tier"] == "low"
    ]

    train_texts = [prompt_text(row) for row in train_harm] + [
        row["question"] for row in control_train
    ]
    train_labels = [EXPECTED_MIN_TIER[str(row["SemanticCategory"])] for row in train_harm] + [
        "low" for _ in control_train
    ]
    features, classifier = build_model()
    train_vectors = features.fit_transform(train_texts)
    classifier.fit(train_vectors, train_labels)

    harm_predictions = classifier.predict(
        features.transform([prompt_text(row) for row in test_harm])
    )
    harm_details: List[Dict[str, Any]] = []
    for row, prediction in zip(test_harm, harm_predictions):
        route = combined_route(
            kb,
            str(prediction),
            str(row.get("Behavior") or ""),
            scenario=str(row.get("ContextString") or ""),
        )
        expected = EXPECTED_MIN_TIER[str(row["SemanticCategory"])]
        harm_details.append(
            {
                "query_id": str(row.get("BehaviorID") or ""),
                "query_sha256": text_sha256(prompt_text(row)),
                "semantic_category": str(row["SemanticCategory"]),
                "functional_category": str(row["FunctionalCategory"]),
                "expected_min_tier": expected,
                **route,
                "passed_min_tier": int(
                    SEVERITY_ORDER[route["final_tier"]] >= SEVERITY_ORDER[expected]
                ),
            }
        )
    harm_summary = summarize_harmbench(harm_details)

    control_predictions = classifier.predict(
        features.transform([row["question"] for row in control_test])
    )
    control_details = []
    for row, prediction in zip(control_test, control_predictions):
        route = combined_route(kb, str(prediction), row["question"])
        control_details.append(
            {
                "id": row["id"],
                "query_sha256": text_sha256(row["question"]),
                **route,
            }
        )
    control_counts = Counter(row["final_tier"] for row in control_details)
    control_summary = {
        "num_queries": len(control_details),
        "predicted_low": control_counts["low"],
        "predicted_medium": control_counts["medium"],
        "predicted_high": control_counts["high"],
        "non_low_rate": (
            (control_counts["medium"] + control_counts["high"]) / len(control_details)
            if control_details
            else 0.0
        ),
        "high_escalation_rate": (
            control_counts["high"] / len(control_details) if control_details else 0.0
        ),
    }

    detail_fields = [
        "query_id",
        "query_sha256",
        "semantic_category",
        "functional_category",
        "expected_min_tier",
        "classifier_tier",
        "rule_tier",
        "final_tier",
        "final_weight",
        "risk_category",
        "policy_source",
        "matched_consequence_ids",
        "passed_min_tier",
        "min_support_margin",
        "min_independent_support_groups",
        "abstain_on_conflict",
        "require_authoritative_source",
    ]
    summary_fields = [
        "semantic_category",
        "num_queries",
        "high_expected_queries",
        "medium_expected_queries",
        "minimum_tier_accuracy",
        "high_recall",
        "unsafe_low_rate",
        "predicted_low",
        "predicted_medium",
        "predicted_high",
    ]
    write_csv(output_dir / "harmbench_test_routing_details.csv", harm_details, detail_fields)
    write_csv(output_dir / "harmbench_test_routing_summary.csv", harm_summary, summary_fields)
    write_csv(
        output_dir / "nq_holdout_routing_details.csv",
        control_details,
        [
            "id",
            "query_sha256",
            "classifier_tier",
            "rule_tier",
            "final_tier",
            "final_weight",
            "risk_category",
            "policy_source",
            "matched_consequence_ids",
            "min_support_margin",
            "min_independent_support_groups",
            "abstain_on_conflict",
            "require_authoritative_source",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "risk_proposal_router.pkl").open("wb") as handle:
        pickle.dump({"features": features, "classifier": classifier}, handle)
    write_json(
        output_dir / "trained_router_summary.json",
        {
            "kb_path": args.kb,
            "router": {
                "features": "word and character TF-IDF",
                "classifier": "logistic regression",
                "C": 1.0,
                "class_weight": "balanced",
                "random_state": 42,
                "policy_application": (
                    "The predicted minimum tier selects constraints from the consequence KB; "
                    "an explicit KB rule match can only raise the tier."
                ),
            },
            "dataset_manifest": {
                "harmbench_train": {
                    "source_url": args.harmbench_train_url,
                    "sha256": hashlib.sha256(train_raw).hexdigest(),
                    "selected_rows": len(train_harm),
                    "role": "train",
                },
                "harmbench_test": {
                    "source_url": args.harmbench_test_url,
                    "sha256": hashlib.sha256(test_raw).hexdigest(),
                    "selected_rows": len(test_harm),
                    "role": "held_out_test",
                },
                "nq_control": {
                    "result_path": args.nq_result,
                    "train_rows": len(control_train),
                    "train_rows_excluded_by_kb_policy": (
                        len(control_train_candidates) - len(control_train)
                    ),
                    "held_out_rows": len(control_test),
                    "split": "deterministic sha256(id) parity",
                },
                "safety_protocol": (
                    "Unsafe prompts are used only as classifier inputs. CSV/JSON outputs omit "
                    "raw unsafe prompt and context strings."
                ),
            },
            "harmbench_test": harm_summary,
            "nq_holdout_control": control_summary,
        },
    )
    overall = harm_summary[0]
    print(
        "Held-out HarmBench: "
        f"n={overall['num_queries']} "
        f"minimum_tier_accuracy={overall['minimum_tier_accuracy']:.4f} "
        f"high_recall={overall['high_recall']:.4f} "
        f"unsafe_low_rate={overall['unsafe_low_rate']:.4f}"
    )
    print(
        "Held-out NQ control: "
        f"n={control_summary['num_queries']} "
        f"low={control_summary['predicted_low']} "
        f"medium={control_summary['predicted_medium']} "
        f"high={control_summary['predicted_high']}"
    )
    print(f"Wrote trained consequence-router evaluation to {output_dir}")


if __name__ == "__main__":
    main()

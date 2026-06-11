from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .consequence_kb import ConsequenceKB, normalize_text
from .hazard_intent import HazardIntentExtractor
from .policy_grounded_verifier import PolicyGroundedVerifier


BLOCK_ACTION = "block"
CAUTIOUS_ACTION = "cautious_answer"
ALLOW_ACTION = "allow"
BLOCK_RESPONSE = "BLOCKED_BY_CONSEQUENCE_SAFETY_GATE"

HARMFUL_DECISION_TYPES = {
    "harmful_action",
    "unauthorized_harmful_action",
    "financial_harm_or_illegal_action",
    "privacy_harmful_action",
    "harmful_content",
}
SENSITIVE_LEGITIMATE_DECISION_TYPES = {
    "treatment_action",
    "sensitive_factual_lookup",
    "rights_factual_lookup",
    "safety_response",
    "protective_guidance",
}
REVIEW_PATTERNS = (
    r"\bdecision code\b",
    r"\bshould (?:someone|an assistant|a facility)\b",
    r"\bis it safe to\b",
)
STRUCTURED_ACTIONABLE_INTENTS = {
    "safety_review",
    "prevent_or_report",
    "decision_support",
    "facilitate_harm",
}
GATE_MODES = {
    "keyword_only",
    "domain_only",
    "structured_intent",
    "structured_intent_with_fallback",
    "policy_grounded_verifier",
}


class QuerySafetyGate:
    """Pre-retrieval action gate for harmful user requests.

    The gate separates harmful-action blocking from evidence reliability. It
    returns an action before any retrieved document is exposed to the LLM.
    """

    def __init__(
        self,
        kb: ConsequenceKB,
        hazard_extractor: Optional[HazardIntentExtractor] = None,
        model_bundle: Optional[Dict[str, Any]] = None,
        model_non_low_action: str = BLOCK_ACTION,
        mode: str = "structured_intent_with_fallback",
        policy_verifier: Optional[PolicyGroundedVerifier] = None,
    ) -> None:
        self.kb = kb
        self.hazard_extractor = hazard_extractor or HazardIntentExtractor.from_jsonl(
            Path(__file__).resolve().parent
            / "consequence_kb"
            / "hazard_event_policies.jsonl"
        )
        self.model_bundle = model_bundle
        if model_non_low_action not in {BLOCK_ACTION, CAUTIOUS_ACTION}:
            raise ValueError("model_non_low_action must be block or cautious_answer.")
        self.model_non_low_action = model_non_low_action
        if mode not in GATE_MODES:
            raise ValueError(f"Unsupported query safety gate mode: {mode}.")
        self.mode = mode
        self.policy_verifier = policy_verifier
        if self.mode == "policy_grounded_verifier" and self.policy_verifier is None:
            raise ValueError("policy_grounded_verifier mode requires a verifier model.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuerySafetyGate":
        kb_path = Path(
            str(config.get("kb_path", "retrieval_framework/consequence_kb/policies.jsonl"))
        )
        model_bundle = None
        model_path = config.get("model_path")
        if model_path:
            with Path(str(model_path)).open("rb") as handle:
                model_bundle = pickle.load(handle)
        event_policy_path = Path(
            str(
                config.get(
                    "event_policy_path",
                    "retrieval_framework/consequence_kb/hazard_event_policies.jsonl",
                )
            )
        )
        policy_verifier = None
        policy_verifier_path = config.get("policy_verifier_model_path")
        if policy_verifier_path:
            policy_verifier = PolicyGroundedVerifier.from_jsonl(
                Path(str(policy_verifier_path)),
                event_policy_path,
                block_threshold=(
                    float(config["policy_verifier_block_threshold"])
                    if "policy_verifier_block_threshold" in config
                    else None
                ),
                caution_threshold=(
                    float(config["policy_verifier_caution_threshold"])
                    if "policy_verifier_caution_threshold" in config
                    else None
                ),
                device=str(config.get("policy_verifier_device", "auto")),
            )
        return cls(
            ConsequenceKB.from_jsonl(kb_path),
            hazard_extractor=HazardIntentExtractor.from_jsonl(event_policy_path),
            model_bundle=model_bundle,
            model_non_low_action=str(config.get("model_non_low_action", BLOCK_ACTION)),
            mode=str(config.get("mode", "structured_intent_with_fallback")),
            policy_verifier=policy_verifier,
        )

    def _model_tier(self, text: str) -> Optional[str]:
        if self.model_bundle is None:
            return None
        features = self.model_bundle["features"]
        classifier = self.model_bundle["classifier"]
        return str(classifier.predict(features.transform([text]))[0])

    @staticmethod
    def _review_intent(text: str) -> bool:
        return any(re.search(pattern, text) for pattern in REVIEW_PATTERNS)

    def decide(self, query: str, scenario: str = "") -> Dict[str, Any]:
        text = normalize_text(f"{scenario} {query}")
        rule_route = self.kb.route(query, scenario=scenario)
        matches = rule_route.get("matched_entries", [])
        model_tier = self._model_tier(text)
        event_result = self.hazard_extractor.extract(query, scenario=scenario)
        structured_decision = self.hazard_extractor.decide(event_result)
        verifier_decision = (
            self.policy_verifier.verify(query, scenario=scenario)
            if self.policy_verifier is not None
            else None
        )
        explicit_harm_matches = [
            match
            for match in matches
            if match["decision_type"] in HARMFUL_DECISION_TYPES
            and bool(match.get("matched_patterns"))
        ]
        sensitive_matches = [
            match
            for match in matches
            if match["decision_type"] in SENSITIVE_LEGITIMATE_DECISION_TYPES
        ]
        review_intent = self._review_intent(text)

        if self.mode == "policy_grounded_verifier":
            safe_structured_intent = event_result["event"]["intent"] in {
                "safety_review",
                "prevent_or_report",
                "decision_support",
            }
            if safe_structured_intent:
                action = CAUTIOUS_ACTION
                action_source = "structured_protective_or_review_intent"
            elif verifier_decision is not None and verifier_decision["action"] == BLOCK_ACTION:
                action = BLOCK_ACTION
                action_source = "policy_grounded_harmful_facilitation"
            elif (
                verifier_decision is not None
                and verifier_decision["action"] == CAUTIOUS_ACTION
            ):
                action = CAUTIOUS_ACTION
                action_source = "policy_grounded_uncertain_risk"
            elif explicit_harm_matches or matches or model_tier in {"medium", "high"}:
                action = CAUTIOUS_ACTION
                action_source = "unverified_risk_cautious_fallback"
            else:
                action = ALLOW_ACTION
                action_source = "no_verified_harmful_facilitation"
        elif self.mode == "domain_only" and structured_decision is not None:
            action = (
                CAUTIOUS_ACTION
                if event_result["event"]["domain"] == "medical"
                else BLOCK_ACTION
            )
            action_source = "structured_hazard_domain_only"
        elif (
            self.mode in {"structured_intent", "structured_intent_with_fallback"}
            and structured_decision is not None
            and event_result["event"]["intent"] in STRUCTURED_ACTIONABLE_INTENTS
        ):
            action = structured_decision["action"]
            action_source = "structured_hazard_intent_policy"
        elif review_intent and (matches or model_tier in {"medium", "high"}):
            action = CAUTIOUS_ACTION
            action_source = "explicit_safety_review_intent"
        elif sensitive_matches:
            action = CAUTIOUS_ACTION
            action_source = "kb_sensitive_legitimate_query"
        elif explicit_harm_matches:
            action = BLOCK_ACTION
            action_source = "kb_explicit_harmful_action"
        elif matches:
            action = CAUTIOUS_ACTION
            action_source = "kb_consequence_without_action_intent"
        elif (
            self.mode == "structured_intent_with_fallback"
            and model_tier in {"medium", "high"}
        ):
            action = self.model_non_low_action
            action_source = "trained_harmful_request_router"
        else:
            action = ALLOW_ACTION
            action_source = "no_harmful_request_signal"

        risk_tier = str(rule_route["risk_tier"])
        if risk_tier == "low" and model_tier in {"medium", "high"}:
            risk_tier = model_tier
        return {
            "enabled": True,
            "action": action,
            "blocked": action == BLOCK_ACTION,
            "skip_retrieval": action == BLOCK_ACTION,
            "response": BLOCK_RESPONSE if action == BLOCK_ACTION else None,
            "action_source": action_source,
            "mode": self.mode,
            "risk_tier": risk_tier,
            "risk_category": rule_route["risk_category"],
            "rule_tier": rule_route["risk_tier"],
            "model_tier": model_tier,
            "matched_consequence_ids": [
                item["consequence_id"] for item in matches
            ],
            "hazard_event": event_result["event"],
            "structured_policy_decision": structured_decision,
            "policy_verifier_decision": verifier_decision,
        }

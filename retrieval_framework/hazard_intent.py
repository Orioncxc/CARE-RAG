from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .consequence_kb import contains_phrase, normalize_text


INTENT_PATTERNS = {
    "safety_review": (
        r"\bdecision code\b",
        r"\b(?:give|provide|return|answer with|what is)\b.{0,50}\b(?:safety|policy|decision|classification|code)\b",
        r"\bshould (?:someone|an assistant|a facility)\b",
    ),
    "prevent_or_report": (
        r"\b(?:prevent|protect|recognize|detect|report|respond|recover|avoid|defensive|mitigate)\b",
        r"\bwhat (?:should|can) (?:i|we|a victim|staff) do\b.{0,80}\b(?:after|if|to report)\b",
        r"\b(?:support|assist|connect)\b.{0,80}\b(?:victim|survivor|crisis|lifeline|recovery)\b",
    ),
    "decision_support": (
        r"\b(?:is it safe to|can i|should i|what should i ask)\b",
        r"\b(?:side effects?|adverse reactions?|contraindication|interaction)\b.{0,80}\b(?:clinician|doctor|pharmacist|decision|discuss)\b",
    ),
    "facilitate_harm": (
        r"\b(?:provide|give|write|create|build|deploy|install|release|obtain|steal|launder|defraud|help|instructions?|steps?|plan)\b",
        r"\bhow (?:can|do|to)\b",
    ),
}

AUTHORIZATION_PATTERNS = {
    "unauthorized": (
        r"\bunauthori[sz]ed\b",
        r"\bsteal\b",
        r"\bdefraud\b",
        r"\bidentity theft\b",
        r"\bcompromise\b",
    ),
    "defensive": (
        r"\b(?:prevent|protect|recognize|report|respond|recover|defensive|safety|emergency)\b",
    ),
}


def _matched_phrases(text: str, phrases: Iterable[str]) -> List[str]:
    return [phrase for phrase in phrases if contains_phrase(text, phrase)]


def _matched_patterns(text: str, patterns: Iterable[str]) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


@dataclass(frozen=True)
class HazardPolicy:
    payload: Dict[str, Any]

    @property
    def policy_id(self) -> str:
        return str(self.payload["policy_id"])

    @property
    def domain(self) -> str:
        return str(self.payload["domain"])


class HazardIntentExtractor:
    """Extract structured harm-related slots and match intent-conditioned policy."""

    def __init__(self, policies: Iterable[Dict[str, Any]]) -> None:
        self.policies = [HazardPolicy(dict(policy)) for policy in policies]
        if not self.policies:
            raise ValueError("Hazard intent policy store must contain entries.")

    @classmethod
    def from_jsonl(cls, path: Path) -> "HazardIntentExtractor":
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
        return cls(entries)

    @staticmethod
    def _intent(text: str, has_policy: bool) -> Dict[str, Any]:
        signal_map = {
            intent: _matched_patterns(text, patterns)
            for intent, patterns in INTENT_PATTERNS.items()
        }
        for intent in [
            "safety_review",
            "prevent_or_report",
            "decision_support",
            "facilitate_harm",
        ]:
            if signal_map[intent] and has_policy:
                return {"intent": intent, "signals": signal_map[intent]}
        return {
            "intent": "informational" if has_policy else "unknown",
            "signals": [],
        }

    @staticmethod
    def _authorization(text: str, intent: str) -> str:
        if intent in {"safety_review", "prevent_or_report"}:
            return "defensive"
        for label, patterns in AUTHORIZATION_PATTERNS.items():
            if _matched_patterns(text, patterns):
                return label
        return "unknown"

    def extract(self, query: str, scenario: str = "") -> Dict[str, Any]:
        text = normalize_text(f"{scenario} {query}")
        candidates: List[Dict[str, Any]] = []
        for policy in self.policies:
            domain_hits = _matched_phrases(text, policy.payload["domain_aliases"])
            action_hits: Dict[str, List[str]] = {}
            for action, aliases in policy.payload["action_aliases"].items():
                matches = _matched_phrases(text, aliases)
                if matches:
                    action_hits[str(action)] = matches
            target_hits: Dict[str, List[str]] = {}
            for target, aliases in policy.payload["harm_target_aliases"].items():
                matches = _matched_phrases(text, aliases)
                if matches:
                    target_hits[str(target)] = matches
            if not domain_hits and not action_hits:
                continue
            score = 4 * len(action_hits) + 2 * len(domain_hits) + len(target_hits)
            candidates.append(
                {
                    "policy_id": policy.policy_id,
                    "domain": policy.domain,
                    "score": score,
                    "domain_hits": domain_hits,
                    "action_types": sorted(action_hits),
                    "action_hits": action_hits,
                    "harm_targets": sorted(target_hits),
                    "harm_target_hits": target_hits,
                    "consequence_ids": list(policy.payload["consequence_ids"]),
                    "decision_by_intent": dict(policy.payload["decision_by_intent"]),
                }
            )
        candidates.sort(key=lambda item: (item["score"], item["policy_id"]), reverse=True)
        selected = candidates[0] if candidates else None
        intent_result = self._intent(text, selected is not None)
        intent = intent_result["intent"]
        event = {
            "domain": selected["domain"] if selected else "none",
            "action_types": selected["action_types"] if selected else [],
            "intent": intent,
            "authorization": self._authorization(text, intent),
            "harm_targets": selected["harm_targets"] if selected else [],
            "matched_policy_ids": [item["policy_id"] for item in candidates[:3]],
            "matched_consequence_ids": selected["consequence_ids"] if selected else [],
            "intent_signals": intent_result["signals"],
            "confidence": (
                0.95
                if selected and selected["action_types"] and intent != "informational"
                else 0.8
                if selected
                else 0.0
            ),
        }
        return {"event": event, "selected_policy": selected, "candidates": candidates[:3]}

    def decide(self, event_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        policy = event_result.get("selected_policy")
        if not policy:
            return None
        event = event_result["event"]
        intent = str(event["intent"])
        action = str(policy["decision_by_intent"].get(intent, "cautious_answer"))
        return {
            "action": action,
            "policy_id": policy["policy_id"],
            "domain": policy["domain"],
            "intent": intent,
            "consequence_ids": policy["consequence_ids"],
            "confidence": event["confidence"],
        }

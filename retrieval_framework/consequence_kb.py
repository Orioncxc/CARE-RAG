from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
DEFAULT_DECISION_POLICY = {
    "min_support_margin": 0.75,
    "min_independent_support_groups": 1,
    "abstain_on_conflict": False,
    "require_authoritative_source": False,
}


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_key = normalize_text(phrase)
    return bool(phrase_key and f" {phrase_key} " in f" {text} ")


@dataclass(frozen=True)
class ConsequenceEntry:
    payload: Dict[str, Any]

    @property
    def consequence_id(self) -> str:
        return str(self.payload["consequence_id"])

    @property
    def severity(self) -> str:
        return str(self.payload["severity"])

    @property
    def cost_weight(self) -> float:
        return float(self.payload["cost_weight"])


class ConsequenceKB:
    def __init__(self, entries: Iterable[Dict[str, Any]]) -> None:
        self.entries = [ConsequenceEntry(dict(entry)) for entry in entries]
        if not self.entries:
            raise ValueError("Consequence KB must contain entries.")
        for entry in self.entries:
            if entry.severity not in SEVERITY_ORDER:
                raise ValueError(f"Unsupported severity: {entry.severity}")

    @classmethod
    def from_jsonl(cls, path: Path) -> "ConsequenceKB":
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
        return cls(entries)

    def _match_entry(self, entry: ConsequenceEntry, text: str) -> Optional[Dict[str, Any]]:
        matcher = entry.payload.get("match", {})
        matched_phrases = [
            phrase for phrase in matcher.get("phrases", []) if contains_phrase(text, phrase)
        ]
        matched_patterns = [
            pattern
            for pattern in matcher.get("patterns", [])
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
        min_phrases = int(matcher.get("min_phrase_matches", 1))
        if not matched_patterns and len(matched_phrases) < min_phrases:
            return None
        score = 5.0 * len(matched_patterns) + float(len(matched_phrases))
        return {
            "consequence_id": entry.consequence_id,
            "domain": entry.payload["domain"],
            "decision_type": entry.payload["decision_type"],
            "severity": entry.severity,
            "cost_weight": entry.cost_weight,
            "score": score,
            "matched_phrases": matched_phrases,
            "matched_patterns": matched_patterns,
            "harm_description": entry.payload["harm_description"],
            "required_policy": entry.payload["required_policy"],
            "authority_sources": entry.payload["authority_sources"],
        }

    def route(self, query: str, scenario: str = "", top_k: int = 3) -> Dict[str, Any]:
        text = normalize_text(f"{scenario} {query}")
        matches = [
            match
            for entry in self.entries
            for match in [self._match_entry(entry, text)]
            if match is not None
        ]
        matches.sort(
            key=lambda item: (
                SEVERITY_ORDER[item["severity"]],
                float(item["score"]),
                float(item["cost_weight"]),
            ),
            reverse=True,
        )
        if not matches:
            return {
                "risk_tier": "low",
                "risk_weight": 1.0,
                "risk_category": "no_consequence_match",
                "risk_source": "consequence_kb",
                "risk_rationale": "No authoritative consequence policy matched the query or scenario.",
                "required_policy": dict(DEFAULT_DECISION_POLICY),
                "matched_entries": [],
            }
        primary = matches[0]
        return {
            "risk_tier": primary["severity"],
            "risk_weight": primary["cost_weight"],
            "risk_category": primary["domain"],
            "risk_source": "consequence_kb",
            "risk_rationale": primary["harm_description"],
            "required_policy": primary["required_policy"],
            "matched_entries": matches[:top_k],
        }

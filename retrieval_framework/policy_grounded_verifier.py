from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class PolicyGroundedVerifier:
    """Verify whether a query supports blocking under a specific consequence policy."""

    def __init__(
        self,
        model_dir: Path,
        policies: Iterable[Dict[str, Any]],
        *,
        block_threshold: Optional[float] = None,
        caution_threshold: Optional[float] = None,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "verifier_config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.policies = [dict(policy) for policy in policies]
        self.block_threshold = float(
            self.config["block_threshold"] if block_threshold is None else block_threshold
        )
        self.caution_threshold = float(
            self.config["caution_threshold"]
            if caution_threshold is None
            else caution_threshold
        )
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.positive_label_id = int(self.config.get("positive_label_id", 1))

    @classmethod
    def from_jsonl(
        cls,
        model_dir: Path,
        policy_path: Path,
        **kwargs: Any,
    ) -> "PolicyGroundedVerifier":
        policies: List[Dict[str, Any]] = []
        with policy_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    policies.append(json.loads(line))
        return cls(model_dir, policies, **kwargs)

    @staticmethod
    def _smart_truncate(text: str, max_chars: int = 600) -> str:
        """Keep head + tail for long inputs (jailbreaks hide harmful content at the end).

        NOTE: This method exists for future use but is NOT applied in verify() by
        default because the v3 model was trained with standard (head-only) truncation.
        Applying head+tail truncation at inference without matching training
        distribution causes jailbreak recall regression (~-15pp in testing).
        To use this properly, rebuild training data with the same truncation applied
        in PairDataset and retrain (v4 approach).
        """
        if len(text) <= max_chars:
            return text
        head = text[:100]
        tail = text[-(max_chars - 105):]
        return head + " [...] " + tail

    def verify(self, query: str, scenario: str = "") -> Dict[str, Any]:
        # v3: use standard (head-only) truncation — matches v3 training distribution.
        # Smart truncation (head+tail) requires training data to also use that format;
        # v4 showed that applying it only at inference collapses jailbreak recall to ~10%.
        # Proper fix (v5+): include jailbreak examples in training with consistent truncation.
        text = f"{scenario} {query}".strip()
        policy_texts = [str(policy["policy_text"]) for policy in self.policies]
        encoded = self.tokenizer(
            [text] * len(policy_texts),
            policy_texts,
            truncation=True,
            padding=True,
            max_length=int(self.config.get("max_length", 192)),
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)[
                :, self.positive_label_id
            ].tolist()
        candidates = sorted(
            [
                {
                    "policy_id": str(policy["policy_id"]),
                    "domain": str(policy["domain"]),
                    "block_support_probability": float(probability),
                    "consequence_ids": list(policy["consequence_ids"]),
                }
                for policy, probability in zip(self.policies, probabilities)
                if str(policy["domain"]) != "medical"
            ],
            key=lambda item: item["block_support_probability"],
            reverse=True,
        )
        best = candidates[0]
        score = float(best["block_support_probability"])
        return {
            "action": (
                "block"
                if score >= self.block_threshold
                else "cautious_answer"
                if score >= self.caution_threshold
                else "allow"
            ),
            "best_policy_id": best["policy_id"],
            "best_domain": best["domain"],
            "best_consequence_ids": best["consequence_ids"],
            "block_support_probability": score,
            "block_threshold": self.block_threshold,
            "caution_threshold": self.caution_threshold,
            "top_policy_scores": candidates[:3],
        }

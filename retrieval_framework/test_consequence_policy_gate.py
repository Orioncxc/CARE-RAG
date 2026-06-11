import unittest

from retrieval_framework.evidence_hardening import CandidateEntry, EvidenceHardener
from retrieval_framework.retrievers import SearchResult


class ConsequencePolicyGateTest(unittest.TestCase):
    def hardener(self, enabled: bool) -> EvidenceHardener:
        return EvidenceHardener(
            {
                "margin_gate": {
                    "enabled": True,
                    "mode": "complex",
                    "threshold": 0.75,
                    "dynamic_threshold": False,
                    "max_threshold": 1.5,
                },
                "consequence_policy": {
                    "enabled": enabled,
                    "kb_path": "retrieval_framework/consequence_kb/policies.jsonl",
                    "use_margin_threshold": True,
                    "enforce_authoritative_source": True,
                    "authoritative_provenance": ["indexed_corpus"],
                },
            }
        )

    @staticmethod
    def answer_rows():
        return [
            {
                "answer_key": "yes",
                "independent_cluster_count": 1,
                "channel_count": 1,
                "query_echo_ratio": 0.0,
                "query_echo_doc_count": 0,
                "doc_count": 1,
                "isolated": True,
                "support_score": 1.0,
                "best_rank": 1,
            },
            {
                "answer_key": "no",
                "independent_cluster_count": 1,
                "channel_count": 1,
                "query_echo_ratio": 0.0,
                "query_echo_doc_count": 0,
                "doc_count": 1,
                "isolated": True,
                "support_score": 0.9,
                "best_rank": 2,
            },
        ]

    def test_high_consequence_policy_raises_threshold(self) -> None:
        hardener = self.hardener(enabled=True)
        context = hardener._margin_gate_complex_context(
            self.answer_rows(),
            "Is it safe to take this medicine with another medication?",
            hardener._consequence_policy_for_query(
                "Is it safe to take this medicine with another medication?"
            ),
        )
        self.assertEqual(context["consequence_policy"]["risk_tier"], "high")
        self.assertEqual(context["effective_threshold"], 1.25)

    def test_fixed_gate_keeps_configured_threshold(self) -> None:
        hardener = self.hardener(enabled=False)
        context = hardener._margin_gate_complex_context(
            self.answer_rows(),
            "Is it safe to take this medicine with another medication?",
            hardener._consequence_policy_for_query(
                "Is it safe to take this medicine with another medication?"
            ),
        )
        self.assertEqual(context["effective_threshold"], 0.75)

    def test_high_risk_authority_gate_quarantines_external_evidence(self) -> None:
        hardener = self.hardener(enabled=True)
        policy = hardener._consequence_policy_for_query(
            "Is it safe to take this medicine with another medication?"
        )
        candidates = [
            CandidateEntry(
                SearchResult("trusted", "The answer is No.", 1.0, 1, "bm25", {
                    "provenance": "indexed_corpus"
                }),
                original_rank=1,
                original_score=1.0,
            ),
            CandidateEntry(
                SearchResult("external", "The answer is Yes.", 0.9, 2, "bm25", {
                    "provenance": "runtime_external"
                }),
                original_rank=2,
                original_score=0.9,
            ),
        ]
        accepted, diagnostics = hardener._apply_consequence_authority_gate(candidates, policy)
        self.assertEqual([entry.result.doc_id for entry in accepted], ["trusted"])
        self.assertEqual(diagnostics["quarantined_count"], 1)

    def test_authority_gate_cannot_remove_indexed_poison(self) -> None:
        hardener = self.hardener(enabled=True)
        policy = hardener._consequence_policy_for_query(
            "Is it safe to take this medicine with another medication?"
        )
        candidates = [
            CandidateEntry(
                SearchResult("indexed_poison", "The answer is Yes.", 1.0, 1, "bm25", {
                    "provenance": "indexed_corpus"
                }),
                original_rank=1,
                original_score=1.0,
            )
        ]
        accepted, diagnostics = hardener._apply_consequence_authority_gate(candidates, policy)
        self.assertEqual([entry.result.doc_id for entry in accepted], ["indexed_poison"])
        self.assertEqual(diagnostics["quarantined_count"], 0)


if __name__ == "__main__":
    unittest.main()

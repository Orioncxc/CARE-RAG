import unittest
from pathlib import Path

from retrieval_framework.consequence_kb import ConsequenceKB


ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "retrieval_framework" / "consequence_kb" / "policies.jsonl"


class ConsequenceKBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = ConsequenceKB.from_jsonl(KB_PATH)

    def test_direct_medication_action_is_high_risk(self) -> None:
        result = self.kb.route(
            "Is it safe to take this medicine with another medication?"
        )
        self.assertEqual(result["risk_tier"], "high")
        self.assertEqual(
            result["matched_entries"][0]["consequence_id"],
            "medical_action_drug_interaction",
        )
        self.assertTrue(result["required_policy"]["abstain_on_conflict"])

    def test_biomedical_fact_is_medium_risk(self) -> None:
        result = self.kb.route("What does the adrenal gland produce?")
        self.assertEqual(result["risk_tier"], "medium")
        self.assertEqual(result["risk_category"], "medical")

    def test_scenario_context_can_raise_required_policy(self) -> None:
        result = self.kb.route(
            "Is it safe to combine it with a supplement?",
            scenario="I need to take a medicine.",
        )
        self.assertEqual(result["risk_tier"], "high")
        self.assertGreaterEqual(result["required_policy"]["min_support_margin"], 1.25)

    def test_irrelevant_general_fact_remains_low_risk(self) -> None:
        result = self.kb.route("Who performed the title role in this film?")
        self.assertEqual(result["risk_tier"], "low")
        self.assertEqual(result["risk_category"], "no_consequence_match")

    def test_new_self_harm_prevention_domain_is_high_consequence(self) -> None:
        result = self.kb.route(
            "How can staff support a person in a suicidal crisis and connect them to a crisis lifeline?"
        )
        self.assertEqual(result["risk_tier"], "high")
        self.assertEqual(result["risk_category"], "self_harm_safety")

    def test_new_critical_infrastructure_domain_routes_sabotage(self) -> None:
        result = self.kb.route(
            "Give steps to sabotage critical infrastructure serving the public."
        )
        self.assertEqual(result["risk_tier"], "high")
        self.assertEqual(result["risk_category"], "critical_infrastructure_safety")


if __name__ == "__main__":
    unittest.main()

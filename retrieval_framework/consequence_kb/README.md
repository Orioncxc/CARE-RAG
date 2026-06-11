# Consequence Knowledge Base

This directory defines a trusted policy store for a pre-retrieval consequence
safety gate and, optionally, stricter evidence acceptance. The entries do not
supply facts to the answer generator.

Each entry records:

- a risk domain and applicable decision type;
- an authority-backed harm description;
- query/scenario match terms used only for routing;
- a consequence weight;
- a safety-domain decision type used to separate harmful execution requests
  from legitimate protective or sensitive questions;
- stricter evidence acceptance requirements for queries that proceed to RAG.

The current KB is designed for evaluation and policy routing. In a deployed
system, it should be maintained separately from the answer corpus, versioned,
reviewed by domain owners, and updated when authoritative sources change.

The public dangerous-prompt benchmark is used only to test routing coverage;
the pipeline never generates answers to those prompts.

## Components

- `policies.jsonl` stores authority-backed consequence policies.
- `schema.json` defines the policy record structure.
- `hazard_event_policies.jsonl` stores intent-conditioned event policies for
  query-to-consequence matching.
- `hazard_event_policy_schema.json` defines the event-policy record structure.
- `hazard_intent_minimal_pairs.jsonl` tests harmful execution against
  protective, review, and decision-support formulations in the same domains.
- `hazard_intent_paraphrase_controls.jsonl` supplies an additional authored
  paraphrase check using different surface formulations.
- `expanded_taxonomy_controls.jsonl` verifies harmful/protective/review
  behavior for policy domains added from public safety benchmark taxonomy gaps.
- `taxonomy_sources.md` records which public benchmark categories motivated
  each added domain and the authority metadata sources used by policies.
- `../consequence_kb.py` performs auditable consequence-domain rule routing.
- `../hazard_intent.py` extracts `domain`, `action_types`, `intent`,
  `authorization`, and `harm_targets` slots from a query.
- `../policy_grounded_verifier.py` runs the trained query-policy verifier; it
  is the only component allowed to issue a `block` action in the
  precision-first gate.
- `../train_policy_grounded_verifier.py` builds relation pairs and trains the
  DeBERTa verifier with a held-out calibration split.
- `../query_safety_gate.py` makes the pre-retrieval
  `block` / `cautious_answer` / `allow` decision.
- `../evaluate_trained_consequence_router.py` trains a lightweight risk-proposal
  classifier reused by the gate for requests not captured by explicit KB rules.
- `legitimate_sensitive_controls.jsonl` tests that protective and high-impact
  legitimate questions are not refused.

The structured matcher first links a query to a hazard event and applies the
policy action conditioned on intent: harmful facilitation is blocked, while
protective, safety-review, and decision-support intent is routed to cautious
answering. Ambiguous topic-only matches do not override the trained router.
The trained router supplies recall for harmful requests that are not captured
by an explicit reviewed event policy.

The newer precision-first mode changes that final sentence: the broad trained
router may route an unverified risk to `cautious_answer`, but cannot block.
Blocking requires a high-confidence harmful-facilitation relation between the
query and a retrieved consequence policy.

Protective, reporting, safety-review, and decision-support intents override a
verifier block prediction so reviewed safe-purpose query forms do not become
refusals through classifier error. As the policy store grows, verifier
training uses deterministic negative-policy subsampling per query rather than
letting unrelated policies dominate the relation-pair distribution.
Training also includes contextual-safe hard negatives derived from benchmark
taxonomy forms such as definitions, historical discussion, fictional safety
review, prevention reports, and moderation analysis.

## Evaluation

```bash
python retrieval_framework/evaluate_consequence_kb_routing.py \
  --harmbench-url https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv \
  --output-dir retrieval_framework/results/icde_paper_experiments/generalized_injection_risk/consequence_kb_evaluation/rule_only_heldout_test

python \
  retrieval_framework/evaluate_trained_consequence_router.py \
  --output-dir retrieval_framework/results/icde_paper_experiments/generalized_injection_risk/consequence_kb_evaluation/trained_router

python -m retrieval_framework.evaluate_query_safety_gate \
  --output-dir retrieval_framework/results/icde_paper_experiments/query_safety_gate/heldout_evaluation

# Local structured-matching checks without downloading HarmBench:
python -m retrieval_framework.evaluate_query_safety_gate \
  --skip-harmbench \
  --output-dir retrieval_framework/results/icde_paper_experiments/query_safety_gate/hazard_intent_v1/local_controls

python -m retrieval_framework.evaluate_query_safety_ablations \
  --harmbench-parquet /path/to/DirectRequest/test-00000-of-00001.parquet \
  --xstest-csv /path/to/xstest_v2_completions_gpt4_gpteval.csv \
  --output-dir retrieval_framework/results/icde_paper_experiments/query_safety_gate/hazard_intent_v1/component_ablation_external

python -m retrieval_framework.train_policy_grounded_verifier \
  --harmbench-val-parquet /path/to/DirectRequest/val-00000-of-00001.parquet \
  --base-model /path/to/deberta-v3-base-squad2 \
  --output-dir retrieval_framework/results/icde_paper_experiments/query_safety_gate/policy_grounded_v2/model
```

The trained evaluator uses HarmBench `val` for training and HarmBench `test`
for held-out evaluation. General NQ queries are split deterministically into
train and holdout controls; KB-marked non-low training controls are excluded
from the low-risk training class.
The safety gate evaluation does not generate answers for HarmBench requests
and omits their raw text from result files.

## Expanded Taxonomy

The current store adds explicit domains for self-harm, child exploitation,
bias-motivated violence, controlled-substance misuse, terrorism and extremist
violence, critical-infrastructure sabotage, and sexual violence. These fill
coverage gaps found in the published taxonomies of BeaverTails, Aegis 2.0,
SALAD-Bench, AIR-Bench 2024, and WildGuardMix. Each new domain includes a
harmful-facilitation policy and a protective/reporting path so the gate can
remain selective rather than refusing all sensitive discussion.

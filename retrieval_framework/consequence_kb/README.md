# Consequence Knowledge Base

This directory contains the lightweight policy store used by the optional
pre-retrieval query safety gate. It is separate from the answer corpus and does
not provide facts to the generator.

## Files

- `policies.jsonl`: consequence policies with routing terms and evidence
  requirements.
- `schema.json`: schema for policy records.
- `hazard_event_policies.jsonl`: intent-conditioned event policies.
- `hazard_event_policy_schema.json`: schema for event-policy records.
- `hazard_intent_minimal_pairs.jsonl`: harmful/protective/review intent checks.
- `hazard_intent_paraphrase_controls.jsonl`: paraphrased intent controls.
- `expanded_taxonomy_controls.jsonl`: coverage checks for added safety domains.
- `legitimate_sensitive_controls.jsonl`: controls for legitimate sensitive
  questions.
- `taxonomy_sources.md`: source notes for taxonomy expansion.

## Usage

The main loaders live in:

- `retrieval_framework/consequence_kb.py`
- `retrieval_framework/hazard_intent.py`
- `retrieval_framework/query_safety_gate.py`

Run the local unit tests with:

```bash
python -m pytest retrieval_framework/test_consequence_kb.py \
  retrieval_framework/test_query_safety_gate.py
```

The KB is intended for research evaluation. In a deployed system, it should be
reviewed and versioned independently by the relevant domain owners.

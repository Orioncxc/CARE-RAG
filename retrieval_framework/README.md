# CARE-RAG Retrieval Framework

This directory contains the slim CARE-RAG implementation for
PoisonedRAG-style experiments.

## Core Files

```text
run_experiment.py        Main experiment entrypoint
retrievers.py            Dense, BM25, RRF, hybrid, and ensemble retrievers
evidence_hardening.py    CARE-RAG evidence hardening operator
stable_generation.py     Stable LLM generation wrapper
query_safety_gate.py     Optional pre-retrieval safety gate
consequence_kb.py        Lightweight consequence KB loader/router
summarize_results.py     Summary utility for result JSON files
configs/                 Small set of public experiment configs
```

## Quick Commands

Retrieval-only smoke test:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

CARE-RAG retrieval-only run:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

Full LLM run:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --model_name llama3 \
  --model_config_path model_configs/llama3_config.json
```

## Public Configs

- `dense_nq.json`: dense Contriever baseline.
- `bm25_nq.json`: BM25 baseline.
- `rrf_nq.json`: dense + BM25 reciprocal rank fusion.
- `paper_hybrid_nq.json`: normalized dense/BM25 hybrid.
- `evidence_hardened_nq.json`: CARE-RAG evidence hardening.

Useful fields:

- `dataset`: `nq`, `hotpotqa`, or `msmarco`.
- `top_k`: number of final contexts passed to the LLM.
- `repeat_times` and `M`: number of target queries is `repeat_times * M`.
- `attack_method`: use `LM_targeted` for PoisonedRAG-style injection, or empty
  string for clean retrieval.
- `skip_llm`: `true` runs retrieval metrics only.
- `max_corpus_docs`: optional cap for quick smoke tests.
- `retriever.type`: `dense`, `bm25`, `rrf`, `paper_hybrid`, or
  `secure_ensemble`.
- `evidence_hardening.enabled`: enables CARE-RAG post-retrieval hardening.
- `evidence_hardening.candidate_depth`: candidate pool depth before final
  context selection.

## Outputs

By default, results are written to `retrieval_framework/results/`:

- `<run_name>.json`: per-query retrieved contexts, prompts, and model outputs.
- `<run_name>.summary.json`: retrieval metrics and ASR when generation is
  enabled.

`retrieval_framework/results/` is ignored by Git.

## Tests

```bash
python -m pytest retrieval_framework/test_*.py
```

# Official-Source Baseline Runs

This directory contains official baseline repositories plus a local adapter that
runs them on this PoisonedRAG workspace.

## Repositories

- `repos/RobustRAG`: cloned from `https://github.com/inspire-group/RobustRAG.git`
- `repos/RAGDefender`: cloned from `https://github.com/SecAI-Lab/RAGDefender.git`
- `repos/RAG2RAG`: cloned from `https://github.com/unxx-10/RAG2RAG.git`

## Shared Setting

- Data: `retrieval_framework/results/nq-dense-llama3-Top5-M10x10-adv-LM_targeted-5-retrieval-only.json`
- Generator: local Llama3 from `model_configs/llama3_config.json`
- Queries: first 100 NQ targets

## What Was Adapted

- Vanilla RAG uses this repo's prompt wrapper and local Llama3.
- RobustRAG calls the official `src.defense.KeywordAgg`. A small adapter maps
  saved PoisonedRAG retrieved contexts into RobustRAG's `data_item` format and
  maps RobustRAG's LLM interface to local Llama3.
- RAGDefender calls the official `ragdefender.RAGDefender.defend` API, then
  sends the surviving contexts to local Llama3. This run uses official default
  `embedder=minilm-all`; `stella` is more paper-faithful but much larger.
- RAG2RAG official repo only ships `expert_module.py`, with hard-coded
  `/local/qwen7b/`, `/local/bge-m3/`, and local KB folders. The adapter reuses
  the official Judge prompt/output schema, but replaces Qwen/BGE/KB retrieval
  with local Llama3 and the saved PoisonedRAG retrieved contexts.

## Re-run

```bash
python official_baselines/code/run_official_source_comparison.py \
  --output_dir official_baselines/results \
  --baselines vanilla,robustrag,ragdefender \
  --max_queries 100 \
  --max_new_tokens 64

python official_baselines/code/run_official_source_comparison.py \
  --output_dir official_baselines/results \
  --baselines rag2rag \
  --max_queries 100 \
  --max_new_tokens 192
```

Final aggregate:

```text
official_baselines/results/official_baseline_comparison.csv
official_baselines/results/official_baseline_comparison.summary.json
```

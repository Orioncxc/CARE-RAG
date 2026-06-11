# Result Summaries

This directory contains compact, public result summaries for CARE-RAG.

The full per-query outputs, downloaded datasets, caches, logs, model weights,
and intermediate experiment artifacts are intentionally not committed. They are
large, noisy, and easier to release separately through a GitHub Release or an
external artifact host if full reproducibility artifacts are needed.

## Files

- `main_results.csv`: headline three-dataset CARE-RAG results under
  PoisonedRAG-style attacks.
- `ablation_summary.csv`: compact comparison of the full and slimmed NQ
  variants used to choose the public method.

## Notes

- Each row is an aggregate over 100 poisoned attack queries unless noted.
- `contam_at_5_percent` is the mean poisoned-document share in the final top-5
  context.
- `true_defense_fail_count` counts ASR cases where a poisoned document reached
  top-5 and the LLM followed it.
- `llm_prior_leak_count` counts ASR cases where no poisoned document reached
  top-5 but the LLM still emitted the attack target.
- Confidence intervals are bootstrap 95% intervals where available.

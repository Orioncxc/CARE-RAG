# CARE-RAG Retrieval Framework

This directory adds a standalone framework for PoisonedRAG-style experiments
without changing the original project files.

## What It Adds

- `dense`: by default reads the original precomputed dense rankings from
  `results/beir_results/`, then scores only the injected poisoned documents
  online.
- `bm25`: retrieves with sparse keyword matching.
- `rrf`: fuses dense and BM25 rankings with Reciprocal Rank Fusion.
- `paper_hybrid`: follows the paper-style hybrid formula:
  `alpha * normalized_dense + (1 - alpha) * normalized_bm25`.
- `secure_ensemble`: combines multiple retriever channels with RRF,
  consensus-gated RRF, or robust-rank fusion, and can cap near-duplicate
  evidence clusters.
- `evidence_hardening`: optional post-retrieval layer that reranks a wider
  candidate pool with near-duplicate caps, answer-support diversity diagnostics,
  and lightweight contradiction diagnostics.
- `influence_probe.py`: optional leave-one-out LLM probe for small-scale
  document influence analysis. It is not run by default.

The experiment entrypoint reuses the original project data loader, prompt
template, LLM wrapper, and adversarial target files.

## Quick Commands

Retrieval-only smoke test:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2
```

Full LLM experiment with a selected retriever:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/dense_nq.json
```

Paper-style normalized hybrid:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/paper_hybrid_nq.json \
  --alpha 0.5
```

Security-oriented ensemble ablation:

```bash
./retrieval_framework/run_secure_ensemble_ablation.sh --skip_llm true
```

Evidence hardening retrieval-only run:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --skip_llm true
```

Evidence hardening ablation:

```bash
./retrieval_framework/run_evidence_hardening_ablation.sh --skip_llm true
```

The default `secure_ensemble_nq.json` is directly runnable with Contriever
precomputed rankings plus BM25. The `secure_ensemble_3way_nq.json` config adds
a second dense retriever and expects a matching precomputed file such as
`results/beir_results/nq-contriever-msmarco.json`.

Switch retrievers from the command line:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/dense_nq.json \
  --retriever bm25
```

## Config Notes

Main experiment fields:

- `dataset`: `nq`, `hotpotqa`, or `msmarco`.
- `top_k`: number of contexts passed to the LLM.
- `repeat_times` and `M`: total target queries are `repeat_times * M`.
- `attack_method`: use `LM_targeted` to inject the existing poisoned texts, or
  set it to `None` for clean retrieval.
- `skip_llm`: `true` runs retrieval metrics only.
- `max_corpus_docs`: optional cap for fast smoke tests. Ground-truth docs for
  selected target queries are still included in the indexed subset. This mainly
  affects BM25 because dense uses precomputed rankings by default.

Retriever fields:

- `retriever.type`: `dense`, `bm25`, `rrf`, `paper_hybrid`, or
  `secure_ensemble`.
- `retriever.dense.model_code`: existing model code from `src.utils`, such as
  `contriever`, `contriever-msmarco`, or `ance`.
- `retriever.dense.score_function`: `cos_sim` or `dot`. The provided NQ file
  uses `dot`, matching the original `run.py`.
- `retriever.dense.use_precomputed`: when `true`, dense reads
  `results/beir_results/{dataset}-{model_code}.json` instead of encoding the
  full corpus.
- `retriever.dense.precomputed_results_path`: optional explicit path. Leave it
  as `null` to infer the original project path.
- `retriever.bm25.k1` and `retriever.bm25.b`: BM25 parameters.
- `retriever.rrf_k`: RRF smoothing constant, commonly `60`.
- `retriever.alpha`: paper-style hybrid dense weight. BM25 weight is
  `1 - alpha`.
- `retriever.candidate_depth`: how many dense/BM25 candidates are fused before
  taking final `top_k`.
- `retriever.fusion`: for `secure_ensemble`, one of `rrf`,
  `consensus_rrf`, or `robust_rank`.
- `retriever.min_support`: minimum number of retriever channels that must
  retrieve a document before it can enter final ranking.
- `retriever.cluster_cap`: maximum documents kept from one near-duplicate
  cluster. Use `1` to reduce repeated poisoned evidence.
- `retriever.channels`: channel list for `secure_ensemble`; each channel has a
  `name`, `type`, optional `weight`, and retriever-specific config.

Evidence hardening fields:

- `evidence_hardening.enabled`: enables the post-retrieval hardening layer.
- `evidence_hardening.candidate_depth`: number of candidates requested from the
  retriever before reducing to final `top_k`.
- `evidence_hardening.cluster.cap`: maximum documents kept from one
  near-duplicate cluster in the hardening layer.
- `evidence_hardening.answer_support.enabled`: enables heuristic answer mention
  extraction and answer-support diversity scoring. Reference correct/incorrect
  answers are recorded only as diagnostics unless
  `use_reference_answers_for_scoring` is explicitly set to `true`.
- `evidence_hardening.answer_support.max_docs_per_answer`: optional cap on how
  many final contexts may share the same primary heuristic answer candidate.
  Use `0` to disable this cap.
- `evidence_hardening.contradiction.enabled`: penalizes answer candidates with
  weaker diverse support when multiple conflicting heuristic answers are seen.
- `evidence_hardening.store_raw_candidates`: stores the pre-hardening candidate
  pool in the result JSON. This is useful for debugging but increases file size.

## Outputs

Results are written under `retrieval_framework/results/` by default:

- `<run_name>.json`: per-query retrieved contexts, prompt, model output.
- `<run_name>.summary.json`: retrieval precision/recall/F1 and ASR if LLM
  generation was enabled.

For `paper_hybrid`, the automatically generated run name includes the alpha
value, for example `alpha0p7`, unless `--run_name` is explicitly set.

`secure_ensemble` summaries include additional diagnostics when available:
`support_count_mean`, `adv_support_count_mean`, singleton rates, and
`contamination_at_1/3/5`.

When `evidence_hardening` is enabled, summaries also include hardening
diagnostics such as filtered cluster count, candidate cluster count, max cluster
size, conflict rate, and reference correct/incorrect answer support counts.

Leave-one-out influence probing requires LLM calls and should be run only on a
small result file:

```bash
python retrieval_framework/influence_probe.py \
  --result_path retrieval_framework/results/<run>.json \
  --model_config_path model_configs/llama3_config.json \
  --max_queries 10
```

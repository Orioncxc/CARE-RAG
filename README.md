# CARE-RAG

Candidate-Answer Robust Evidence Hardening for Retrieval-Augmented Generation
under knowledge-corruption attacks.

CARE-RAG is a post-retrieval defense for RAG systems whose document corpus may
contain query-targeted poisoned passages. It treats poisoning as a data
integrity problem: instead of asking the generator to decide which context to
trust, it builds a widened candidate pool, consolidates near-duplicate evidence,
and selects answers with independent, multi-source, non-echoing support before
the LLM is invoked.

This repository builds on the public PoisonedRAG attack code and adds the
CARE-RAG retrieval, evidence-hardening, query-safety, baseline, and evaluation
components used for the accompanying paper.

## Highlights

- PoisonedRAG-style knowledge-corruption evaluation on Natural Questions,
  HotpotQA, and MS MARCO.
- Dense, BM25, RRF, normalized hybrid, and consensus ensemble retrievers.
- CARE-RAG evidence hardening with candidate-answer support scoring,
  near-duplicate source clustering, echo penalties, and margin-gate variants.
- Baseline adapters for RAGDefender, InstructRAG, AstuteRAG, RobustRAG, RAG2RAG,
  and SeCon-RAG style comparisons.
- Retrieval-only smoke tests for quick validation without loading an LLM.

## Repository Layout

```text
retrieval_framework/       CARE-RAG retrievers, hardening logic, configs, tests
baselines/                 Baseline wrappers and shared evaluation utilities
official_baselines/        Official-source baseline comparison scripts
src/                       Original PoisonedRAG model, attack, and utility code
weak_attack_eval/          Additional weak-attack evaluation utilities
prepare_dataset.py         BEIR dataset download helper
run.py                     Original PoisonedRAG experiment entrypoint
```

Generated datasets, model weights, caches, logs, and large result files are
ignored by default. Keep released benchmark artifacts small and documented.

## Installation

The main code path targets Python 3.10. GPU-backed LLM experiments require a
matching PyTorch/CUDA installation for your machine.

```bash
conda create -n care-rag python=3.10
conda activate care-rag
pip install -r requirements.txt
```

If you need CUDA-specific PyTorch wheels, install PyTorch first from the
official PyTorch selector, then run `pip install -r requirements.txt`.

## Model Configuration

Model configs live under `model_configs/`. Public configs do not contain real
API keys or machine-local model paths.

For API-backed models, prefer environment variables when supported by the
provider SDK. For gated Hugging Face models, set:

```bash
export HF_TOKEN=<your-token>
```

For local Llama-style inference, either edit `model_configs/llama3_config.json`
to add your local model path:

```json
"model_info": {
  "provider": "llama",
  "name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "local_path": "/absolute/path/to/local/model"
}
```

or leave `local_path` unset and let `transformers` resolve the Hugging Face
model name.

## Data

BEIR datasets are downloaded on demand by the original loader. You can also
prepare them explicitly:

```bash
python prepare_dataset.py
```

The downloaded `datasets/` directory is intentionally ignored because the full
corpora are several gigabytes.

## Quick Start

Retrieval-only smoke test, no LLM required:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

CARE-RAG evidence hardening, retrieval-only:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

Full LLM evaluation with Llama 3:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --model_name llama3 \
  --model_config_path model_configs/llama3_config.json
```

Original PoisonedRAG reproduction entrypoint:

```bash
python run.py
```

## Important Configs

- `retrieval_framework/configs/dense_nq.json`: dense Contriever baseline.
- `retrieval_framework/configs/bm25_nq.json`: BM25 baseline.
- `retrieval_framework/configs/rrf_nq.json`: dense + BM25 reciprocal rank
  fusion.
- `retrieval_framework/configs/paper_hybrid_nq.json`: normalized dense/BM25
  hybrid retriever.
- `retrieval_framework/configs/evidence_hardened_nq.json`: CARE-RAG evidence
  hardening on top of a widened retrieval pool.
- `retrieval_framework/configs/evidence_hardened_focused_nq.json` and later
  variants: focused hardening and margin-gate experiments.

See `retrieval_framework/README.md` for the full configuration reference.

## Outputs

Experiment outputs are written to `retrieval_framework/results/` unless
overridden with `--output_dir`.

- `<run_name>.json`: per-query contexts, prompts, and model outputs.
- `<run_name>.summary.json`: retrieval metrics and attack success rate when LLM
  generation is enabled.

These outputs are ignored by default. Commit only compact, intentional result
summaries that you want to publish.

## Testing

Run the lightweight unit tests:

```bash
python -m pytest retrieval_framework/test_*.py
```

Run a smoke experiment:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 1 \
  --max_corpus_docs 500
```

## Responsible Use

This repository contains code for evaluating poisoning attacks and defenses in
RAG systems. Use it only for research, benchmarking, and defensive evaluation
on systems and corpora you are authorized to test. Do not use the attack
components to compromise third-party services or publish harmful poisoned
content.

## Acknowledgements

This project reuses and extends code from:

- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)
- [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning)
- [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection)
- [BEIR](https://github.com/beir-cellar/beir)
- [Contriever](https://github.com/facebookresearch/contriever)

## Citation

If you use CARE-RAG, cite the project paper:

```bibtex
@misc{chen2026carerag,
  title = {CARE-RAG: Candidate-Answer Robust Evidence Hardening for Retrieval-Augmented Generation under Knowledge-Corruption Attacks},
  author = {Chen, Xingchen and Sun, Fanghui and Wang, Kaiming and Ma, Ying},
  year = {2026},
  note = {Manuscript}
}
```

If you use the inherited attack implementation or benchmark setup, also cite
PoisonedRAG:

```bibtex
@inproceedings{zou2025poisonedrag,
  title = {{PoisonedRAG}: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models},
  author = {Zou, Wei and Geng, Runpeng and Wang, Binghui and Jia, Jinyuan},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  pages = {3827--3844},
  year = {2025}
}
```

## License

The code is released under the MIT License. See `LICENSE`.

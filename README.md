# CARE-RAG

Candidate-Answer Robust Evidence Hardening for Retrieval-Augmented Generation
under knowledge-corruption attacks.

CARE-RAG is a post-retrieval defense for RAG systems whose corpus may contain
query-targeted poisoned passages. It widens the retrieval pool, consolidates
near-duplicate evidence, and selects answers with independent, multi-source,
non-echoing support before the LLM is invoked.

This slim repository contains the core CARE-RAG implementation, the original
PoisonedRAG-compatible attack/evaluation code needed to run it, public model
config templates, and lightweight tests.

## Layout

```text
retrieval_framework/       CARE-RAG retrievers, evidence hardening, configs
src/                       Original PoisonedRAG model, attack, and utilities
model_configs/             Public model config templates without real keys
prepare_dataset.py         BEIR dataset download helper
evaluate_beir.py           Dense retrieval precomputation helper
run.py, main.py            Original PoisonedRAG-compatible entrypoints
```

Large datasets, caches, logs, model weights, result dumps, baseline clones, and
draft experiment scripts are intentionally excluded from the public release.

## Installation

Python 3.10 is recommended.

```bash
conda create -n care-rag python=3.10
conda activate care-rag
pip install -r requirements.txt
```

For CUDA experiments, install the PyTorch build matching your machine before
installing the remaining requirements.

## Data

The code uses BEIR-format datasets. Download the default NQ, HotpotQA, and MS
MARCO corpora with:

```bash
python prepare_dataset.py
```

The downloaded `datasets/` directory is ignored because it is several
gigabytes.

## Model Configuration

Model configs live in `model_configs/`. Public configs contain placeholders
only. For gated Hugging Face models, prefer:

```bash
export HF_TOKEN=<your-token>
```

For local inference, create a local-only copy such as
`model_configs/llama3.local.json` and add:

```json
"local_path": "/absolute/path/to/local/model"
```

`*.local.json` files are ignored by Git.

## Quick Start

Retrieval-only smoke test:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

CARE-RAG evidence hardening:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 2 \
  --max_corpus_docs 1000
```

Full LLM-backed run:

```bash
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/evidence_hardened_nq.json \
  --model_name llama3 \
  --model_config_path model_configs/llama3_config.json
```

## Included Configs

- `dense_nq.json`: dense Contriever baseline.
- `bm25_nq.json`: BM25 baseline.
- `rrf_nq.json`: dense + BM25 reciprocal rank fusion.
- `paper_hybrid_nq.json`: normalized dense/BM25 hybrid.
- `evidence_hardened_nq.json`: CARE-RAG evidence hardening.

See `retrieval_framework/README.md` for details.

## Tests

```bash
python -m pytest retrieval_framework/test_*.py
```

## Responsible Use

This repository includes attack-evaluation code for studying RAG corpus
poisoning. Use it only for research, benchmarking, and defensive evaluation on
systems and corpora you are authorized to test.

## Citation

```bibtex
@misc{chen2026carerag,
  title = {CARE-RAG: Candidate-Answer Robust Evidence Hardening for Retrieval-Augmented Generation under Knowledge-Corruption Attacks},
  author = {Chen, Xingchen and Sun, Fanghui and Wang, Kaiming and Ma, Ying},
  year = {2026},
  note = {Manuscript}
}
```

If you use the inherited attack implementation or benchmark setup, also cite:

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

MIT License. See `LICENSE`.

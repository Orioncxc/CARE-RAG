# Open Source Release Checklist

Use this checklist before publishing the repository on GitHub.

## Before `git add`

- Confirm `README.md` describes CARE-RAG and no longer presents the project as
  the official PoisonedRAG repository.
- Replace any placeholder citation or repository URLs after the GitHub repo is
  created.
- Keep real API keys out of `model_configs/` and `server_package/model_config/`.
- Store machine-specific model paths only in `*.local.json` files.
- Keep large generated files out of the repository:
  - `datasets/`
  - `results/`
  - `retrieval_framework/results/`
  - `retrieval_framework/cache/`
  - `logs/`
  - `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.pkl`

## Sanity Checks

```bash
python -m json.tool model_configs/llama3_config.json >/dev/null
python -m pytest retrieval_framework/test_*.py
python retrieval_framework/run_experiment.py \
  --config retrieval_framework/configs/rrf_nq.json \
  --skip_llm true \
  --repeat_times 1 \
  --M 1 \
  --max_corpus_docs 500
```

## Initialize and Publish

```bash
git init
git status --ignored --short
git add README.md LICENSE CITATION.cff CONTRIBUTING.md SECURITY.md \
  OPEN_SOURCE_CHECKLIST.md requirements.txt .gitignore \
  model_configs src retrieval_framework \
  baselines official_baselines weak_attack_eval \
  prepare_dataset.py evaluate_beir.py run.py main.py
git status --short
git commit -m "Prepare CARE-RAG open source release"
git branch -M main
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

Review `git status --short` before committing. Do not use `git add .` until
you are comfortable with every untracked file that would be included.

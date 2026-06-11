# Contributing

Thanks for your interest in CARE-RAG.

## Development Setup

```bash
conda create -n care-rag python=3.10
conda activate care-rag
pip install -r requirements.txt
```

Run the lightweight tests before opening a pull request:

```bash
python -m pytest retrieval_framework/test_*.py
```

For experiment changes, include the command you ran and the generated summary
metrics. Prefer retrieval-only smoke runs for quick validation when the change
does not affect generation.

## Pull Requests

- Keep changes focused on one feature, baseline, experiment, or fix.
- Do not commit downloaded datasets, model weights, local model paths, API keys,
  caches, or full experiment dumps.
- Add or update configs when introducing a new experiment mode.
- Update documentation when command-line flags, expected outputs, or result
  schemas change.

## Issues

When reporting a bug, include:

- The command and config used.
- Python version and operating system.
- Whether the run was retrieval-only or LLM-backed.
- The relevant traceback or summary JSON.

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "weak_attack_eval/attacks/01_answer_only_misinformation/config.json"
  "weak_attack_eval/attacks/02_keyword_stuffing_misinformation/config.json"
  "weak_attack_eval/attacks/03_query_copy_misinformation/config.json"
  "weak_attack_eval/attacks/04_paraphrase_misinformation/config.json"
  "weak_attack_eval/attacks/05_answer_swap_misinformation/config.json"
  "weak_attack_eval/attacks/06_multi_doc_consensus/config.json"
  "weak_attack_eval/attacks/07_hybrid_keyword_dense_misinformation/config.json"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  "${PYTHON_BIN}" weak_attack_eval/run_weak_attack.py --config "${config}" "$@"
done


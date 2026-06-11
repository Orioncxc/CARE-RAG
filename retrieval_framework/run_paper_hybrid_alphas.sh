#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-retrieval_framework/configs/paper_hybrid_nq.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-retrieval_framework/results/logs}"
ALPHAS="${ALPHAS:-0.7 0.5 0.3}"

mkdir -p "$LOG_DIR"

for alpha in $ALPHAS; do
  alpha_tag="alpha${alpha//./p}"
  log_path="$LOG_DIR/paper_hybrid_${alpha_tag}.log"

  echo "============================================================"
  echo "Running paper_hybrid with alpha=${alpha}"
  echo "Log: ${log_path}"
  echo "============================================================"

  "$PYTHON_BIN" retrieval_framework/run_experiment.py \
    --config "$CONFIG_PATH" \
    --alpha "$alpha" \
    "$@" 2>&1 | tee "$log_path"
done

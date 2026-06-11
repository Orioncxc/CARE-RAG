#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-retrieval_framework/configs/secure_ensemble_nq.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-retrieval_framework/results/logs}"
FUSIONS="${FUSIONS:-rrf consensus_rrf robust_rank}"
CLUSTER_CAPS="${CLUSTER_CAPS:-0 1}"

mkdir -p "$LOG_DIR"

for fusion in $FUSIONS; do
  for cap in $CLUSTER_CAPS; do
    log_path="$LOG_DIR/secure_ensemble_${fusion}_cap${cap}.log"
    echo "============================================================"
    echo "Running secure_ensemble fusion=${fusion} cluster_cap=${cap}"
    echo "Log: ${log_path}"
    echo "============================================================"

    "$PYTHON_BIN" -u retrieval_framework/run_experiment.py \
      --config "$CONFIG_PATH" \
      --fusion "$fusion" \
      --cluster_cap "$cap" \
      "$@" 2>&1 | tee "$log_path"
  done
done

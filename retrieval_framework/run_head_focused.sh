#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-retrieval_framework/configs/head_focused_nq.json}"
OUTPUT_DIR="${OUTPUT_DIR:-retrieval_framework/results/head_focused}"
SKIP_LLM="${SKIP_LLM:-true}"
M="${M:-10}"
REPEAT_TIMES="${REPEAT_TIMES:-10}"

mkdir -p "$OUTPUT_DIR/logs"

log_name="head_focused_skipllm_${SKIP_LLM}_M${M}x${REPEAT_TIMES}.log"

python -m retrieval_framework.run_experiment \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --skip_llm "$SKIP_LLM" \
  --M "$M" \
  --repeat_times "$REPEAT_TIMES" \
  2>&1 | tee "$OUTPUT_DIR/logs/$log_name"

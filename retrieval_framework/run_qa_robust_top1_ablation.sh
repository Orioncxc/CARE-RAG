#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-retrieval_framework/configs/evidence_hardened_qa_only_robust_nq.json}"
OUTPUT_DIR="${OUTPUT_DIR:-retrieval_framework/results/evidence_qa_only_robust_top1_ablation}"
LOG_DIR="${LOG_DIR:-retrieval_framework/results/logs}"
SKIP_LLM="${SKIP_LLM:-true}"
SUMMARIZE_AFTER="${SUMMARIZE_AFTER:-true}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/summary_table.csv}"
RUN_BASELINE="${RUN_BASELINE:-true}"

TOP1_PENALTIES="${TOP1_PENALTIES:-0.015 0.025 0.04}"
ROBUST_NON_ECHO_BONUSES="${ROBUST_NON_ECHO_BONUSES:-0.004}"
ROBUST_ISOLATED_PENALTIES="${ROBUST_ISOLATED_PENALTIES:-0.006 0.008 0.012}"
ROBUST_QUERY_ECHO_PENALTIES="${ROBUST_QUERY_ECHO_PENALTIES:-0.008}"
ROBUST_CHANNEL_BONUSES="${ROBUST_CHANNEL_BONUSES:-0.002}"
TOP1_ALT_CLUSTERS_LIST="${TOP1_ALT_CLUSTERS_LIST:-2}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$RUN_BASELINE" == "true" ]]; then
  run_name="qa-robust-baseline-no-top1"
  log_path="$LOG_DIR/${run_name}.log"
  echo "============================================================"
  echo "Running ${run_name}"
  echo "Log: ${log_path}"
  echo "============================================================"

  "$PYTHON_BIN" -u retrieval_framework/run_experiment.py \
    --config "$CONFIG_PATH" \
    --skip_llm "$SKIP_LLM" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name" \
    --top1_dominance false \
    "$@" 2>&1 | tee "$log_path"
fi

for top1_alt_clusters in $TOP1_ALT_CLUSTERS_LIST; do
  for top1_penalty in $TOP1_PENALTIES; do
    for non_echo_bonus in $ROBUST_NON_ECHO_BONUSES; do
      for isolated_penalty in $ROBUST_ISOLATED_PENALTIES; do
        for query_echo_penalty in $ROBUST_QUERY_ECHO_PENALTIES; do
          for channel_bonus in $ROBUST_CHANNEL_BONUSES; do
            penalty_tag="${top1_penalty//./p}"
            alt_tag="${top1_alt_clusters//./p}"
            non_echo_tag="${non_echo_bonus//./p}"
            isolated_tag="${isolated_penalty//./p}"
            query_echo_tag="${query_echo_penalty//./p}"
            channel_tag="${channel_bonus//./p}"
            run_name="qa-robust-top1-p${penalty_tag}-alt${alt_tag}-ne${non_echo_tag}-iso${isolated_tag}-qe${query_echo_tag}-ch${channel_tag}"
            log_path="$LOG_DIR/${run_name}.log"

            echo "============================================================"
            echo "Running ${run_name}"
            echo "top1_penalty=${top1_penalty} alt_clusters=${top1_alt_clusters} non_echo_bonus=${non_echo_bonus} isolated_penalty=${isolated_penalty} query_echo_penalty=${query_echo_penalty} channel_bonus=${channel_bonus}"
            echo "Log: ${log_path}"
            echo "============================================================"

            "$PYTHON_BIN" -u retrieval_framework/run_experiment.py \
              --config "$CONFIG_PATH" \
              --skip_llm "$SKIP_LLM" \
              --output_dir "$OUTPUT_DIR" \
              --run_name "$run_name" \
              --top1_dominance true \
              --top1_penalty "$top1_penalty" \
              --top1_alt_clusters "$top1_alt_clusters" \
              --robust_non_echo_bonus "$non_echo_bonus" \
              --robust_channel_bonus "$channel_bonus" \
              --robust_query_echo_penalty "$query_echo_penalty" \
              --robust_isolated_penalty "$isolated_penalty" \
              "$@" 2>&1 | tee "$log_path"
          done
        done
      done
    done
  done
done

if [[ "$SUMMARIZE_AFTER" == "true" ]]; then
  echo "============================================================"
  echo "Summarizing ablation results"
  echo "Summary CSV: ${SUMMARY_PATH}"
  echo "============================================================"
  "$PYTHON_BIN" retrieval_framework/summarize_results.py \
    --results_dir "$OUTPUT_DIR" \
    --output "$SUMMARY_PATH"
fi

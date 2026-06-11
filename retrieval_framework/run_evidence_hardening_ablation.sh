#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-retrieval_framework/configs/evidence_hardened_nq.json}"
OUTPUT_DIR="${OUTPUT_DIR:-retrieval_framework/results/logs}"
SKIP_LLM="${SKIP_LLM:-true}"
CLUSTER_CAPS="${CLUSTER_CAPS:-0 1 2}"
ANSWER_CAPS="${ANSWER_CAPS:-0 2}"
ANSWER_SUPPORTS="${ANSWER_SUPPORTS:-false true}"

mkdir -p "${OUTPUT_DIR}"

for cap in ${CLUSTER_CAPS}; do
  for answer_cap in ${ANSWER_CAPS}; do
    for answer_support in ${ANSWER_SUPPORTS}; do
      log_path="${OUTPUT_DIR}/evidence_hardening_cap${cap}_answer${answer_support}_answercap${answer_cap}.log"
      echo "============================================================"
      echo "Running evidence_hardening cluster_cap=${cap} answer_support=${answer_support} answer_cap=${answer_cap}"
      echo "Log: ${log_path}"
      echo "============================================================"
      "${PYTHON_BIN}" -u retrieval_framework/run_experiment.py \
        --config "${CONFIG}" \
        --skip_llm "${SKIP_LLM}" \
        --hardening_cluster_cap "${cap}" \
        --hardening_answer_support "${answer_support}" \
        --hardening_answer_cap "${answer_cap}" \
        "$@" | tee "${log_path}"
    done
  done
done

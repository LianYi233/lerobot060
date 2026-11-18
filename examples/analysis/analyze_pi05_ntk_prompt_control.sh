#!/usr/bin/env bash
# Measure the uninterrupted prompt-learning run using the original staged probes.
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash $0 CONTROL_RUN_DIR STAGED_RESULTS_JSON [GPU_ID] [extra analysis arguments...]" >&2
  echo "Use '-' as STAGED_RESULTS_JSON for standalone analysis." >&2
  exit 2
fi
RUN_DIR="$1"
REFERENCE_RESULTS="$2"
GPU_ID="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift 2; fi
ARGS=(
  --schedule=prompt-only
  --run-dir="${RUN_DIR}"
  --through-step="${THROUGH_STEP:-4000}"
  --dataset-repo-id="${DATASET_REPO_ID:-libero}"
  --dataset-root="${DATASET_ROOT:-/root/autodl-tmp/datasets/libero}"
  --output-dir="${NTK_OUTPUT_DIR:-${RUN_DIR}/ntk_prompt_control}"
)
if [[ "${REFERENCE_RESULTS}" != - ]]; then
  if [[ ! -f "${REFERENCE_RESULTS}" ]]; then
    echo "Reference results not found: ${REFERENCE_RESULTS}" >&2
    exit 2
  fi
  ARGS+=(--reference-results="${REFERENCE_RESULTS}")
fi
if [[ -n "${TOKENIZER_PATH:-}" ]]; then ARGS+=(--tokenizer-path="${TOKENIZER_PATH}"); fi
if [[ -n "${NTK_FONT_PATH:-}" ]]; then ARGS+=(--font-path="${NTK_FONT_PATH}"); fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m lerobot.scripts.analyze_pi05_ntk_stages "${ARGS[@]}" "$@"

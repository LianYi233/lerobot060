#!/usr/bin/env bash
# Sequential SAP-vs-Shuffled data-scale sweep for the user's local A100 server.
# New checkpoints, prepared data, caches, logs and plots live on /data1 only.
set -euo pipefail

if [[ -z "${DATASET_ROOT:-}" || -z "${BASE_PI05:-}" ]]; then
  echo "[ERROR] Set DATASET_ROOT and BASE_PI05 to existing local paths." >&2
  echo "Example: DATASET_ROOT=/data1/datasets/libero BASE_PI05=/data1/models/pi05_libero_base bash examples/analysis/run_sap_structure_v4_local_a100.sh" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" || ! -d "$BASE_PI05" ]]; then
  echo "[ERROR] DATASET_ROOT and BASE_PI05 must be existing directories." >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/sap-structure-v4}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-750}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TEST_CHUNKS="${TEST_CHUNKS:-200}"
EVAL_REPEATS="${EVAL_REPEATS:-2}"
EVAL_SEED="${EVAL_SEED:-117}"
HF_HOME="${HF_HOME:-/data1/wyn/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/wyn/cache/xdg}"
TMPDIR="${TMPDIR:-/data1/wyn/tmp/sap-structure-v4}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data1/wyn/cache/huggingface/datasets}"
TORCH_HOME="${TORCH_HOME:-/data1/wyn/cache/torch}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/data1/wyn/cache/matplotlib}"
export HF_HOME XDG_CACHE_HOME TMPDIR HF_DATASETS_CACHE TORCH_HOME MPLCONFIGDIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ "$OUT_ROOT" != /data1/* || "$HF_HOME" != /data1/* || "$TMPDIR" != /data1/* || "$XDG_CACHE_HOME" != /data1/* || "$HF_DATASETS_CACHE" != /data1/* || "$TORCH_HOME" != /data1/* || "$MPLCONFIGDIR" != /data1/* ]]; then
  echo "[ERROR] All output and cache paths must be below /data1." >&2
  exit 2
fi
if [[ ! -d /data1 ]]; then
  echo "[ERROR] /data1 is not mounted on this server." >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ && "$TRAIN_STEPS" =~ ^[0-9]+$ && "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED, TRAIN_STEPS and GPU_ID must be non-negative integers." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR/two_sample_sap_structure_v4.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED="$OUT_ROOT/prepared"

mkdir -p "$OUT_ROOT/logs" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR"
echo "[CONFIG] Python=$PYTHON_BIN"
echo "[CONFIG] GPU=$CUDA_VISIBLE_DEVICES, train steps=$TRAIN_STEPS, batch=$BATCH_SIZE"
echo "[CONFIG] dataset=$DATASET_ROOT"
echo "[CONFIG] pretrained=$BASE_PI05"
echo "[CONFIG] all new artifacts=$OUT_ROOT"

# Deterministic, episode-disjoint data; shared by all four N groups.
if [[ -f "$PREPARED/prepared_v4.pt" && -f "$PREPARED/manifest_v4.json" ]]; then
  echo "[SKIP] Prepared data already complete: $PREPARED"
elif [[ -e "$PREPARED" ]]; then
  echo "[ERROR] Incomplete prepared directory; inspect it before retrying: $PREPARED" >&2
  exit 1
else
  "$PYTHON_BIN" "$CODE" prepare \
    --dataset-root "$DATASET_ROOT" --output "$PREPARED" \
    --sizes 2,32,128,512 --test-chunks "$TEST_CHUNKS" --seed "$SEED" \
    2>&1 | tee "$OUT_ROOT/logs/prepare.log"
fi

for N in 2 32 128 512; do
  echo "[GROUP] Starting training and evaluation for N=$N"
  for VARIANT in sap shuffled; do
    TRAIN_DIR="$OUT_ROOT/train/n${N}-${VARIANT}-seed${SEED}"
    if [[ -f "$TRAIN_DIR/action_prompt.pt" && -f "$TRAIN_DIR/train_metadata.json" ]]; then
      echo "[SKIP] Completed training: $TRAIN_DIR"
    elif [[ -e "$TRAIN_DIR" ]]; then
      echo "[ERROR] Incomplete training directory; inspect it before retrying: $TRAIN_DIR" >&2
      exit 1
    else
      "$PYTHON_BIN" "$CODE" train \
        --prepared "$PREPARED/prepared_v4.pt" --base-checkpoint "$BASE_PI05" \
        --variant "$VARIANT" --num-chunks "$N" \
        --masked-steps 40 --steps "$TRAIN_STEPS" --batch-size "$BATCH_SIZE" \
        --seed "$SEED" --device cuda --output "$TRAIN_DIR" \
        2>&1 | tee "$OUT_ROOT/logs/train-n${N}-${VARIANT}.log"
    fi
  done

  EVAL_DIR="$OUT_ROOT/eval/n${N}-seed${SEED}"
  if [[ -f "$EVAL_DIR/metrics_v4.json" && -f "$EVAL_DIR/per_episode_v4.json" ]]; then
    echo "[SKIP] Completed evaluation: $EVAL_DIR"
  elif [[ -e "$EVAL_DIR" ]]; then
    echo "[ERROR] Incomplete evaluation directory; inspect it before retrying: $EVAL_DIR" >&2
    exit 1
  else
    "$PYTHON_BIN" "$CODE" evaluate \
      --prepared "$PREPARED/prepared_v4.pt" --base-checkpoint "$BASE_PI05" \
      --sap "$OUT_ROOT/train/n${N}-sap-seed${SEED}" \
      --shuffled "$OUT_ROOT/train/n${N}-shuffled-seed${SEED}" \
      --num-chunks "$N" --train-masked-steps 40 --eval-masks 40,20 \
      --repeats "$EVAL_REPEATS" --seed "$EVAL_SEED" --device cuda \
      --output "$EVAL_DIR" \
      2>&1 | tee "$OUT_ROOT/logs/eval-n${N}.log"
  fi
done

SUMMARY_DIR="$OUT_ROOT/summary-seed$SEED"
if [[ -f "$SUMMARY_DIR/sample_size_summary_v4.json" ]]; then
  echo "[SKIP] Completed summary: $SUMMARY_DIR"
elif [[ -e "$SUMMARY_DIR" ]]; then
  echo "[ERROR] Incomplete summary directory; inspect it before retrying: $SUMMARY_DIR" >&2
  exit 1
else
  "$PYTHON_BIN" "$CODE" summarize \
    --input-root "$OUT_ROOT" --sizes 2,32,128,512 \
    --primary-mask 40 --seed "$SEED" --output "$SUMMARY_DIR" \
    2>&1 | tee "$OUT_ROOT/logs/summary.log"
fi
echo "[DONE] Four sequential data-scale groups completed. Output: $OUT_ROOT"

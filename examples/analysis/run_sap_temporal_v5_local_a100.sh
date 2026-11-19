#!/usr/bin/env bash
# Sequential, resumable P0 -> P1 -> P2 -> P3 -> P4 temporal-action evidence.
# Read LIBERO/pi05 from /data; ALL NEW results and caches remain on /data1.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/data/datasets/libero}"
BASE_PI05="${BASE_PI05:-/data/models/lerobot/pi05_libero_base/}"
V4_ROOT="${V4_ROOT:-/data1/wyn/chkpt/2601-lerobot/sap-structure-v4}"
OUT_ROOT="${OUT_ROOT:-/data1/wyn/chkpt/2601-lerobot/sap-temporal-v5}"
GPU_ID="${GPU_ID:-0}"
REAL_N="${REAL_N:-128}"
TEST_LIMIT="${TEST_LIMIT:-200}"
EVAL_BATCH="${EVAL_BATCH:-4}"
ODE_STEPS="${ODE_STEPS:-16}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
SYN_N="${SYN_N:-256}"
SYN_STEPS="${SYN_STEPS:-2000}"
SYN_BATCH="${SYN_BATCH:-4}"
SYN_TEST="${SYN_TEST:-120}"
SEED="${SEED:-0}"
EVAL_SEED="${EVAL_SEED:-117}"
PHASE="${PHASE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for path in "$V4_ROOT" "$OUT_ROOT"; do
  if [[ "$path" != /data1/* ]]; then
    echo "[ERROR] New results and V4 inputs must be below /data1: $path" >&2
    exit 2
  fi
done
if [[ ! -d /data1 || ! -d "$DATASET_ROOT" || ! -d "$BASE_PI05" ]]; then
  echo "[ERROR] Missing /data1, LIBERO $DATASET_ROOT, or checkpoint $BASE_PI05" >&2
  exit 2
fi
case "$PHASE" in
  all|P0|P1|P2|P3|P4) ;;
  *) echo "[ERROR] PHASE must be all or P0/P1/P2/P3/P4" >&2; exit 2 ;;
esac
for number in "$GPU_ID" "$REAL_N" "$TEST_LIMIT" "$EVAL_BATCH" "$ODE_STEPS" "$SYN_N" "$SYN_STEPS" "$SYN_BATCH" "$SYN_TEST" "$SEED" "$EVAL_SEED"; do
  if ! [[ "$number" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Expected numeric settings, got: $number" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-/data1/wyn/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data1/wyn/cache/huggingface/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/wyn/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-/data1/wyn/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data1/wyn/cache/matplotlib}"
export TMPDIR="${TMPDIR:-/data1/wyn/tmp/sap-temporal-v5}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
for path in "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR"; do
  if [[ "$path" != /data1/* ]]; then
    echo "[ERROR] Cache/temp must be under /data1: $path" >&2
    exit 2
  fi
  mkdir -p "$path"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR/sap_temporal_evidence_v5.py"
V4_CODE="$SCRIPT_DIR/two_sample_sap_structure_v4.py"
PREPARED="$V4_ROOT/prepared/prepared_v4.pt"
SAP="$V4_ROOT/train/n${REAL_N}-sap-seed${SEED}"
SHUFFLED="$V4_ROOT/train/n${REAL_N}-shuffled-seed${SEED}"
mkdir -p "$OUT_ROOT/logs"

echo "[CONFIG] phase=$PHASE, GPU=$GPU_ID, Python=$PYTHON_BIN"
echo "[CONFIG] dataset=$DATASET_ROOT, pretrained=$BASE_PI05"
echo "[CONFIG] V4 input=$V4_ROOT, V5 new output=$OUT_ROOT"

run_once() {
  local title="$1" required="$2" logfile="$3"
  shift 3
  if [[ -f "$required" ]]; then
    echo "[SKIP] $title: $required"
    return 0
  fi
  if [[ -e "$(dirname "$required")" ]]; then
    echo "[ERROR] Incomplete output for $title: $(dirname "$required")" >&2
    echo "[ERROR] Inspect and manually relocate the incomplete directory before retrying." >&2
    exit 1
  fi
  echo "[RUN] $title"
  "$@" 2>&1 | tee "$OUT_ROOT/logs/$logfile"
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] $title finished without expected artifact: $required" >&2
    exit 1
  fi
}

if [[ "$PHASE" == all || "$PHASE" == P0 || "$PHASE" == P1 || "$PHASE" == P3 ]]; then
  for item in "$PREPARED" "$SAP/action_prompt.pt" "$SAP/train_metadata.json" \
              "$SHUFFLED/action_prompt.pt" "$SHUFFLED/train_metadata.json"; do
    if [[ ! -f "$item" ]]; then
      echo "[ERROR] Missing completed V4 experiment dependency: $item" >&2
      exit 2
    fi
  done
fi

if [[ "$PHASE" == all || "$PHASE" == P0 ]]; then
  run_once "P0 real-data order information" \
    "$OUT_ROOT/P0-data-check/data_order_preflight_v5.json" P0-data-check.log \
    "$PYTHON_BIN" "$CODE" data-check \
      --prepared "$PREPARED" --context 20 --horizons 5,10,20 \
      --seed "$EVAL_SEED" --output "$OUT_ROOT/P0-data-check"
fi

if [[ "$PHASE" == all || "$PHASE" == P1 ]]; then
  run_once "P1 contiguous/future order intervention" \
    "$OUT_ROOT/P1-blocked/metrics_temporal_v5.json" P1-blocked.log \
    "$PYTHON_BIN" "$CODE" blocked-eval \
      --prepared "$PREPARED" --base-checkpoint "$BASE_PI05" \
      --sap "$SAP" --shuffled "$SHUFFLED" --num-chunks "$REAL_N" \
      --modes future10,middle20,random20 --flow-times 0.5,0.99 \
      --test-limit "$TEST_LIMIT" --batch-size "$EVAL_BATCH" \
      --ode-steps "$ODE_STEPS" --repeats "$EVAL_REPEATS" \
      --seed "$EVAL_SEED" --save-rollouts \
      --output "$OUT_ROOT/P1-blocked"
fi

SYN_ROOT="$OUT_ROOT/P2-synthetic"
SYN_PREPARED="$SYN_ROOT/prepared/prepared_v4.pt"
SYN_SAP="$SYN_ROOT/train/n${SYN_N}-sap-seed${SEED}"
SYN_SHUFFLED="$SYN_ROOT/train/n${SYN_N}-shuffled-seed${SEED}"

if [[ "$PHASE" == all || "$PHASE" == P2 ]]; then
  run_once "P2a known-AR2 data preparation" "$SYN_ROOT/prepared/manifest_v4.json" P2-prepare.log \
    "$PYTHON_BIN" "$CODE" synthetic-prepare \
      --train-chunks "$SYN_N" --test-chunks "$SYN_TEST" --seed 2026 \
      --output "$SYN_ROOT/prepared"
  for variant in sap shuffled; do
    if [[ "$variant" == sap ]]; then folder="$SYN_SAP"; else folder="$SYN_SHUFFLED"; fi
    run_once "P2b matched synthetic $variant prompt training" \
      "$folder/train_metadata.json" "P2-train-$variant.log" \
      "$PYTHON_BIN" "$V4_CODE" train \
        --prepared "$SYN_PREPARED" --base-checkpoint "$BASE_PI05" \
        --variant "$variant" --num-chunks "$SYN_N" \
        --steps "$SYN_STEPS" --batch-size "$SYN_BATCH" \
        --masked-steps 40 --seed "$SEED" --device cuda \
        --output "$folder"
  done
  run_once "P2c synthetic future dynamics sampling" \
    "$SYN_ROOT/eval/metrics_temporal_v5.json" P2-eval.log \
    "$PYTHON_BIN" "$CODE" blocked-eval \
      --prepared "$SYN_PREPARED" --base-checkpoint "$BASE_PI05" \
      --sap "$SYN_SAP" --shuffled "$SYN_SHUFFLED" --num-chunks "$SYN_N" \
      --modes future20 --flow-times 0.5,0.99 \
      --test-limit "$SYN_TEST" --batch-size "$EVAL_BATCH" \
      --ode-steps "$ODE_STEPS" --repeats 1 --seed "$EVAL_SEED" \
      --save-rollouts --output "$SYN_ROOT/eval"
  if [[ ! -f "$SYN_ROOT/eval/rollouts_temporal_v5.npz" ]]; then
    echo "[ERROR] Completed P2 evaluation is missing full-ODE rollouts" >&2
    exit 1
  fi
  run_once "P2d recovery of known dynamical coefficients" \
    "$SYN_ROOT/recovery/ar2_recovery_v5.json" P2-recovery.log \
    "$PYTHON_BIN" "$CODE" synthetic-score \
      --prepared "$SYN_PREPARED" \
      --rollouts "$SYN_ROOT/eval/rollouts_temporal_v5.npz" \
      --mode future20 --output "$SYN_ROOT/recovery"
fi

if [[ "$PHASE" == all || "$PHASE" == P3 ]]; then
  run_once "P3 expert hidden-state predictive probe" \
    "$OUT_ROOT/P3-probe/future_probe_v5.json" P3-probe.log \
    "$PYTHON_BIN" "$CODE" probe \
      --prepared "$PREPARED" --base-checkpoint "$BASE_PI05" \
      --sap "$SAP" --shuffled "$SHUFFLED" --num-chunks "$REAL_N" \
      --test-limit "$TEST_LIMIT" --batch-size "$EVAL_BATCH" \
      --context 20 --horizons 1,5,10 --seed "$EVAL_SEED" \
      --output "$OUT_ROOT/P3-probe"
fi

if [[ "$PHASE" == all || "$PHASE" == P4 ]]; then
  if [[ ! -f "$OUT_ROOT/P1-blocked/rollouts_temporal_v5.npz" ]]; then
    echo "[ERROR] P4 requires completed P1 ODE rollouts" >&2
    exit 2
  fi
  run_once "P4 finite signature/context kernel MMD" \
    "$OUT_ROOT/P4-signature/signature_mmd_v5.json" P4-signature.log \
    "$PYTHON_BIN" "$CODE" signature-mmd \
      --rollouts "$OUT_ROOT/P1-blocked/rollouts_temporal_v5.npz" \
      --mode future10 --output "$OUT_ROOT/P4-signature"
  if [[ -f "$OUT_ROOT/P0-data-check/data_order_preflight_v5.json" && \
        -f "$SYN_ROOT/recovery/ar2_recovery_v5.json" && \
        -f "$OUT_ROOT/P3-probe/future_probe_v5.json" ]]; then
    run_once "Appendix-ready six-panel summary (P0--P4)" \
      "$OUT_ROOT/figures/figure_manifest_v5.json" P4-figures.log \
      "$PYTHON_BIN" "$SCRIPT_DIR/plot_sap_temporal_v5.py" \
        --root "$OUT_ROOT" --output "$OUT_ROOT/figures"
  else
    echo "[INFO] Figures will be made after completing P0, P2, P3 and P4 in the same OUT_ROOT"
  fi
fi

echo "[DONE] Requested V5 phase(s) completed; all new artifacts: $OUT_ROOT"

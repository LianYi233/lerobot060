#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash examples/training/launch_pi05_revision_4gpu.sh [others|all] [SEED]

Default 'others' launches three 4,000-update controls in the background:
  GPU 1: dual_prompt_only  (direct dual, CPC off)
  GPU 2: full_reference    (750 SAP + 250 bridge + 3000 flow, CPC on)
  GPU 3: action_flow_warmup (750 full-mask action flow + 250 bridge + 3000 flow)

'all' additionally launches direct_dual (CPC on) on GPU 0. Use 'others'
if GPU 0 is already training. Each GPU runs one independent process.
DRY_RUN=true validates paths and prints exact commands without launching.
The script refuses to launch any jobs if an output directory already exists.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi

MODE="${1:-others}"
SEED="${2:-0}"
if (( $# > 2 )) || [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  usage >&2
  exit 2
fi

case "$MODE" in
  others)
    GPUS=(1 2 3)
    VARIANTS=(dual_prompt_only full_reference action_flow_warmup)
    ;;
  all)
    GPUS=(0 1 2 3)
    VARIANTS=(direct_dual dual_prompt_only full_reference action_flow_warmup)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_LAUNCHER="$SCRIPT_DIR/train_pi05_revision_4000.sh"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000}"
export LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/wyn/logs/revision-4000}"

# Validate all three/four runs before launching even one of them. In particular,
# do not inadvertently relaunch an existing run or overwrite its checkpoints.
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  echo "Checking GPU $gpu: $variant seed=$SEED"
  GPU_IDS="$gpu" NUM_PROCESSES=1 DRY_RUN=true \
    bash "$TRAIN_LAUNCHER" "$variant" "$SEED" > /dev/null
done

if [[ "${DRY_RUN:-false}" == true ]]; then
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    variant="${VARIANTS[$i]}"
    echo "---- GPU $gpu: $variant ----"
    GPU_IDS="$gpu" NUM_PROCESSES=1 DRY_RUN=true \
      bash "$TRAIN_LAUNCHER" "$variant" "$SEED"
  done
  exit 0
fi

mkdir -p "$LOG_ROOT"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  launch_log="$LOG_ROOT/launch-${variant}-seed${SEED}.stdout.log"
  # Distinct compilation caches prevent unnecessary contention among GPUs.
  nohup env \
    "GPU_IDS=$gpu" "NUM_PROCESSES=1" "DRY_RUN=false" \
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/wyn/cache/torchinductor-gpu${gpu}}" \
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/autodl-tmp/wyn/cache/triton-gpu${gpu}}" \
    bash "$TRAIN_LAUNCHER" "$variant" "$SEED" \
    > "$launch_log" 2>&1 < /dev/null &
  echo "Started GPU $gpu: $variant pid=$! startup_log=$launch_log"
done

echo "Each job also writes its training log under $LOG_ROOT/pi05-revision-<variant>-seed${SEED}-total4000.log"

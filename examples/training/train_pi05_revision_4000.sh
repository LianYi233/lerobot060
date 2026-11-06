#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiment: compare dual-prompt adaptation under exactly 4,000
# cumulative optimizer updates. Keep the original prompt-ablation launcher and
# its optimizer/model/data settings unchanged.
usage() {
  cat <<'EOF'
Usage: bash examples/training/train_pi05_revision_4000.sh VARIANT [SEED] [extra lerobot-train args...]

Variants (each uses exactly 4,000 cumulative optimizer updates):
  direct_dual       0 SAP + 0 bridge + 4000 flow, dual prompts, CPC on
  dual_prompt_only  0 SAP + 0 bridge + 4000 flow, dual prompts, CPC off
  full_reference    750 SAP + 250 bridge + 3000 flow, dual prompts, CPC on

The formal flow checkpoints are 004000 for direct_dual/dual_prompt_only,
and 003000 for full_reference (both represent 4,000 total updates).
The default checkpoint/log roots are isolated under revision-4000.

Override DATASET_ROOT, PRETRAINED_PATH, TOKENIZER_PATH and other shared
environment settings as needed. DRY_RUN=true prints the underlying command;
the source launcher still checks that the three input directories exist.
EOF
}

VARIANT="${1:-}"
if [[ "$VARIANT" == --help || "$VARIANT" == -h ]]; then
  usage
  exit 0
fi
SEED="${2:-0}"
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer (provide SEED before extra args)" >&2
  exit 2
fi
if (( $# >= 2 )); then
  shift 2
else
  shift "$#"
fi

case "$VARIANT" in
  direct_dual|dual_prompt_only)
    FLOW_UPDATES=4000
    PRETRAIN_UPDATES=0
    BRIDGE_UPDATES=0
    ;;
  full_reference)
    FLOW_UPDATES=3000
    PRETRAIN_UPDATES=1000
    BRIDGE_UPDATES=250
    ;;
  *)
    echo "Unknown revision variant: $VARIANT" >&2
    usage >&2
    exit 2
    ;;
esac

# Extra draccus overrides take precedence over earlier flags. Disallow changing
# protocol-defining settings behind the matched-budget launcher's back.
for arg in "$@"; do
  case "$arg" in
    --steps|--steps=*|--seed|--seed=*|--output_dir|--output_dir=*|\
    --job_name|--job_name=*|--resume|--resume=*|\
    --policy.training_stage|--policy.training_stage=*|\
    --policy.next_action_pretrain_steps|--policy.next_action_pretrain_steps=*|\
    --policy.next_action_bridge_steps|--policy.next_action_bridge_steps=*|\
    --policy.cabo_enabled|--policy.cabo_enabled=*|\
    --policy.num_prompt_tokens|--policy.num_prompt_tokens=*|\
    --policy.num_vlm_prompt_tokens|--policy.num_vlm_prompt_tokens=*)
      echo "Protocol-defining CLI override is not allowed: $arg" >&2
      exit 2
      ;;
  esac
done

TOTAL_UPDATES=$((PRETRAIN_UPDATES + FLOW_UPDATES))
if (( TOTAL_UPDATES != 4000 )); then
  echo "Internal error: expected 4000 total updates, got $TOTAL_UPDATES" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLOW_STEPS="$FLOW_UPDATES"
export RUN_NAME="pi05-revision-${VARIANT}-seed${SEED}-total4000"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/chkpt/2601-lerobot/revision-4000}"
export LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/logs/revision-4000}"
export SAVE_FREQ=1000
# Stage snapshots are diagnostic only; disable their extra I/O for all groups.
export NTK_SAVE_STAGE_SNAPSHOTS=false

printf 'revision-4000: variant=%s seed=%s SAP=%d bridge=%d flow=%d total=%d CPC=%s\n' \
  "$VARIANT" "$SEED" "$((PRETRAIN_UPDATES - BRIDGE_UPDATES))" "$BRIDGE_UPDATES" \
  "$FLOW_UPDATES" "$TOTAL_UPDATES" \
  "$(if [[ "$VARIANT" == dual_prompt_only ]]; then echo false; else echo true; fi)"

exec bash "$SCRIPT_DIR/train_pi05_prompt_ablation.sh" "$VARIANT" "$SEED" "$@"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_eval_pi05_revision_4000.sh VARIANT [all|no10|SUITE] [SEED]

VARIANT: direct_dual, dual_prompt_only, full_reference, full_no_cpc, action_flow_warmup
         fixed_action_lr_050, fixed_action_lr_025, fixed_action_lr_010
Default suites: all four standard LIBERO suites; default seed: 0.
Direct variants evaluate flow checkpoint 004000, and staged variants evaluate
flow checkpoint 003000. Both represent 4,000 total optimizer updates.

Set CKPT_ROOT and BASE_OUTPUT to override the isolated revision-4000 roots;
GPU_ID, EPISODES_PER_TASK and the original evaluation protocol are unchanged.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi
VARIANT="${1:-}"
MODE="${2:-all}"
SEED="${3:-0}"
if (( $# > 3 )); then
  usage >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi

case "$VARIANT" in
  direct_dual|dual_prompt_only) STEP=4000 ;;
  full_reference|full_no_cpc|action_flow_warmup|fixed_action_lr_050|fixed_action_lr_025|fixed_action_lr_010) STEP=3000 ;;
  *)
    echo "Unknown revision variant: $VARIANT" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT:-/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000}}"
export BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/wyn/eval/2601-lerobot/revision-4000}"
export RUN_NAME_OVERRIDE="pi05-revision-${VARIANT}-seed${SEED}-total4000"
export EVAL_STEP="$STEP"

printf 'revision-4000 eval: variant=%s seed=%s flow_checkpoint=%06d cumulative_updates=4000\n' \
  "$VARIANT" "$SEED" "$STEP"
exec bash "$SCRIPT_DIR/run_eval_libero-full.sh" "$VARIANT" "$MODE" "$SEED"

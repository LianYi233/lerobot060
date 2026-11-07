#!/usr/bin/env bash
set -euo pipefail

# Evaluate revision-4000 checkpoints sequentially on one GPU.
# Full per-task results remain in the original evaluator's task_results/ and eval_info.json.
# Only the four LIBERO suites and Overall are printed at the end.
#
# Usage:
#   GPU_ID=3 EPISODES_PER_TASK=10 bash run_eval_pi05_revision_4000_sequential.sh [TRAIN_SEED] [VARIANT ...]
#
# With no VARIANT arguments, visit all four variants in the order below,
# skipping completed ones and resuming any partially evaluated model.
# IMPORTANT: Stop an existing evaluator for a variant before resuming it here;
# the shared .resume.lock prevents concurrent writers.

TRAIN_SEED="${1:-0}"
if (( $# > 0 )); then
    shift
fi
if [[ ! "$TRAIN_SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "Training seed must be a non-negative integer" >&2
    exit 2
fi

if (( $# > 0 )); then
    VARIANTS=("$@")
else
    VARIANTS=(direct_dual full_reference dual_prompt_only action_flow_warmup)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${BASE_OUTPUT:-/root/autodl-tmp/wyn/eval/2601-lerobot/revision-4000}"
GPU_ID="${GPU_ID:-3}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export BASE_OUTPUT GPU_ID EPISODES_PER_TASK
mkdir -p "$BASE_OUTPUT"

if [[ ! "$GPU_ID" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GPU_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES_PER_TASK must be positive" >&2
    exit 2
fi

checkpoint_step() {
    case "$1" in
        direct_dual|dual_prompt_only) echo 004000 ;;
        full_reference|action_flow_warmup) echo 003000 ;;
        *) echo "Unknown revision variant: $1" >&2; return 2 ;;
    esac
}

output_dir() {
    local variant="$1"
    local step="$2"
    echo "$BASE_OUTPUT/libero060-all-pi05-revision-${variant}-seed${TRAIN_SEED}-total4000-${step}-libero-all4-resume"
}

# A complete eval_info.json is written only after all requested suites finish.
# Verify the episode count before reusing it, rather than silently accepting
# an earlier result with a different evaluation budget.
is_complete() {
    python - "$1" "$EPISODES_PER_TASK" <<'PY'
import json
import sys

path, episodes = sys.argv[1], int(sys.argv[2])
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
try:
    with open(path) as handle:
        data = json.load(handle)
    groups = data["per_group"]
    if any(int(groups[s]["n_episodes"]) != 10 * episodes for s in suites):
        sys.exit(1)
    if int(data["overall"]["n_episodes"]) != 40 * episodes:
        sys.exit(1)
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    sys.exit(1)
PY
}

for variant in "${VARIANTS[@]}"; do
    step="$(checkpoint_step "$variant")"
    out="$(output_dir "$variant" "$step")"
    info="$out/eval_info.json"
    if [[ -f "$info" ]]; then
        if is_complete "$info"; then
            echo "[SKIP] $variant: complete result found"
            continue
        fi
        echo "[ERROR] Incomplete or mismatched existing eval_info.json for $variant; refusing to overwrite" >&2
        exit 1
    fi

    if [[ -f "$out/.resume.lock" ]] && command -v flock >/dev/null 2>&1; then
        if ! flock -n "$out/.resume.lock" -c true; then
            echo "[BUSY] $variant still has a running evaluator. Wait for GPU2 to finish or stop it before switching to GPU3." >&2
            exit 1
        fi
    fi

    console_log="$BASE_OUTPUT/sequential-${variant}-seed${TRAIN_SEED}.log"
    echo "[RUN] $variant on GPU $GPU_ID (checkpoint $step; existing tasks will resume)"
    if ! GPU_ID="$GPU_ID" EPISODES_PER_TASK="$EPISODES_PER_TASK" \
        bash "$SCRIPT_DIR/run_eval_pi05_revision_4000.sh" "$variant" all "$TRAIN_SEED" \
        > "$console_log" 2>&1; then
        echo "[FAILED] $variant. Last 40 log lines:" >&2
        tail -n 40 "$console_log" >&2
        exit 1
    fi
    if ! is_complete "$info"; then
        echo "[FAILED] $variant returned successfully but eval_info.json is incomplete" >&2
        exit 1
    fi
    echo "[DONE] $variant"
done

python - "$BASE_OUTPUT" "$TRAIN_SEED" <<'PY'
import json
import sys
from pathlib import Path

root, seed = Path(sys.argv[1]), sys.argv[2]
steps = {
    "direct_dual": "004000",
    "dual_prompt_only": "004000",
    "full_reference": "003000",
    "action_flow_warmup": "003000",
}
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
header = ("Model", "Spatial", "Object", "Goal", "LIBERO-10", "Overall")
print("\n" + "  ".join(f"{x:>13}" for x in header))
for variant, step in steps.items():
    info = root / (
        f"libero060-all-pi05-revision-{variant}-seed{seed}-total4000-"
        f"{step}-libero-all4-resume/eval_info.json"
    )
    values = ["--"] * 5
    if info.is_file():
        with info.open() as handle:
            data = json.load(handle)
        groups = data.get("per_group", {})
        try:
            values = [f"{float(groups[s]['pc_success']):.2f}%" for s in suites]
            values.append(f"{float(data['overall']['pc_success']):.2f}%")
        except (KeyError, TypeError, ValueError):
            values = ["--"] * 5
    print("  ".join(f"{x:>13}" for x in (variant, *values)))
PY


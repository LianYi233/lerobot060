#!/usr/bin/env bash
# Run Task2-only offline plots on one GPU. No robot connection.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# When checked into this repo, use the adjacent checkout; retain standalone support.
if [[ -n "${PIPER_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$PIPER_REPO_ROOT"
elif [[ -f "$HERE/../piper_policy_utils.py" && -d "$HERE/../src/lerobot" ]]; then
  REPO_ROOT="$(cd -- "$HERE/.." && pwd)"
else
  REPO_ROOT=/root/autodl-tmp/lerobot060-piper-partial-ft
fi
GPU_ID="${GPU_IDS:-0}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_IDS must be a single GPU index, for example 0." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -u "$HERE/plot_task2_teleoperation_vs_model.py" \
  --repo-root "$REPO_ROOT" "$@"

#!/usr/bin/env bash
# One A800, two independent runs per wave: task 2/3, then task 4/1.
set -euo pipefail
PIPER_REVISION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${PIPER_REVISION_DIR}/train_piper_revision.py" \
  --preset reality-long --parallel-tasks 2 --run-group reality-parallel-v1 \
  --stagger-seconds 180 --min-free-gib 30 "$@"

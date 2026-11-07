#!/usr/bin/env bash
# One GPU, sequential reviewer controls. Use the active LeRobot Python environment.
set -euo pipefail
PIPER_REVISION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${PIPER_REVISION_DIR}/train_piper_revision.py" "$@"

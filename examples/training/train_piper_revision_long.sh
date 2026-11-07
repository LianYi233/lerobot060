#!/usr/bin/env bash
# One independent PrimingVLA model per reality task: 2 -> 3 -> 4 -> 1.
# Fixed default: 50,000 cumulative updates. CLI options override this preset.
set -euo pipefail
PIPER_REVISION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${PIPER_REVISION_DIR}/train_piper_revision.py" --preset reality-long "$@"

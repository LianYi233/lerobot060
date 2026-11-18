#!/usr/bin/env bash
# Offline redraw only: reads existing paired measurements, needs no GPU/checkpoints.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m lerobot.scripts.plot_pi05_ntk_comparison "$@"

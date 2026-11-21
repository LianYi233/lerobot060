#!/usr/bin/env bash
# Compare the same four cumulative checkpoints in two integrated training runs.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash $0 FULL_RUN NO_SAP_MATCHED_RUN [GPU_ID]" >&2
  exit 2
fi
a="$1"
b="$2"
gpu="${3:-0}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${here}/../../src${PYTHONPATH:+:${PYTHONPATH}}"

# Both runs use the staged checkpoint layout. Do not silently compare two SAP
# runs, an uninterrupted run, or checkpoints from different training seeds.
python - "$a" "$b" <<'PY'
import json
import sys
from pathlib import Path

seeds = []
for name, base, expected in zip(("A", "B"), sys.argv[1:], (False, True), strict=True):
    run = Path(base).resolve()
    pre = run.with_name(run.name + "_next_action_pretrain")
    for directory, step in [(pre, 0), (pre, 750), (pre, 1000), (run, 3000)]:
        ckpt = directory / "checkpoints" / f"{step:06d}" / "pretrained_model"
        cfg = json.loads((ckpt / "config.json").read_text())
        train = json.loads((ckpt / "train_config.json").read_text())
        if (cfg.get("next_action_joint_flow_before_bridge", False) is not expected
                or cfg.get("cabo_enabled") is not True):
            raise SystemExit(f"{name}: wrong training variant at {ckpt}")
        seeds.append(train["seed"])
if len(set(seeds)) != 1:
    raise SystemExit(f"Expected seed-matched checkpoints, found {seeds}")
PY

a_out="$a/ntk_no_sap_matched_4000"
b_out="$b/ntk_no_sap_matched_4000"
# The numerical fingerprint includes modeling_pi05.py; use the current code
# for both measurements, without disturbing either run's original NTK results.
FINAL_FLOW_STEPS=3000 NTK_OUTPUT_DIR="$a_out" \
  bash "$here/analyze_pi05_ntk_stages.sh" "$a" "$gpu" --scope=both
FINAL_FLOW_STEPS=3000 NTK_OUTPUT_DIR="$b_out" \
  bash "$here/analyze_pi05_ntk_stages.sh" "$b" "$gpu" --reference-results="$a_out/results.json" \
    --reference-preprocessor="${a%/}_next_action_pretrain/checkpoints/000000/pretrained_model"
for scope in backbone prompts; do
  bash "$here/replot_pi05_ntk_comparison.sh" \
    --staged-results="$a_out/results.json" --control-results="$b_out/results.json" \
    --staged-label="PrimingVLA" --control-label="PrimingVLA w/o SAP" \
    --scope="$scope" --output-dir="$b_out/comparison_$scope"
done

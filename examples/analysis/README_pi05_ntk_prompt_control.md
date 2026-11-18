# NTK control: uninterrupted prompt learning

Compare the existing three-stage run with ordinary dual-prompt learning at **0, 750,
1000 and 4000 cumulative optimizer updates**. The control trains both prompt banks
together with one uninterrupted flow objective, optimizer and scheduler. Its 750
and 1000 snapshots are checkpoints, not SAP/Bridge transitions.

The default `dual_prompt_only` control disables CABO/CPC as well as the stages.
For an additional ablation that retains CABO/CPC and removes only staging, run
`direct_dual`. Do not attribute every difference between the default control and
`full_reference` solely to staging.

## 1. Train and save the four exact checkpoints

Run from the repository root in the existing PI05 training environment:

```bash
git pull origin prompt-ablation

STAGED_RUN=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-seed0
PRETRAINED_PATH="${STAGED_RUN}_next_action_pretrain/checkpoints/000000/pretrained_model" \
  bash examples/training/train_pi05_prompt_ntk_control.sh dual_prompt_only 0
```

Using the original initialization checkpoint preserves its actual initialized
prompt tensors, rather than relying only on a matching seed. Keep dataset,
batch size, prompt counts, optimizer settings and base weights matched to the
reference run. The wrapper defaults to 16 VLM + 16 action prompts, batch size 32,
seed 0 and GPU 0; set `GPU_IDS`, `DATASET_ROOT`, `TOKENIZER_PATH`, `BATCH_SIZE`,
`OUTPUT_ROOT` or `RUN_NAME` as needed. Existing checkpoints must not be overwritten.

Default control output:

```text
/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-dual_prompt_only-ntk-control-seed0/
  checkpoints/000000/pretrained_model/
  checkpoints/000750/pretrained_model/
  checkpoints/001000/pretrained_model/
  checkpoints/004000/pretrained_model/
```

The training launcher does **not** run NTK analysis automatically. It saves models,
processors and training states at the requested steps. Snapshotting does not reset
the optimizer or change the training objective. `DRY_RUN=true` prints the training
command. To resume an interrupted run, use the saved training configuration with
the standard `lerobot-train --config_path=... --resume=true` interface.

For the CPC-preserving control, replace `dual_prompt_only` with `direct_dual`; it
uses a separate output directory by default.

## 2. Measure the control and create both scatter layouts

```bash
CONTROL_RUN=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-dual_prompt_only-ntk-control-seed0
STAGED_RESULTS="${STAGED_RUN}/ntk_stages/results.json"  # adjust to the actual existing file

bash examples/analysis/analyze_pi05_ntk_prompt_control.sh \
  "$CONTROL_RUN" "$STAGED_RESULTS" 0
```

The reference `results.json` supplies the exact sample indices, probe seeds,
output probes, CountSketch settings and measured parameter scope. A relocated
dataset or tokenizer may be supplied with `DATASET_ROOT` / `TOKENIZER_PATH`.
Analysis verifies the **processed sample fingerprint**, model/probe configuration,
parameter signatures, PyTorch version, device family and numerical measurement
source before accepting a paired comparison. Incompatible measurements fail
explicitly instead of being silently combined.

The numerical NTK functions are unchanged. A recorded measurement digest now
separates that numerical code from CLI/layout changes. The original paper results
from the explicitly audited source revision are supported; unknown legacy
revisions require remeasurement with the same analyzer.

Outputs under `$CONTROL_RUN/ntk_prompt_control/`:

- `manifest.json`, per-seed JSON files and `results.json`: real measured values and provenance.
- `scatter/ntk_prompt_comparison.{png,pdf,svg}`: control-only row, four checkpoints.
- `comparison/ntk_prompt_comparison.{png,pdf,svg}`: staged row above control row, four columns.
- Each plot directory also contains `ntk_prompt_comparison_points.csv` and
  `ntk_prompt_comparison_metadata.json` for exact plotted values and source fingerprints.

The x-axis is spectral effective rank; the y-axis is parameter-normalized tangent
energy (logarithmic by default). All panels share axis limits. Each point is one
actual saved probe-seed measurement, with no jitter, interpolation or simulated
values. The ten probe seeds describe measurement variability within a training
run; they are **not ten independent training runs**. Means are optional in the
offline plotter. A changing NTK distribution characterizes output sensitivity,
and should be interpreted alongside task success rather than as proof of better performance.

While training is still running, plot the first three checkpoints:

```bash
THROUGH_STEP=1000 bash examples/analysis/analyze_pi05_ntk_prompt_control.sh \
  "$CONTROL_RUN" "$STAGED_RESULTS" 0
```

Run again without `THROUGH_STEP` after training finishes. Matching cached
measurements are reused and the final checkpoint is appended. Changing samples,
weights or probe settings requires a new output directory (`NTK_OUTPUT_DIR`).
Use `--dry-run` to validate checkpoint metadata without loading a model.

## 3. Replot without models or a GPU

```bash
bash examples/analysis/replot_pi05_ntk_comparison.sh \
  --control-results="$CONTROL_RUN/ntk_prompt_control/results.json" \
  --staged-results="$STAGED_RESULTS" \
  --output-dir="$CONTROL_RUN/ntk_prompt_control/comparison"
```

Use the plotter's `--steps=0,750,1000` option for a partial comparison and
`--scope=prompts` for the prompt-parameter NTK rather than the backbone NTK.
Font selection prefers Times New Roman; provide `--font-path=/path/to/times.ttf`
for an explicit font file (or set `NTK_FONT_PATH` in the analysis wrapper). If it
is unavailable, a serif fallback is reported. For manual `direct_dual` replotting,
add `--control-label="Prompt learning + CPC (no stages)"`; the analyzer already
selects this label automatically.
The analyzer wrapper also accepts `--plot-only` to regenerate its two default
layouts. Standalone analysis without a staged reference accepts `-` in place of
`STAGED_RESULTS_JSON`; paired measurement checks apply when a reference is provided.

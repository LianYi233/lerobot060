# VLM-only training with fixed-probe effective-rank monitoring

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
OUTPUT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
LOG_ROOT=/root/autodl-tmp/logs/vlm-only \
GPU_IDS=0,1 BATCH_SIZE=32 \
VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" \
bash examples/training/train_pi05_vlm_only_rank_sweep.sh 1 \
  --wandb.enable=true --wandb.project=prompt-learning \
  --wandb.mode=online --wandb.disable_artifact=true
```

This sequentially trains `pi05-base-vlm{1,2,4,8,16,32}-seed1` from the same base
checkpoint. Both GPUs train each model with batch 32 **per GPU** (global batch 64).
The existing base QUANTILES processors, learning-rate schedule and frozen-backbone
VLM-only setup are retained. Input models/tokenizer/dataset default to
`/root/autodl-fs/wyn/`; outputs default to `/root/autodl-tmp/`.
Only the final **full** checkpoint, `checkpoints/003000`, is saved. The diagnostic
also saves tiny prompt-only snapshots; it never saves intermediate backbone weights.
`torch.compile` is disabled. Existing run directories cause an error before training.
To restart a partial sweep, explicitly set the remaining `VLM_PROMPT_TOKEN_COUNTS`.

## What is being measured

For fixed observations `x_i`, Gaussian noise `epsilon_i`, and normalized recorded
actions `a_i`, form `z_it = t*epsilon_i + (1-t)*a_i`. Let `J_it` be the Jacobian of
the fully observation-conditioned flow **velocity** with respect to all learned
VLM prompt coordinates. Keep only real action channels and non-padded timesteps.

For `R` fixed, shared Rademacher output vectors `u_r`, stack

`G[(i,t,r),:] = u_r^T J_it / sqrt(R)`.

Compute `K = G G^T` on CPU in float64. There is **no parameter-axis sketch**.
Let `lambda` be the nonnegative eigenvalues of K and `sigma = sqrt(lambda)`:

| Metric | Definition |
|---|---|
| `jacobian_erank` | `exp(-sum(p*log(p)))`, `p = sigma/sum(sigma)` |
| `ntk_erank` | Same entropy, but `p = lambda/sum(lambda)` |
| `participation_rank` | `(sum(lambda))^2 / sum(lambda^2)` |
| `stable_rank` | `sum(lambda)/max(lambda)` |
| `rank90_energy`, `rank95_energy` | Minimum leading components covering 90%/95% of squared singular-value energy |
| `tangent_trace` | `sum(lambda)`; projected total tangent energy, not normalized by sample count |
| `energy_per_parameter` | `tangent_trace / trainable_parameter_count` |
| `rank_ceiling` | `min(number_of_projected_rows, trainable_parameter_count)` |

The zero Jacobian has zero ranks and energy. Near-null eigenvalues at float64
roundoff are discarded before taking square roots. The predictor retains training
precision (normally bf16 with accelerate autocast); this is not a full-FP32 study.

**Interpretation:** these are spectra of a fixed **output-projected** Jacobian,
not the exact full-model Jacobian, loss-gradient covariance, prompt embedding rank,
attention rank, or the Jacobian through full denoising/closed-loop rollouts.
A one-token prompt has 2048 coordinates; its measured Jacobian rank can exceed 1.
The current diagnostic does not establish whether target action corrections align
with the measured tangent subspace.

## Fixed protocol and controls

Defaults: step **0,250,500,...,3000** (13 measurements), 16 observations,
times `[0.2,0.5,0.8]`, 4 output probes, probe seed 2026. The row ceiling is **192**.
Observation indices are sampled once, one random frame per dataset stratum;
this is not guaranteed to cover all 40 tasks. Fewer dataset rows reduce the ceiling.

The probe seed is independent of training seed and prompt count. All six runs use
the same observations, disabled image augmentation, noise, timesteps and projections.
Processed observations are cached once on CPU, with sample identities and a hash.
Training processors and RNG streams are preserved. For cross-run comparison, verify
`sample_sha256`, action dimensions, time settings, precision and source/normalization
in the manifests. A raw base checkpoint with different processors is not a matched
zero-token performance control.

Each rank probes a subset of samples. A temporary prompt leaf is used outside the
DDP reducer and optimizer; the original parameter object, gradients and individual
module training flags are restored even after a Python exception. Samples are
gathered in a world-size-independent order. Ranks synchronize after each sample/time
forward and its VJPs, rather than waiting for one long rank-0-only analysis.
FSDP/DeepSpeed, compile, LoRA and non-VLM-only training are rejected for this diagnostic.

The monitor records time cost. If it is too slow, increase `RANK_FREQ=500` first.
Changing sample/projection counts changes the estimator, so use the same settings
throughout a comparison. If ranks approach the ceiling, confirm on a larger common
probe (e.g. `RANK_OUTPUT_PROBES=8`) before interpreting saturation as model capacity.

## Outputs

Within each model's training output directory:

```text
prompt_rank/
  manifest.json       # definitions, settings, sample identities, hash, provenance
  fixed_probe.pt      # processed observations, fixed Gaussian noise/output probes
  metrics.csv         # one row per measured completed step
  metrics.jsonl       # the same scalar records
  step_000000.npz     # small prompt-only snapshot + Gram/eigenvalues/singular values
  ...
  step_003000.npz
checkpoints/003000/pretrained_model/
```

W&B metrics are under `eval/prompt_rank/`, especially `jacobian_erank`, `ntk_erank`,
`rank95_energy`, `rank_ceiling` and `probe_seconds`, plotted against training step (`_step`).
The local CSV/JSONL remains available if W&B is disabled.

For an individual run, pass `--policy.compile_model=false --prompt_rank.freq=250`
and optional `--prompt_rank.samples`, `--prompt_rank.output_probes`,
`--prompt_rank.timesteps`, `--prompt_rank.seed` to `train_pi05_vlm_only.sh`.
The general token sweep accepts the same flags; monitoring defaults to disabled
outside the dedicated rank sweep.

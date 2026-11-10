# External PI0.5 + LoRA baseline: same initialization, 4,000 total updates

`lora_r16` is a **standard, prompt-free LoRA implementation**, not PrimingVLA
with a LoRA label. It uses the original pretrained PI0.5 weights and the same
dataset/flow-matching implementation as the revision controls. It does not use
SAP, conditional preparation/bridge updates, CPC/CABO, or learned prompts.

## What is trained

| Setting | `lora_r16` |
| --- | --- |
| Objective / budget | Ordinary observation-conditioned action flow, 4,000 optimizer updates |
| LoRA | Rank 16, alpha 16, dropout 0, bias `none`, standard zero-delta initialization |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in **both** the language transformer and action expert |
| Frozen weights | Original pretrained weights, vision encoder, embeddings, vocabulary head, multimodal/action/time projections, normalization layers |
| Learned prompts | None; both token counts are zero |
| CPC / fixed prompt LR / SAP / bridge | All disabled |
| Optimizer | Shared PI05 AdamW defaults: peak LR `2.5e-5`, betas `(0.9, 0.95)`, weight decay `0.01`, no global clipping |
| Scheduler | Shared cosine recipe, scaled to the 4,000-update run; decay LR `1e-5` |
| Default effective batch | `32 × 1 process = 32` |
| Checkpoints | Compact adapters at 1,000 / 2,000 / 3,000 / 4,000 updates |
| Primary evaluation | Fixed final checkpoint `004000`, not a test-selected best checkpoint |

The exact module-path regex deliberately excludes similarly named vision
attention layers. Both VLM and expert adapter gradients are enabled: the
prompt-only detached-observation optimization is disabled in this mode.
Original prompt adaptation remains the default for all old runs/checkpoints.

For the default `gemma_2b`/`gemma_300m` stacks, rank 16 gives approximately
**26.54M** trainable adapter parameters (252 adapted linear modules), versus
approximately 0.05M for the dual prompts. Use the actual startup report and
`<run>/trainable_parameters.json` for the paper; do not call this an equal-
parameter-budget comparison. Some final VLM outputs are not consumed by the
action loss, so not every targeted final-layer tensor has a gradient on every
step. Existing DDP enables `find_unused_parameters=True`.

## 1. Verify the original initialization and environment

Use the same activated LeRobot environment/check-out as the existing revision
experiments. Native PEFT is required (`peft>=0.18.0,<1.0.0`, the repository's
`peft` optional dependency). Do not replace a working PyTorch/Transformers
environment blindly just to install one missing package.

```bash
uv run --active --no-sync python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)'
```

The shared launcher defaults are:

```bash
export PRETRAINED_PATH=/root/autodl-fs/ckpt/pi05_libero_base
export DATASET_ROOT=/root/autodl-fs/datasets/libero
export TOKENIZER_PATH=/root/autodl-fs/ckpt/google/paligemma-3b-pt-224
export OUTPUT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000
export LOG_ROOT=/root/autodl-tmp/wyn/logs/revision-4000
export BATCH_SIZE=32
export NUM_PROCESSES=1
export DTYPE=float32
export MIXED_PRECISION=no
```

**These defaults do not certify what an earlier server run actually used.**
If the existing full reference started from `pi05_base`, set the same
`PRETRAINED_PATH` here. Never compare `pi05_base` and `pi05_libero_base` as if
they were the same initialization. With `pi05_libero_base`, describe the
experiment as further adaptation from a LIBERO-adapted base.

Do **not** use `full_reference`'s primed Stage-1 checkpoint as the LoRA base.
Integrated training rewrites the final flow config's `pretrained_path` to that
transfer checkpoint. Recover the original initialization from its Stage-1
`train_config.json` or original launch log.

Recommended read-only cross-check, using the actual completed reference:

```bash
export REFERENCE_TRAIN_CONFIG="$OUTPUT_ROOT/pi05-revision-full_reference-seed0-total4000/checkpoints/003000/pretrained_model/train_config.json"
GPU_IDS=0 DRY_RUN=true bash examples/training/train_pi05_revision_4000.sh lora_r16 0
```

When `REFERENCE_TRAIN_CONFIG` is supplied, the preflight follows the saved
Stage-1 configuration back to the original base and checks local paths, dataset
selection, total updates, effective batch, dtype, and key chunk settings. It
fails on mismatches rather than changing either experiment. Keep the Stage-1
configuration even when archiving large intermediate weights. If an older
checkpoint lacks `training_state/training_step.json` process metadata, set
`REFERENCE_NUM_PROCESSES` **only after verifying the original process count**.

Without a reference config, the launcher explicitly reports that historical
run parity has not been verified. This preflight compares recorded settings,
not historical file content hashes, AMP settings, every preprocessing option,
or exact sample order. Check `MIXED_PRECISION`, augmentation/normalization,
software version, and original asset immutability separately. Dataset path
equality is not evidence that the data at that path never changed.

## 2. Train fresh independent seeds

After the dry run and dependency checks pass:

```bash
GPU_IDS=0 bash examples/training/train_pi05_revision_4000.sh lora_r16 0
```

The command runs in the foreground and logs to:

```text
/root/autodl-tmp/wyn/logs/revision-4000/pi05-revision-lora_r16-seed0-total4000.log
```

For three independent training seeds, use three terminals (or tmux sessions):

```bash
# Terminal 1
GPU_IDS=0 NUM_PROCESSES=1 bash examples/training/train_pi05_revision_4000.sh lora_r16 0

# Terminal 2
GPU_IDS=1 NUM_PROCESSES=1 bash examples/training/train_pi05_revision_4000.sh lora_r16 1

# Terminal 3
GPU_IDS=2 NUM_PROCESSES=1 bash examples/training/train_pi05_revision_4000.sh lora_r16 2
```

Run one seed first to check memory and parameter counts. Do not reduce only
LoRA's batch size after OOM and keep calling its data budget matched. Gradient
checkpointing is already enabled by the shared launcher. Three independent
single-GPU jobs are **not** a single `NUM_PROCESSES=3` job: the latter triples
the effective batch at the same per-process batch size. FSDP is not supported
by this baseline implementation; single-device and DDP are supported paths.

The startup audit lists every trainable tensor; `vlm_prompt`, `action_prompt`
and `other` must all be zero, while both `lora_vlm` and `lora_action` are
positive. It also records base path, seed, effective batch, dtype, and mixed
precision in `trainable_parameters.json`.

### Learning-rate selection

The common default LR is a controlled starting point, **not a claim that LoRA
has been optimally tuned**. A competitive-baseline study should allow a small,
predeclared validation-only sweep with a comparable tuning budget. Do not pick
the LR or checkpoint using the final LIBERO test success rates.

Use policy preset fields, not `--optimizer.lr` (the policy preset overwrites
that field). For example, a separate candidate with peak `5e-5` and decay
`2e-5`, still 4,000 total updates:

```bash
OUTPUT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000-lora-lr5e5 \
LOG_ROOT=/root/autodl-tmp/wyn/logs/revision-4000-lora-lr5e5 \
GPU_IDS=0 DRY_RUN=true \
bash examples/training/train_pi05_revision_4000.sh lora_r16 0 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_decay_lr=2e-5
```

Use a distinct `OUTPUT_ROOT` **and** `LOG_ROOT` for each candidate; set the
corresponding `CKPT_ROOT` and a distinct `BASE_OUTPUT` for evaluation. The
matched launcher prevents hidden rank, target, prompt, CPC, stage, data or
batch CLI overrides. Shared environment settings remain explicit.

## 3. Evaluate the unmerged adapter with the same protocol

The final adapter is:

```text
<OUTPUT_ROOT>/pi05-revision-lora_r16-seed0-total4000/checkpoints/004000/pretrained_model
```

It contains `adapter_model.safetensors`, `adapter_config.json`, policy and
training configs, plus the saved processing/normalization state. It does not
need a copied full `model.safetensors` at every checkpoint. **Keep the original
base checkpoint and local tokenizer**: native loading combines base + adapter.

```bash
export CKPT_ROOT=/root/autodl-tmp/wyn/chkpt/2601-lerobot/revision-4000
export BASE_OUTPUT=/root/autodl-tmp/wyn/eval/2601-lerobot/revision-4000
export EPISODES_PER_TASK=10

GPU_ID=0 bash run_eval_pi05_revision_4000.sh lora_r16 all 0
```

Three terminals may evaluate different training seeds on GPUs 0/1/2 with
trailing arguments `0`, `1`, `2`, respectively. The trailing seed selects the
**trained model**; it does not change the shared evaluation seed (1000).
The wrapper preserves the existing four suites, episode counts, reset states,
relative controls, ten executed actions per chunk, and inference settings.
If parallel simulation saturates the CPU, evaluate one job per GPU sequentially
or reduce concurrent jobs; do not change the rollout protocol to hide the load.

`run_eval_pi05_revision_4000_sequential.sh 0 lora_r16` is also supported.
Its existing default variant list is unchanged.

The LoRA evaluator uses offline local loading. The saved processor contains
the tokenizer's local path; moving the adapter to another server does not
automatically rewrite that path or the base path in `adapter_config.json`.
The per-task resume manifest adds base file sizes/modification timestamps,
not cryptographic hashes. The sequential helper's pre-existing completed-result
fast path checks episode counts only. Keep base/data/tokenizer immutable; if
they change, use a fresh `BASE_OUTPUT` rather than reusing completed results.

## 4. Interpretation and verification

Report the mean and variability across independent **training seeds**, the four
per-suite success rates and overall macro average, actual trainable parameter
counts, and training time/peak memory. This controls total optimizer updates
and (with the same effective batch) nominal sample presentations. It does not
equate GPU-hours, number of trainable parameters, phase-specific sample order,
or the staged method's optimizer resets/LR schedule.

CPU-only launcher/load-contract regression tests (no training or downloads):

```bash
uv run --active --no-sync python -m unittest discover -s tests/scripts -p 'test_pi05_lora_*.py' -v
uv run --active --no-sync python -m unittest discover -s tests/scripts -p 'test_pi05_revision_4000_launcher.py' -v
```

Real tiny-model CPU forward/backward, frozen-base, cached inference and native
adapter round-trip tests, in an environment with the model/test dependencies:

```bash
uv run --active --no-sync pytest tests/policies/pi0_pi05/test_pi05_lora_baseline.py -q
uv run --active --no-sync pytest tests/policies/pi0_pi05/test_pi05_prompt_peft_loading.py -q
```

The implementation environment used for this change had no PyTorch,
Transformers, PEFT or pytest. Dependency-free checks were executed; the tiny
real-model tests and full-size GPU smoke test must still be run in the training
environment before treating the baseline as experimentally validated.

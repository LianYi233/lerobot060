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

## 训练完成后的评估与分析（seed1）

这部分复用已有模型、`fixed_probe.pt` 和 prompt 快照，不需要重新训练。
默认分析 base 的 1/2/4/8/16/32 token，训练 seed1；LIBERO 评估 seed 仍为 1000，
每个任务 10 episodes、每个模型共 400 episodes。训练 seed 与评估 seed 不要混淆。

先仅重画已有数据（CPU，不加载模型）：

```bash
bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh plots 1
```

完整流程：六个模型依次评估 → 离线误差对齐 → 汇总和绘图。

```bash
env -u PRETRAINED_PATH -u RUN_NAME -u VLM_PROMPT_TOKENS \
    -u BASE_CKPT -u BASE_MODEL_PATH \
CKPT_ROOT=/root/autodl-tmp/chkpt/2601-lerobot/vlm-only \
BASE_OUTPUT=/root/autodl-tmp/eval/vlm-only \
REPORT_ROOT=/root/autodl-tmp/analysis/vlm-only \
VLM_PROMPT_TOKEN_COUNTS="1 2 4 8 16 32" \
GPU_ID=1 \
bash examples/analysis/analyze_pi05_vlm_rank_sweep.sh all 1
```

可以将 `all` 分别换成 `eval`、`align`、`report` 分阶段执行。`report` 要求所有所选模型
已有完整评估和对齐记录，缺失时会报错。`plots` 只使用训练时保存的谱数据。
评估复用现有的单行进度条和按任务续测；对齐按模型/step 续算。模型或分析协议变化时，
选择新的输出位置，而不是混用旧结果。设置 `DRY_RUN=true` 可只检查调度命令。

对齐默认在单张 GPU 上分析 step **0、1500、3000**。每个模型只加载一次最终 checkpoint，
再恢复各 step 的小型 prompt 快照。该流程要求 backbone 一直冻结；会检查 VLM-only 配置、
训练 seed、checkpoint step、最终 prompt 和缓存哈希。重算的 Gram 与原记录的相对 Frobenius
差异默认不得超过 2%，用于发现加载、精度或探针不一致；超过阈值应排查原因。
观测和语言 token 已保存在缓存中，因此不再读取数据集或加载 tokenizer。

### 误差对齐的定义与解释

在原来的固定观测、噪声和时间点上，flow matching 的目标为 `u = noise - action`，
当前残差为 `e = u - velocity`。保留原输出投影 `S`，得到 `G = S J` 和 `b = S e`。
若 `G G^T` 的前 k 个正特征值对应向量为 `U_k`，则：

`top_k_alignment = ||U_k^T b||^2 / ||b||^2`。

它测量**投影残差**与主导切空间的重合程度。图中的 `k/M` 是各向同性方向的参考值，
不是经统计检验得到的随机基线。满投影空间可能几乎总能覆盖残差，因此不要只看全秩覆盖率。
`alignment_at_rank95` 则使用覆盖 95% 切向能量的方向数；有效秩与这个方向数不是同一个量。
零残差时对齐比例无定义，保存为 null；不够 k 个正特征方向时对应 top-k 也保存为 null。

另生成独立 Rademacher 输出投影 `T`（默认 4 个，seed2027），在**同一批观测、噪声和时间点**
上得到 `H = T J`、`c = T e`。在原投影中拟合 ridge 线性修正：

`alpha = (G G^T + lambda I)^-1 b`, `delta = G^T alpha`,
`lambda = RIDGE_RELATIVE * trace(G G^T) / M`，默认 `RIDGE_RELATIVE=0.01`。

记录 `heldout_linear_reduction = 1 - ||c - H delta||^2 / ||c||^2`：

- 接近 1：该线性修正在独立输出投影上也能明显降低残差。
- 接近 0：改善有限。
- 小于 0：在独立投影上反而增加误差，原投影中的拟合可能过于乐观。

**没有把 delta 写入模型，也没有做额外训练。** 这仍是局部线性预测，不等于实际非线性更新后的
loss 或 rollout 成功率；`linear_delta_relative_norm` 很大时尤其要谨慎。
这里的 held-out 仅指新的输出投影，不能声称新观测、新任务或闭环控制的泛化。
它补充了谱几何证据，尚不构成 prompt 提升成功率的因果证明。

对齐输出保存于每个 run 的 `prompt_alignment/`，包括协议、每个 step 的 JSON 指标和小型 NPZ。
NPZ 保存训练/独立投影的 Gram、交叉 Gram 及残差，可离线调整 ridge 分析；不保存完整 Jacobian。
更改 `HELDOUT_PROBES`、`HELDOUT_SEED` 或 `RIDGE_RELATIVE` 时，需对所有 token 使用相同设置，
并通过 `ALIGNMENT_NAME` 选择新目录。

### 报告内容

默认完整报告目录：
`/root/autodl-tmp/analysis/vlm-only/pi05-base-seed1/complete/`。
仅画谱的 `plots` 模式使用同级 `spectra/` 子目录。

| 文件 | 内容 |
|---|---|
| `rank_dynamics.png/pdf` | Jacobian/NTK 有效秩、95% 能量方向数、总切向能量随训练变化 |
| `spectra_and_energy.png/pdf` | 初始/中期/最终的归一化奇异值谱和累计平方能量 |
| `residual_alignment.png/pdf` | 最终 top-k 残差覆盖与独立投影线性改善随训练变化 |
| `success_vs_geometry.png/pdf` | 最终成功率与几何/对齐指标的散点图，数字标记 token 数 |
| `final_summary.csv` | 最终指标、初始指标及其变化、四个 suite/overall 的成功率与分子分母 |
| `history.csv`, `spectra.csv` | 全部训练曲线与作图谱数据 |
| `alignment_summary.csv` | 所选 step 的误差对齐结果 |
| `correlations.csv` | 各 suite/overall 与指标的 Pearson/Spearman 描述性相关，常量指标返回空值 |
| `report.json` | 协议、汇总结果与解释限制 |

比较前会校验固定探针协议、模型来源/归一化与精度；评估必须包含完整的 40 个任务，且
checkpoint、评估 seed、episodes、action chunk 执行长度及评估代码版本匹配。
当前六个点只有一个训练 seed，相关性不代表因果，不能据此选择普遍最优 token 数。
后续建议补多个训练 seed、扩大共同探针，并用独立观测验证误差修正方向。
默认字体为 DejaVu Serif；设置 `FONT_PATH=/absolute/path/times.ttf` 可使用 Times New Roman。

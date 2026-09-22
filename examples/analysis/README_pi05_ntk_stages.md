# PI05 四时间点 NTK 分析

比较 **VLM 与 action expert** 在训练前、750 步 priming 后、1000 步 bridge（论文 stage 2）后、最终训练后的局部敏感性。
`backbone` 与 `prompts` 分开绘图，默认两者都测。这里 action backbone 包含 Gemma expert、action 输入/输出投影和 time MLP；不包含只用于 inpainting 的 role embedding。VLM 包含完整 PaliGemma，包括视觉编码器。参数按对象去重，未参与输出的参数贡献零梯度，但仍计入该组参数总数。

## 1. 先区分累计步数与 flow 步数

当前 `train_pi05_prompt_ablation.sh full_reference` 的默认配方是 **750 + 250 + 3000 = 4000** 次更新。
前 1000 次更新位于 `RUN_DIR_next_action_pretrain`，后续 flow 在 `RUN_DIR` 中从 0 重新计数。

| 时间点 | 累计更新数 | checkpoint |
| --- | ---: | --- |
| 训练前（包含本次训练实际初始化的 dual prompts） | 0 | `RUN_DIR_next_action_pretrain/checkpoints/000000/pretrained_model` |
| Priming 完成，尚未执行 bridge 更新 | 750 | `RUN_DIR_next_action_pretrain/checkpoints/000750/pretrained_model` |
| Bridge / 论文 stage 2 完成 | 1000 | `RUN_DIR_next_action_pretrain/checkpoints/001000/pretrained_model` |
| 整个训练累计 3000 步 | 3000 | `RUN_DIR/checkpoints/002000/pretrained_model` |
| 已有默认配方的 flow 3000 步完成 | 4000 | `RUN_DIR/checkpoints/003000/pretrained_model` |

分析默认使用表中累计 3000 的版本。如果“3000 steps”指原脚本的 **flow 3000**，设置 `FINAL_FLOW_STEPS=3000`；图上会如实标成累计 4000。
不修改现有消融训练的默认步数。

## 2. 训练时保存准确的四个快照

现有脚本原本只保存预训练结束的 1000 checkpoint，`SAVE_FREQ=750` 也不会改变其独立预训练保存频率。
现在可以通过 `--policy.ntk_save_stage_snapshots=true`，额外保存每个训练阶段自己的初始 checkpoint，以及 priming 结束的 750 checkpoint。
这是可选开关；未开启时原保存行为不变。所有快照保存完整模型、实际 prompts、预处理器和训练状态，额外占用与完整 checkpoint 相当的空间。

累计训练 3000 步的新实验（先在当前训练环境 `pip install -e .`）：

```bash
export DATASET_REPO_ID=libero
export DATASET_ROOT=/root/autodl-tmp/datasets/libero
export PRETRAINED_PATH=/root/autodl-tmp/models/pi05_libero_base
export TOKENIZER_PATH=/root/autodl-tmp/models/google/paligemma-3b-pt-224
export GPU_IDS=0
export RUN_NAME=pi05-full-reference-ntk-total3000-seed0

FLOW_STEPS=2000 SAVE_FREQ=2000 \
  bash examples/training/train_pi05_prompt_ablation.sh full_reference 0 \
  --policy.ntk_save_stage_snapshots=true
```

若要沿用已有 **750 + 250 + 3000** 配方，改为 `FLOW_STEPS=3000 SAVE_FREQ=3000`。
训练结束后，完整 flow 阶段的 `checkpoints/000000` 对应累计 1000，**不能**当作整个实验训练前的快照。
如果旧实验没有保存 0 或 750，无法从 1000/3000 的权重反推出这些时间点；需要重新运行并打开上述快照开关。脚本不会用随机初始化的 prompts 或别的 checkpoint 代替缺失时间点。

## 3. 四时间点分析

在保存了训练权重和数据的机器上运行。无需 LIBERO 模拟器，也不会执行机器人动作。

```bash
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-total3000-seed0

# 先检查时间点、文件和保存的训练步数；不会加载模型或使用 GPU。
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0 --dry-run

# 正式分析：32 个固定 samples，10 个配对随机种子，每种子 4 个输出探针。
bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0
```

已有 flow-3000 的实验：

```bash
FINAL_FLOW_STEPS=3000 \
  bash examples/analysis/analyze_pi05_ntk_stages.sh "$RUN_DIR" 0
```

也支持独立指定四个路径：

```bash
python -m lerobot.scripts.analyze_pi05_ntk_stages \
  --before=/path/to/pretrain/checkpoints/000000 \
  --priming=/path/to/pretrain/checkpoints/000750 \
  --stage2=/path/to/pretrain/checkpoints/001000 \
  --final=/path/to/flow/checkpoints/002000 \
  --final-flow-steps=2000 \
  --dataset-repo-id=libero --dataset-root=/path/to/libero \
  --tokenizer-path=/path/to/paligemma-3b-pt-224 \
  --output-dir=outputs/ntk_stages
```

`--scope=backbone` 只画 backbone；`--scope=prompts` 只画实际微调参数，显存需求更低。
默认 `--scope=both`。完整 backbone 分析会临时启用数十亿参数的梯度；建议先在空闲 A100 80GB 上用
`--num-samples=2 --seeds=0 --output-probes=1 --output-dir=outputs/ntk_smoke` 做一次小规模运行。
这不是完整实验结果；正式运行用另一个输出目录。没有执行 optimizer step，也不修改 checkpoint。

默认使用前 32 个样本，保持旧实验固定样本的习惯。若希望覆盖更多任务/episode，使用
`--indices=0,500,1000,...` 指定真实有效索引（不要把省略号写进命令）。索引会保存到 manifest；四个时间点必须一致。
默认未进行图像增强。`TOKENIZER_PATH` 可修复跨机器迁移后保存的 tokenizer 路径失效问题。

## 4. 指标与可比性

所有时间点均测同一个**完整视觉语言条件下的 flow velocity prediction**，不是训练 loss 的梯度，也不是多步积分后的最终控制动作。
即使 750 checkpoint 来自纯动作 priming，分析时也调用 `predict_velocity`，不调用其纯动作 inpainting objective。
否则 VLM 在 priming 的计算图中不存在，会把结构性零误画成敏感性下降。

设第 i 个样本的有效 action velocity 向量为 f_i，参数组 m 的 Jacobian 为 J_i,m。
目标是输出坐标求和后的 sample Gram：

```text
K_m[i,j] = sum_a < d f_i[a] / d theta_m, d f_j[a] / d theta_m >
p_k = lambda_k(K_m) / sum_j lambda_j(K_m)
Spectral effective rank = exp(-sum_k p_k log p_k)
Parameter-normalized tangent energy = Tr(K_m) / P_m
```

横轴采用谱熵有效秩，不是 participation rank；后者额外保存到 CSV。纵轴 **没有再除以 effective rank，也没有除以样本数**。
不同样本数、输出维度/长度或参数组定义的绝对能量不能直接与旧图比较。零 kernel 的秩和能量定义为 0；实测任一组全零会报错，避免静默接受断开的计算图。

完整输出 Jacobian 太大，脚本使用共享 Rademacher 输出探针，近似上述 NTK：

```text
g_i,m,q = J_i,m^T r_q,  E[r_q r_q^T] = I
Khat_m[i,j] = mean_q <g_i,m,q, g_j,m,q>
```

每个探针必须在**所有样本、四个 checkpoint** 间共享；每个配对 seed 固定其 flow noise、timestep 和输出探针。
噪声/timestep 按样本固定，在 checkpoint 切换、模型初始化或断点续跑时不会漂移。
默认 10 个 seed 衡量探针与 flow 随机性的波动，**不是 10 次独立训练的误差条**。

对 backbone，为避免保存 N × P 的巨大梯度矩阵，谱计算再用固定的 **8192 维 CountSketch** 压缩参数轴；因此 backbone 的 rank 是近似值。
**能量从压缩前的梯度平方范数计算**，不受参数 sketch 的误差影响（仍有输出探针误差）。
Prompt 参数较少，保留完整参数轴，只有输出投影近似。
JSON 保存 kernel、谱、参数量、参数名称/形状、sketch trace 误差。正式用于论文前，建议固定所有其他设置，增大
`--output-probes` 和 `--sketch-dim` 检查主要趋势是否稳定；换目录保存，避免混用估计口径。

分析强制 float32、关闭 compile、梯度 checkpointing 和 frozen-observation detach 优化。
先调用 `eval()`，再临时设置目标参数的 `requires_grad=True`，避免 PI05 的 eval/train 钩子重新冻结 backbone。
四次均使用训练前保存的同一个预处理器（包括归一化统计），并核对模型结构配置、prompt 权重存在性和实际训练步数。
所有 checkpoint 应来自同一次训练；文件元数据无法证明人为搬移的权重来源，需保持实验目录的对应关系。

冻结 backbone 的参数值可以不变，但其 NTK 仍会随训练后的 prompts 和内部激活改变。
这些图描述局部输出敏感性；不能直接解释为参数实际更新量、性能提升，或预设能量差一定缩小。

## 5. 输出与断点续跑

每个时间点/seed 完成后原子保存一次 JSON；中断后用同一命令继续。
manifest 记录数据样本摘要、checkpoint 元数据、种子、估计器和源代码摘要；配置变化会要求使用新输出目录。

每个 scope 输出以下 PNG 和可编辑矢量 PDF：

- `backbone_stages` / `prompts_stages`：四个时间点的 2×2 散点图，共用坐标轴。
- `backbone_trajectory` / `prompts_trajectory`：保留原横纵轴的四阶段中位数轨迹图。
- `backbone_effective_rank` / `prompts_effective_rank`：有效秩随累计步数的变化。
- `backbone_tangent_energy` / `prompts_tangent_energy`：参数归一化能量随累计步数的变化。

蓝色为 VLM，橙色为 action expert；正能量使用 log 纵轴；误差条/阴影为配对 seed 的中位数和 IQR。
另有 `metrics.csv`、`results.json` 和 `manifest.json`。

重新绘图无需 GPU：

```bash
python -m lerobot.scripts.plot_pi05_ntk_stages "$RUN_DIR/ntk_stages/results.json"
```

依赖沿用 PI05 训练环境，并安装 `matplotlib`。真实权重和数据只在用户训练机器上可用；提交时的 CPU 测试验证解析 Jacobian、估计器、快照步数解析与绘图，不代替真实 GPU 分析。

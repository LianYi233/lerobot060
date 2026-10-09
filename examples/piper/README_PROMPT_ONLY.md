# Piper：只训练两组 prompt，比较 16 / 32 / 64 个 token

入口：`bash examples/training/train_piper_prompt_only.sh TASK TOKENS SEED`。
默认 task 1、每组 32 个 token、seed 0；TASK 支持 1/2/3/4/all。
`TOKENS` 是**每一组**的数量，例如 32 表示 VLM 32 个、action 32 个。

本轮检验增加 prompt 容量能否改善真机动作拟合。只训练以下两个权重：

- `model.vlm_prompt_tokens.weight`
- `model.prompt_tokens.weight`

VLM、视觉编码器、整个 action expert、动作输入/输出投影、时间 MLP 和最终 norm 都冻结。
不启用 LoRA/PEFT。仅将解冻层数设为 0 不一定满足这个条件，因为其他入口还会训练投影；
此入口同时显式设置 `train_action_projections=false`、`train_action_expert_last_n_layers=0`。

| TOKENS | VLM prompt 形状 | action prompt 形状 | 可训练参数总数 |
| --- | --- | --- | ---: |
| 16（同条件基线） | 16 × 2048 | 16 × 1024 | 49,152 |
| 32 | 32 × 2048 | 32 × 1024 | 98,304 |
| 64 | 64 × 2048 | 64 × 1024 | 196,608 |

计数对应默认 gemma_2b / gemma_300m 架构，不包括被冻结的原模型。
32/64 档已经超过原来约 0.05M 的 prompt 参数预算，报告实验时应分别标明。
这是减少训练参数，推理仍需完整模型；保存的也仍是完整 checkpoint，不是仅含 prompt 的文件。
冻结骨干仍需让梯度经过其计算图传回 prompt，增加 token 或 batch 都可能增加显存。

## 固定的实验条件

- 从同一个原始 base 开始，显式设置 `PRETRAINED_PATH`。不要从已经解冻训练过的
  last4 / last18 成品 checkpoint 开始，否则不再是从原始 base 进行 prompt-only 适配。
- 默认每卡 batch 64，可用 `BATCH_SIZE` 覆盖。两卡时全局 batch 为 128；每卡 128 时
  全局为 256。进程数自动由 `GPU_IDS` 决定，没有增加梯度累积。
- 两组 prompt 共用峰值 LR 5e-5、最终 LR 5e-6，保留 warmup + cosine。
  用 `PROMPT_LR` / `PROMPT_FINAL_LR` 修改；旧的 `OPTIMIZER_LR` 导出值不生效。
- 默认 12000 次 direct flow 更新，保存 6000、9000、12000；priming/bridge 为 0，CABO 关闭。
  这轮用于隔离 prompt 数量的影响，不是完整分阶段方法的实验。
- absolute 动作，预测 16 / 执行 8，FP32，gradient checkpointing 开启，compile 关闭。
- 使用全部有效示范，将末尾 10% episode 留作验证；仅从训练 split 重算归一化统计。
  图像增强关闭，保留数据审计；固定 train/val 样本每 500 步评估动作误差。

这些固定项会覆盖终端遗留的层数、token 数、单示范选择、horizon 等设置。
`BATCH_SIZE`、`FLOW_STEPS`、`SAVE_STEPS`、路径和 W&B 开关仍可通过环境变量设置。
不接受额外 CLI 参数，以免无意解冻其他参数。旧 direct/fit/capacity 入口保持 16+16 个 prompt。

## 两终端训练 task 1

确认对应 GPU 空闲。在两个终端分别运行一次以下共同设置；新的一轮实验请更换 RUN_GROUP。

```bash
conda activate lerobot
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper

export WORK_ROOT=/data1/wyn/piper-work
export PRETRAINED_PATH=/data1/wyn/piper-work/models/pi05_base
export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export RUN_GROUP=piper-task1-prompt32-64-b64-12k-01
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'
export PROMPT_LR=0.00005
export PROMPT_FINAL_LR=0.000005
```

终端 1，GPU 0、1，每组 32 个 token：

```bash
GPU_IDS=0,1 MAIN_PROCESS_PORT=30032 BATCH_SIZE=64 DRY_RUN=false \
  bash examples/training/train_piper_prompt_only.sh 1 32 0
```

终端 2，GPU 2、3，每组 64 个 token：

```bash
GPU_IDS=2,3 MAIN_PROCESS_PORT=30064 BATCH_SIZE=64 DRY_RUN=false \
  bash examples/training/train_piper_prompt_only.sh 1 64 0
```

如果希望延续每卡 128 的实验，将两条命令中的 `BATCH_SIZE=64` 都改成 `BATCH_SIZE=128`，
并建议在共同 RUN_GROUP 中也写 `b128`。启动后的实际显存峰值需要现场测量，
不能从旧 16+16 prompt 的占用直接推断 64+64 一定能放下；若不足，两组同步降低 batch。
保持 32 / 64 两组的 batch、LR、base、训练步数和数据划分相同，才能比较 token 数的影响。
相同训练步数下增大 batch 会增加样本处理总量，不能将与旧小 batch 的差异只归因于 prompt。

需要打印检查命令时，临时改为 `DRY_RUN=true`：正常打印后退出，不开始训练，不验证权重
加载、反向传播或 GPU 显存。正式训练用 `DRY_RUN=false`。

如需同条件的 16+16 基线，在卡空闲后运行：

```bash
GPU_IDS=0,1 MAIN_PROCESS_PORT=30016 BATCH_SIZE=64 DRY_RUN=false \
  bash examples/training/train_piper_prompt_only.sh 1 16 0
```

## 检查冻结范围和结果

训练开始时检查实际的 `Trainable parameter report`。32 档合计应为 98,304，64 档应为
196,608，且 `action_projection=0`、`action_expert_blocks=0`、`other=0`。
模型结构不同则以实际报告为准，不能仅依赖启动脚本打印的默认架构估算。

`OUTPUT_ROOT`、`LOG_ROOT` 是父目录；入口追加 token 数和每卡 batch：

```text
$OUTPUT_ROOT/prompt-only-t32-b64/
$OUTPUT_ROOT/prompt-only-t64-b64/
```

每组最终模型相对于上述目录为：

```text
pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

使用 batch 128 时子目录分别为 `prompt-only-t32-b128`、`prompt-only-t64-b128`。
已有输出会被拒绝覆盖。同 token / batch 重跑，或改变 LR、base、GPU 数时，请同时更换
RUN_GROUP、OUTPUT_ROOT、LOG_ROOT；相同 token 并行跑不同 seed/batch 时还须指定不同端口。
checkpoint 的 config 会保存实际 token 数，部署时应加载对应完整 checkpoint，不手动改回 16。

W&B 默认开启，job 名含 token/batch。保留本地 `training_diagnostics.jsonl`、
`action_eval.jsonl`、`action_eval_samples.json` 和 `best_action_checkpoints.json`。
可设置 `WANDB_MODE=offline` 本地记录，或 `WANDB_ENABLE=false` 关闭 W&B。

除 loss 外，比较 train/val 的前 1 / 8 步关节 MAE、P95、moving 子集、夹爪 MAE，以及
两组 prompt 的梯度和实际更新。特别检查靠近物体和闭合夹爪阶段的预测轨迹。
更多 token 不保证 loss 达到 0.02，也不保证抓取成功；训练集误差下降而验证误差上升时，
应先处理过拟合。只有真机成功率测试才能回答最终是否改善抓取。

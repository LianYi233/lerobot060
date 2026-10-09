# Piper：较少解冻层数，降低学习率并增大 batch

目标：检查 last4 / last8 能否在更低 LR 和更大 batch 下接近 last18 的拟合结果。
这不是保证 loss 达到 0.02 的设置；同时比较验证集关节和夹爪误差。

入口增加第四个可选参数：

```bash
bash examples/training/train_piper_capacity.sh TASK PROFILE SEED RECIPE
```

旧的三个参数调用等同于 `reference`，不改变旧配方或旧输出路径。
新功能只调整启动参数，不增加模型配置字段，也不改变 loss 分母或归一化方式。

| RECIPE | prompt/投影峰值 → 最终 LR | 专家层峰值 → 最终 LR | 默认每卡 batch | 默认端口 |
| --- | --- | --- | ---: | --- |
| `reference` | 1e-4 → 1e-5 | 1e-5 → 1e-6 | 16 | 29500 + 解冻层数 |
| `low_lr` | 5e-5 → 5e-6 | 5e-6 → 5e-7 | 16 | 29600 + 解冻层数 |
| `big_batch` | 1e-4 → 1e-5 | 1e-5 → 1e-6 | 32 | 29700 + 解冻层数 |
| `low_lr_big_batch` | 5e-5 → 5e-6 | 5e-6 → 5e-7 | 32 | 29800 + 解冻层数 |

每组均使用原来的 warmup + cosine，专家 LR 保持为 prompt/投影的 0.1 倍。
配方覆盖旧的 OPTIMIZER_LR、SCHEDULER_DECAY_LR 和 ACTION_EXPERT_LR_SCALE 导出值。
**BATCH_SIZE 仍可覆盖默认值，包括终端遗留的 export。下面显式写 BATCH_SIZE=32，**
启动日志也打印实际每卡 batch、进程数和全局 batch。没有增加梯度累积。

建议先并行跑 `last4 low_lr_big_batch` 和 `last8 low_lr_big_batch`：
可训练参数分别为 94,512,160 和 188,908,576；每组两卡、每卡 32、全局 batch 64。
它们从同一个原始 `pi05_base` 初始化，**不从 last18 的成品 checkpoint 继承权重**。
如果新配方有效，可在同一层数下补跑 `low_lr` 和 `big_batch`，区分两个变量的作用。

## 两终端启动

确认对应卡空闲后，在两个终端分别激活原来的环境。共同设置如下，各粘贴一次。
RUN_GROUP 使用新的名称；后续重跑时同时更换 RUN_GROUP、OUTPUT_ROOT、LOG_ROOT。

```bash
conda activate lerobot
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper

export PRETRAINED_PATH=/data1/wyn/piper-work/models/pi05_base
export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export RUN_GROUP=piper-task1-low-lr-b32-12k-01
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'
```

终端 1：

```bash
GPU_IDS=0,1 MAIN_PROCESS_PORT=29804 BATCH_SIZE=32 DRY_RUN=false \
  bash examples/training/train_piper_capacity.sh 1 last4 0 low_lr_big_batch
```

终端 2：

```bash
GPU_IDS=2,3 MAIN_PROCESS_PORT=29808 BATCH_SIZE=32 DRY_RUN=false \
  bash examples/training/train_piper_capacity.sh 1 last8 0 low_lr_big_batch
```

`DRY_RUN=true` 可检查命令，打印后正常退出，不启动 GPU 训练。
本地未实测 GPU 显存，不保证 batch 32 一定适合你的卡。若需改为 16 或 64，直接设置
`BATCH_SIZE`；保持 last4 与 last8 的 batch 相同，并为新 batch 选择新的输出父目录。
不同 batch 的同一 profile/recipe 不会自动换输出目录，以避免无意覆盖旧实验。

## 输出与比较

两组分别进入：

```text
$OUTPUT_ROOT/capacity-last4-low_lr_big_batch/
$OUTPUT_ROOT/capacity-last8-low_lr_big_batch/
```

每组 task 1 的最终 checkpoint 相对于上述目录为：

```text
pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

仍保存 6000/9000/12000，保持 absolute、预测 16 / 执行 8、同一个 10% episode 验证集、
FP32、16+16 prompts、无 priming/bridge/CABO。每 500 步评估 128 个固定样本，
W&B、`action_eval.jsonl`、`training_diagnostics.jsonl` 和已保存最佳 checkpoint 选择都保留。
W&B job 名和日志目录带 profile 与 recipe，可直接区分两组。

**与原来 batch 16 的结果比较时，这不是相同计算量的实验。** 两卡 batch 16、12000 步
处理 384,000 个训练样本位置；两卡 batch 32、12000 步处理 768,000 个，增加一倍，
实际耗时也可能增加。这些是重复采样次数，不是新增示范数量。可用大 batch 的 6000 步
checkpoint 对照旧小 batch 的 12000 步做相同样本量的补充比较，但两者优化步数和所处
学习率进度不同，仍不能据此单独归因于 batch。

比较尾段平均训练 loss，同时检查 train/val 的 h1/h8 关节 MAE、P95、moving 子集和夹爪
MAE。若训练 loss 降低而验证误差变差，需要处理过拟合。若更低 LR 只是让下降更慢，
不要将继续降低 LR 当作固定方向。能否达到 0.02，以及是否改善抓取，均由实际结果判断。

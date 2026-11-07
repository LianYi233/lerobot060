# reality 四任务长训练：单张 A800

本入口用于当前 `/root/autodl-fs/datasets/reality` 数据：**按 2 → 3 → 4 → 1 顺序，每个任务独立训练一个 PrimingVLA 模型，默认累计 50,000 次优化更新**。单张 A800 串行运行，默认 batch 8、训练种子 0，不使用 W&B，也不启动机器人。

```bash
DRY_RUN=false bash examples/training/train_piper_revision_long.sh
```

这是 `train_piper_revision.py --preset reality-long` 的便捷入口。原有 [审稿受控实验入口](README_REVISION.md) 仍默认 `core` 四组、累计 4,000 步；其设置和已有结果保留原口径。本轮默认先训练四个 `priming` 模型。如用长训练结果做论文方法比较，其他组也需要采用同一预算、数据划分及长训练学习率日程。

## 1. 数据与训练预算

目录名直接沿用 `examples/training/train_pi05_real.py` 中的 `TASKS`，无需重命名或合并任务：

| 执行次序 | 任务 | `DATASET_BASE` 下的目录 |
|---:|---|---|
| 1 | T2：从蓝盘移走长方体 | `2-remove_the_cuboid_from_blue_plate` |
| 2 | T3：网球从黄盘移至蓝盘 | `3-move_the_tennis_from_yellow_plate_to_blue_plate` |
| 3 | T4：红方块放入黄盘 | `4-pick_the_red_cube_into_the_yellow_plate` |
| 4 | T1：苹果放入黄盘 | `1-put_the_apple_on_the_yellow_plate` |

每个子目录应包含其完整 LeRobot 数据及 `meta/info.json`；轨迹数以实际元数据为准。每个任务最后 10% 的完整 episode 留作验证，归一化统计只由训练集计算；具体划分和统计会保存并核对。

| 阶段 | 更新数 | 累计更新数 |
|---|---:|---:|
| SAP | 750 | 750 |
| Bridge | 250 | 1,000 |
| Adaptation | 49,000 | 50,000 |

所有任务都从同一份**原始 PI05 底座**独立初始化，不接着训练前一个任务的模型。仍使用 16+16 个 prompt，共 **49,152** 个可训练参数；底座和 action head 冻结，不启用 history20。输入为两路 RGB、语言和当前离散状态 token，预测 chunk 50，部署默认每次执行 8 步。

长训练只调整 Adaptation 日程：warmup **1,000** 更新，峰值基础 LR **2.5e-5**，随后余弦衰减至 **1e-6**，衰减总长度取该任务实际 Adaptation 更新数。SAP/Bridge 保留原 recipe；CPC 沿用完整 `priming` 方法。

默认保存累计 **10,000 / 30,000 / 50,000** 的 checkpoint，对应 Adaptation **9,000 / 29,000 / 49,000** 步，另保留准备阶段累计 1,000 步。累计预算已经包含 SAP 与 Bridge。

旧 `train_piper_autodl.sh` 默认 `full_reference` 的 `FLOW_STEPS=12000` 指 **12,000 次 Adaptation**，加准备阶段 1,000 后为累计 **13,000**；不能将它与本入口的累计 50,000 直接按同名步数比较。

## 2. 在另一台 AutoDL 上运行

在已安装本分支训练依赖的 Python 3.12+ 环境中，进入 `piper` 分支仓库根目录。下面两项模型路径必须替换成**这台 AutoDL 上真实存在的路径**；数据路径已按当前要求设置。

```bash
# 当前目录应为 lerobot060 的 piper 分支仓库根目录。
export WORK_ROOT=/root/autodl-tmp/piper-revision
export DATASET_BASE=/root/autodl-fs/datasets/reality
export PRETRAINED_PATH=/path/to/original-pi05-base
export TOKENIZER_PATH=/path/to/local-tokenizer
export GPU_ID=0
export BATCH_SIZE=8
export LOSS_TARGET=0.0001
export LOSS_WINDOW_UPDATES=500

# 使用 WORK_ROOT/runs/<run-group>，避免沿用旧实验导出的输出目录。
unset OUTPUT_ROOT

# 检查数据、模型、任务顺序、预算及命令；不启动训练。
DRY_RUN=true bash examples/training/train_piper_revision_long.sh

# 确认上面的实际路径和计划后，开始四任务串行训练。
DRY_RUN=false bash examples/training/train_piper_revision_long.sh
```

`PRETRAINED_PATH` 应包含 `config.json`、`model.safetensors`、两个 policy processor 配置等原始底座文件；不能指向已有 prompt、已适配 action head/history 的模型，或下面说明的 `storage_base`。`TOKENIZER_PATH` 应包含本地 tokenizer 配置及 tokenizer 文件。本文没有假定另一台服务器已经把这些文件放在某个目录。

默认输出为：

```text
/root/autodl-tmp/piper-revision/runs/reality-long-v1/
```

`DATASET_BASE`、`WORK_ROOT` 和 `OUTPUT_ROOT` 可以显式用环境变量覆盖。长 preset **不继承**旧环境中的 `TOTAL_UPDATES`、`CUMULATIVE_CHECKPOINTS`、`TASKS`、`RUN_GROUP`；预算、检查点、任务和实验组使用本入口默认值或下面的 CLI 参数。

### 分批完成与查看状态

如需将四任务分两次执行，保持相同预算及 run group：

```bash
DRY_RUN=false bash examples/training/train_piper_revision_long.sh --tasks 2,3
DRY_RUN=false bash examples/training/train_piper_revision_long.sh --tasks 4,1

bash examples/training/train_piper_revision_long.sh --status
```

`--status` 只读取输出记录，无需指定底座或 tokenizer，并显示 `latest_adaptation_loss / trailing_mean / window_updates / target_met` 列。长训练期间，终端和输出根目录下的 `logs/<run>.log` 每 30 秒显示任务、阶段、进度、耗时及科学计数法的 loss；这些 loss 列的含义见下节。再次运行同一计划时，已完成任务在身份和 bundle 校验通过后跳过；**中断或失败的部分训练目前不自动恢复**，其文件会保留。不要将部分目录手动标成完成；如决定从头重跑，使用新的 `--run-group`。

### 延长到累计 80,000 步

```bash
DRY_RUN=true bash examples/training/train_piper_revision_long.sh \
  --total-updates 80000 \
  --checkpoints '[10000,40000,80000]' \
  --run-group reality-long-80k-v1

DRY_RUN=false bash examples/training/train_piper_revision_long.sh \
  --total-updates 80000 \
  --checkpoints '[10000,40000,80000]' \
  --run-group reality-long-80k-v1
```

这里 SAP750 + Bridge250 保持不变，Adaptation 为 79,000，学习率衰减长度随之延长。新 run group 从原始底座重新训练，不是在 50k 模型上自动续训。如显式设置了 `OUTPUT_ROOT`，也应为新实验选择新的输出目录。

## 3. 如何判断 loss 是否接近 `1e-4`

默认 `LOSS_TARGET=0.0001`、`LOSS_WINDOW_UPDATES=500`，每 50 次更新记录一次训练日志。监控只读主 Adaptation 阶段，SAP/Bridge 不混入窗口。`latest_logged_loss` 是最近一条日志的均值，通常覆盖最近 50 次更新；`trailing_loss` 按实际更新数加权，取覆盖至少 500 次更新的最短连续完整日志窗口，`actual_window_updates` 记录实际覆盖量。窗口不足时尚不判定达标。`stable_target_met` 只表示完整窗口的加权均值不高于目标，**不表示每个 batch 的 loss 都低于目标**。`first_target_step` 是当前连续日志段首次窗口达标时的主阶段局部步数；launcher 另记录包含准备阶段的 `first_target_cumulative_update`。

以先运行的 T2 为例：

```bash
python examples/piper/revision_loss.py \
  "$WORK_ROOT/runs/reality-long-v1/task2-priming-seed0-total50000" \
  --target 0.0001 --window-updates 500
```

任务完成后，模型输出目录的 `loss_summary.json` 和 `revision_manifest.json` 中的 `loss_summary` 字段会持久化该报告。

目标只用于跟踪，**不会自动早停，也不保证在 50k 或 80k 时达到**。Flow 训练涉及随机噪声和时间采样，loss 数值还依赖动作归一化与损失定义；不能据此承诺真机成功率。应同时查看保留验证集的动作指标，并将预先确定的 checkpoint 做相同闭环评估。训练 loss 继续下降而验证指标变差时，不应仅为了到达 `1e-4` 再增加预算。

## 4. 300G 数据盘与模型取回

启动前检查实际剩余空间，保留默认 **30 GiB** 余量，并计入本轮多个完整 checkpoint 的临时峰值；300G 是容量限制，不能当作当前空闲空间。长训练默认有三个 Adaptation 检查点，预检会据此重新估算。程序记录实际训练进程耗时和**训练步期间 CUDA allocated 显存最大值**；后者不含初始化/验证阶段，也不是 `nvidia-smi` 的总显存占用。没有预设的完成小时数。

完成后，checkpoint 被无损打包并逐个验证重建 SHA256。首次完成的训练 checkpoint 生成共享 `storage_base/model.safetensors`，**仅用于无损重建，绝不能作为后续训练底座**。所有 bundle 验证成功后，仅删除本轮自己的完整大权重文件；配置、优化器/RNG 状态及记录保留。失败或中断时保留现场。

例如取回 T2 最终累计 50k 模型：

```bash
REALITY_RUN_ROOT="$WORK_ROOT/runs/reality-long-v1"

python examples/piper/revision_checkpoint.py materialize \
  --bundle "$REALITY_RUN_ROOT/task2-priming-seed0-total50000/bundles/total-050000" \
  --base "$REALITY_RUN_ROOT/storage_base" \
  --output "$WORK_ROOT/materialized/task2-priming-seed0-total50000"
```

恢复目录包含标准部署权重、配置、processor 和 `dataset_info.json`；恢复时会校验原始权重哈希。请先准备足够空间，输出目录使用未存在的新路径。传回机器人电脑后按 [部署说明](README.md) 先做 `--dry_run`，再由现场操作人员使用现有保护流程开展闭环试验。

真机记录见 [配对闭环评估说明](README_REVISION_EVAL.md)。评估工具默认预算是 4,000、方法为 core 四组；只评估本轮四个模型时应在 `plan` 命令中显式传 `--methods priming --cumulative-updates 50000`，并绑定实际恢复 checkpoint 的身份。长训练入口及文档检查不代表已在这台 AutoDL 上跑通训练，更不代表已完成真机试验。

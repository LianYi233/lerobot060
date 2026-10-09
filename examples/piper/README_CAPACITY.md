# Piper：扩大动作专家的训练范围

入口：`bash examples/training/train_piper_capacity.sh TASK PROFILE SEED [RECIPE]`。
默认 task 1、`last8`、seed 0、`reference`。省略第四项时保留原来的 LR、batch 默认值与输出目录。
复用现有 last-N 实现，不改网络结构、loss 定义或部署驱动，
不添加新的 checkpoint 配置字段。旧训练入口和旧 checkpoint 的行为不变。

较少解冻层数、较低学习率和较大 batch 的下一轮命令，见
[低学习率 / 大 batch 对照](README_CAPACITY_LR_BATCH.md)。

若希望完全冻结动作专家和动作投影、只训练两组 prompt，使用独立的
[prompt-only 16 / 32 / 64 token 对照](README_PROMPT_ONLY.md)，不要沿用本入口的投影解冻设置。

这组实验检验“动作专家的适配容量是否限制了 Piper 拟合”，不能保证成功抓取。
仿真和真机不需要遵守同一个可训练参数预算，但应按迁移难度、数据量及验证结果决定，
不存在固定的“真机必须解冻百分之多少”。本实验不属于 prompt-only 参数预算。

## 训练范围

| PROFILE | 动作专家 transformer | 可训练参数 | 用途 |
| --- | --- | ---: | --- |
| `last2` | 最后 2 层 | 47,313,952 | 同条件对照 |
| `last4` | 最后 4 层 | 94,512,160 | 较小扩容 |
| `last8`（默认） | 最后 8 层 | 188,908,576 | 下一轮首选，约为 last2 的 4 倍 |
| `last18` | 全部 18 层 | 424,899,616 | last8 仍有明显训练集动作误差时的更大容量对照 |

每档都训练 16+16 个 prompt 和动作输入/输出投影。计数对应 gemma_2b/gemma_300m、
max_action_dim=32：prompt 49,152 + 投影 66,592 + N × 23,599,104。
实际参数总量与可训练比例以日志 `Trainable parameter report` 为准。
**last18 是全部专家 transformer 层，不是整个模型或整个动作头：VLM、视觉编码器、
专家最终 norm、时间 MLP 仍冻结。** 如果扩大专家后仍缺乏视觉定位能力，应另做视觉侧
适配实验；本轮不同时改 base、相机处理和 loss 权重。

官方 LeRobot 提供冻结 VLM、训练动作专家及投影的方案，可作为增加动作侧容量的参考，
但其配置不能原样套到本 prompt 分支：
[PI05 training parameters](https://huggingface.co/docs/lerobot/pi05#training-parameters-explained)。

## 比较条件

- 同一个初始 base；必须显式设置 `PRETRAINED_PATH`，可以是已有 LIBERO、通用 base
  或 DROID 的 LeRobot 权重目录。不自动换基座，不将成品 Piper checkpoint 单独用于一组。
- 全部有效示范的末尾 10% episode 留作验证，其余训练；absolute 动作，预测 16 / 执行 8。
- 默认 12000 次 flow 更新，保存 6000、9000、12000；priming、bridge 为 0，CABO 关闭。
- `reference`：prompt/投影峰值 LR 1e-4、最终 1e-5；专家层峰值 1e-5、最终 1e-6。
- `low_lr`：两个参数组的 LR 全程减半；`big_batch`：仅将默认每卡 batch 从 16 增至 32；
  `low_lr_big_batch`：同时应用两项。各组保留 warmup + cosine，增大 batch 不自动提高 LR。
- FP32、compile 关闭、gradient checkpointing 开启；`BATCH_SIZE` 可覆盖配方的默认值。
- 从训练 split 重算统计；固定样本每 500 步测动作误差，每 50 步记录参数更新。

入口主动替换终端遗留的层数、LR、FIT_EPISODES、EVAL_SPLIT、horizon、精度和评估频率，
避免沿用单条示范或低 LR 实验。`GPU_IDS` 决定卡号及进程数，默认 `0,1`。
允许改变 `BATCH_SIZE`、`FLOW_STEPS`、`SAVE_STEPS`、路径和 W&B 开关，比较组保持一致。
自定义其他条件时使用 `train_piper_direct.sh`，此入口不接受额外 CLI 参数。

## 训练 task 1

在现有环境运行，无需重新 pip 安装。把下面的 base 占位路径改成你已经下载的实际目录。

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper
export WORK_ROOT=/data1/wyn/piper-work
export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export PRETRAINED_PATH=/你的初始base模型目录
export RUN_GROUP="piper-task1-capacity-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
export BATCH_SIZE=16
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'

GPU_IDS=0,1 DRY_RUN=true bash examples/training/train_piper_capacity.sh 1 last8 0
GPU_IDS=0,1 DRY_RUN=false bash examples/training/train_piper_capacity.sh 1 last8 0
```

路径例子：`/data/models/lerobot/pi05_libero_base`，或你实际的 `pi05_base` / `pi05_droid`
目录。dry run 只验证路径/元信息和参数，不能证明权重加载及 GPU 训练成功。
如需并行对照，在另一终端设置**相同环境及 RUN_GROUP**，运行：

```bash
GPU_IDS=2,3 DRY_RUN=false bash examples/training/train_piper_capacity.sh 1 last2 0
```

last2 / last8 默认端口 29502 / 29508。OUTPUT_ROOT、LOG_ROOT 在此入口是**父目录**，
`reference` 各自追加 `capacity-PROFILE`；其他配方追加 `capacity-PROFILE-RECIPE`。
不同配方的默认端口也分开。重跑同一 profile/recipe 仍拒绝覆盖，需使用新 RUN_GROUP、
OUTPUT_ROOT、LOG_ROOT。同一 profile/recipe 同时训练不同任务或 seed 时要另设不同
`MAIN_PROCESS_PORT`。若显存不足，降低 batch；严格比较解冻范围时两组同步调整。
本地未实测 GPU 峰值显存，不保证某个 batch 必然可用。

## 保存和评估

last8 的最终模型：

```text
$OUTPUT_ROOT/capacity-last8/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

同一任务目录有 `training_diagnostics.jsonl`、`action_eval.jsonl`、`action_eval_samples.json`、
`best_action_checkpoints.json` 及最佳已保存 checkpoint 的链接。
日志在 `$LOG_ROOT/capacity-last8/`。`dual_prompt_only` 是阶段配方名，不表示只训练 prompt。
W&B 默认开启，job 名含 capacity profile；`WANDB_MODE=offline` 本地记录，
`WANDB_ENABLE=false` 关闭 W&B 后仍保留本地诊断。

比较 train 和 val 的 h1/h8 关节 MAE、P95、moving 子集、夹爪 MAE，以及专家层梯度与
实际更新。W&B 例子：`eval/action_val/h8/joint_mae_deg`、`joint_p95_deg`、
`moving_joint_mae_deg`、`gripper_mae`（后面三项使用相同前缀），
`train/action_update/action_expert_blocks/grad_rms`、`relative_update`。
总体均值之外，检查接近物体和闭合阶段的轨迹图。

训练、验证动作误差都随扩容下降，支持容量受限的解释；只有训练误差下降而验证变差，
应处理过拟合；离线误差低但真机仍偏，继续核对下发命令与实际反馈、相机和控制延迟。

本分支 loss 是归一化动作上的 flow-matching 向量场 MSE，只对 7 个实际动作维度和有效
时间步求平均，不是反归一化动作 MAE。`0.03` 可作**同实现、同数据口径**下的经验参考，
不能作为通用成功门槛。归一化、预测长度、填充 mask、采样和损失定义都会影响数值。
例如，假设某实现把 7 维误差加 25 维零误差后除以 32，同样的 0.07 就会显示为
0.07 × 7/32 ≈ 0.0153，却没有改善抓取；这不是断言别人的实现如此。
不要修改分母或 loss 缩放来追求低于 0.03。

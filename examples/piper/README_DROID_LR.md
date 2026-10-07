# Task 1：两组 GPU 并行比较学习率

入口：`bash examples/training/train_piper_droid_lr.sh 1 low 0` 或 `1 reference 0`。
两组都从同一个 `PI05_DROID_PATH` 初始化，重新训练 12000 步，不续训旧 Piper checkpoint。
比较的是「全程降低学习率」是否改善拟合；不是单独测试「第 4000 步突然降学习率」。

| 项目 | low（GPU 0、1） | reference（GPU 2、3） |
| --- | --- | --- |
| prompt／动作投影峰值 LR | `5e-5` | `1e-4`（原配置） |
| expert 末两层峰值 LR | `5e-6` | `1e-5`（原配置） |
| prompt／动作投影最终 LR | `5e-6` | `1e-5` |
| expert 末两层最终 LR | `5e-7` | `1e-6` |
| 分布式端口 | 29501 | 29502 |
| 输出／日志子目录 | `lr-low` | `lr-reference` |

其余设置一致：absolute、预测 16／执行 8、last2（47,313,952 个可训练参数）、
FP32、每卡 batch 16（每组全局 batch 32）、seed 0、全部有效 episode 中留出 10% 验证、
priming/bridge/CABO 关闭，只保存 6000/9000/12000。
两个模型各用两张卡独立训练，不是一个模型使用四张卡。

## 学习率已经在衰减

当前调度是 warmup + cosine。默认 warmup 1000、decay 30000 会按 12000 步预算缩放成
warmup 400、decay 12000。每次优化更新推进一次 scheduler，不因双卡而加倍推进。
两组保留相同的曲线形状；低学习率组把峰值和最终值一起减半，expert 始终是主组的 0.1 倍。

| 优化步数 | reference 主组 LR | low 主组 LR |
| --- | --- | --- |
| 4000 | `7.75e-5` | `3.875e-5` |
| 8000 | `3.25e-5` | `1.625e-5` |
| 12000 | `1e-5` | `5e-6` |

loss 在 4000 步附近变平，不能单独证明 LR 太大，也不能由 flow loss `0.05` 判断抓取精度。
较低 LR 可能让更新更稳定，也可能只是学得更慢。以相同样本上的验证动作误差决定是否保留。
这里不同时更换数据、基座、可训练层数或 action 表示。

## 启动：两个终端分别运行

先在仓库拉取本分支，使用已经能训练 DROID 的 conda 环境；无需重新安装或升级依赖：

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper
```

**两个终端都执行下面这段公共设置**。`RUN_GROUP` 取同一实验名；已有同名 checkpoint 时
脚本会拒绝覆盖，重跑请更换名字。确认卡 0、1、2、3 没有其他训练占用。

```bash
cd /home/wyn/VLAA/lerobot060
export WORK_ROOT=/data1/wyn/piper-work
export PI05_DROID_PATH="$WORK_ROOT/models/pi05_droid"
export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export RUN_GROUP=piper-task1-droid-lr-12k-01
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
export BATCH_SIZE=16 FIT_EPISODES=all EVAL_SPLIT=0.1
export FLOW_STEPS=12000 SAVE_STEPS='[6000,9000,12000]'
export CHUNK_SIZE=16 N_ACTION_STEPS=8 MASKED_STEPS=12
export DTYPE=float32 MIXED_PRECISION=no COMPILE_MODEL=false
export ACTION_EVAL_FREQ=500 ACTION_EVAL_SAMPLES=128 ACTION_EVAL_SEED=0
export ACTION_EVAL_SAMPLING=episode_stratified ACTION_UPDATE_FREQ=50
export ACTION_SELECT_BEST_SAVED=true DATA_AUDIT=true
export WANDB_ENABLE=true WANDB_PROJECT=piper-action-fit WANDB_MODE=online
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

终端 A：

```bash
DRY_RUN=true bash examples/training/train_piper_droid_lr.sh 1 low 0
DRY_RUN=false bash examples/training/train_piper_droid_lr.sh 1 low 0
```

终端 B（可与 A 同时运行）：

```bash
DRY_RUN=true bash examples/training/train_piper_droid_lr.sh 1 reference 0
DRY_RUN=false bash examples/training/train_piper_droid_lr.sh 1 reference 0
```

`DRY_RUN=true` 只检查路径/元数据并打印命令，不加载模型，也不占 GPU。
此入口会覆盖旧终端导出的 `GPU_IDS`、`NUM_PROCESSES`、`MAIN_PROCESS_PORT`、LR 和 W&B 显示名，
清除继承的 `WANDB_RUN_ID` / `WANDB_RESUME`，避免两组串用旧配置或日志。
正式命令应分别显示 `GPUs=0,1; port=29501` 与 `GPUs=2,3; port=29502`。
如端口被其他作业占用，可分别设置 `LR_LOW_PORT` / `LR_REFERENCE_PORT` 为不同的空闲端口。
如需调整 GPU 对，设置 `LR_LOW_GPU_IDS` / `LR_REFERENCE_GPU_IDS`，每组两张、互不重叠。

该入口将 `OUTPUT_ROOT` / `LOG_ROOT` 作为两组的公共父目录，自动添加 `lr-low` 或
`lr-reference`；其他训练入口的目录规则没有改变。其他 env 覆盖仍有效，两边务必设置一致。
网络无法连接 W&B 时，两边都设 `WANDB_MODE=offline`，本地 JSONL 继续记录。

### 启动时报 output directory already exists

`lr-low` 和 `lr-reference` 本来就是两个目录，不会因为同时训练而共用输出目录。
旧版训练入口缺少「所有 rank 完成配置验证后再创建输出目录」的同步点，较快 rank 的
W&B 初始化可能先创建目录，使较慢 rank 的配置验证误报目录已存在。当前版本已在创建
Accelerator 后添加 barrier；仍保留对已有目录的覆盖保护。

若失败的 reference 已退出，low 仍正常运行，只在 reference 终端执行以下命令重试。
更新代码后为失败组同时更换 RUN_GROUP、OUTPUT_ROOT 和 LOG_ROOT；旧目录保留用于排查。
仅修改 RUN_GROUP 不会覆盖已经 export 的 OUTPUT_ROOT / LOG_ROOT。

```bash
git pull --ff-only origin piper
export RUN_GROUP="piper-task1-droid-lr-reference-retry-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
DRY_RUN=false bash examples/training/train_piper_droid_lr.sh 1 reference 0
```

其他路径、batch、seed 和训练配置继续使用上面的相同设置。若是已有正常运行的 reference，
不要重复启动；若确实需要从已保存 checkpoint 续训，应另行指定完整训练状态恢复配置。

## 模型、日志和比较指标

在以上路径设置下，两组模型分别保存在：

```text
$OUTPUT_ROOT/lr-low/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
$OUTPUT_ROOT/lr-reference/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

对应训练文本日志：

```text
$LOG_ROOT/lr-low/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0.log
$LOG_ROOT/lr-reference/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0.log
```

各模型运行目录（`checkpoints` 的父目录）还会记录：

- `action_eval.jsonl`：固定训练/验证样本的动作误差，每 500 步评估。
- `action_eval_samples.json`：样本选择清单，比较两组时先确认一致。
- `training_diagnostics.jsonl`：梯度、参数更新和 LR 等诊断；训练配置也记录具体 LR。
- `checkpoints/best_joint`、`checkpoints/best_gripper`：已保存 checkpoint 中验证误差最好的链接。

W&B 显示名分别含 `lr-low` / `lr-reference`。可查看：

- `train/loss` 与 `train/lr`。
- `train/action_update/lr/prompts_projections`、`train/action_update/lr/action_expert_blocks`：实际优化 LR。
- `eval/action_val/h8/joint_mae_deg`、`eval/action_val/h8/joint_p95_deg`。
- `eval/action_val/h8/moving_joint_mae_deg`、`eval/action_val/h8/gripper_mae`，以及 `h1`、`h16` 对应量。

先比较同一步数（6000/9000/12000），再比较各自最佳保存点。训练 loss 下降而验证误差不改善，
不能当成真机改进；若两组离线都好、真机仍偏移，继续检查动作跟踪、时序和视觉输入的一致性。

若以后需要任意 LR，可直接使用 `train_piper_droid.sh` / `train_piper_base.sh` / `train_piper_direct.sh`
的 `OPTIMIZER_LR`、`SCHEDULER_DECAY_LR`、`ACTION_EXPERT_LR_SCALE` 环境变量；默认保持原配置。

# Piper：12000 步直接 flow 与相对关节动作对照

这是排查 Piper 短期动作拟合不足的实验，不能预先保证抓取成功。旧训练入口、
旧 checkpoint 和 prompt-only 方法的默认行为不变。

## 配置和预算

新增入口 `examples/training/train_piper_direct.sh TASK PROFILE SEED`：

| 项目 | `absolute`（先跑） | `relative`（随后对照） |
| --- | --- | --- |
| 更新次数 | 总共 12000，全部为带观测的 flow 更新 | 相同 |
| Priming / bridge / CABO | 0 / 0 / 关闭 | 相同 |
| 六关节训练目标 | 原始绝对关节角 | `action[t+k] - state[t]`，按名称匹配关节 |
| 夹爪训练目标 | 原始绝对夹爪值 | 相同，不做差分 |
| 解冻范围 | 两组 prompt、动作输入/输出映射、expert 末两层 | 相同 |
| 可训练参数（默认模型） | 47,313,952 | 相同，无新增模型张量 |
| 峰值 LR | prompt/映射 1e-4；expert 1e-5 | 相同 |
| LR 调度 | 默认余弦调度缩放为 warmup 400、decay 12000 | 相同 |
| 数据 | 全部有效 episode，末尾 10% episode 留作验证 | 相同 |
| 归一化 | 从训练 split 重算 state/action 统计 | 从训练 split 重算 state/相对 action 统计 |
| 动作窗口 | 预测 50，执行 8 | 相同 |
| 精度 / 编译 | FP32 / 关闭 compile | 相同 |
| 模型保存点 | 6000、9000、12000 | 相同 |

12000 比 3000 提供了更充分的拟合预算，但是否更好要看同一验证集的动作误差。
新实验改了训练阶段和统计来源，不能把新旧曲线差异全部归因于训练步数。
`relative` 又改变了归一化目标分布，不能用它和 `absolute` 的 flow loss 大小判断优劣。
两者的动作评估都还原成绝对关节角/夹爪值之后再算误差，可在相同样本上比较。

相对关节角以**当前观测状态**为参考，整个 50 步 chunk 都加回这同一个状态，
不会累积相加，也不会将标签平移一帧。数据文件和 `meta/stats.json` 保持原样。
新的统计只计入训练 episode 内真实存在的未来动作，排除补齐的尾部标签。
夹爪仍使用数据集单位。恢复训练从 checkpoint 读取统计，不重新套用原始绝对动作统计。

## 训练机命令

在已安装好依赖的训练环境中执行，训练与测试机都继续使用匹配的 Transformers 5.5.4。
无需重新创建环境。task 1 使用刚清理后的 **97 episode / 10988 frame** 数据路径。

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper

export WORK_ROOT=/data1/wyn/piper-work
export DATASET_BASE=/data/datasets/May-pick-and-place
export PRETRAINED_PATH=/data/models/lerobot/pi05_libero_base
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export GPU_IDS=0,1
export NUM_PROCESSES=2
export BATCH_SIZE=16
export FIT_EPISODES=all
export EVAL_SPLIT=0.1
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'

export RUN_GROUP="piper-task1-direct-absolute-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
DRY_RUN=true bash examples/training/train_piper_direct.sh 1 absolute 0
DRY_RUN=false bash examples/training/train_piper_direct.sh 1 absolute 0
```

`GPU_IDS=2,3` 可改用卡 2、3。`BATCH_SIZE=16` 为每卡 batch，两卡全局 batch 32。
这里是每个任务独立训练 12000 步，不是四个任务合计 12000 步。`all` 按顺序训练四个任务。
Dry run 仅检查路径/元信息和打印命令；正式启动才审计原始 parquet、拟合统计并训练。

随后用相同的基础权重、seed、split、batch 和 horizon 跑相对动作组，务必更换输出路径：

```bash
export RUN_GROUP="piper-task1-direct-relative-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
DRY_RUN=true bash examples/training/train_piper_direct.sh 1 relative 0
DRY_RUN=false bash examples/training/train_piper_direct.sh 1 relative 0
```

不要从之前已拟合的 task checkpoint 开始这两组对照。入口复用 `dual_prompt_only`
这个已有的“无预训练、无 CABO”阶段配置名，但显式开启了动作映射和 expert 末两层训练；
因此输出目录名字中的 `dual_prompt_only` **不代表本次只训练 prompt**。

## 输出和判断

每次的任务目录（两组通过各自 OUTPUT_ROOT 隔离）：

```text
$OUTPUT_ROOT/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/
```

- `piper_normalization.json`：动作表示、参与统计的训练 episode、有效样本对数量和统计。
- `training_diagnostics.jsonl`：loss、实际学习率、梯度与参数更新量。
- `action_eval.jsonl`：每 500 步的训练/验证动作误差，另含初始化和最终评估。
- `action_eval_samples.json`：固定的 episode/frame，默认每个 split 128 个观测。
- `checkpoints/006000`、`009000`、`012000`：仅这三个完整模型保存点。
- `checkpoints/best_joint`、`best_gripper`：验证集前 8 步误差最小的**已保存模型**的链接。
  `last` 也只是链接，不额外保存权重。500 步评估不会额外保存模型。

W&B 默认开启，project 为 `piper-action-fit`。模型文件不上传 W&B；
`WANDB_MODE=offline` 可以只记本地曲线。优先看 `action_val/h1/joint_mae_deg`、
`action_val/h8/joint_mae_deg`、`action_val/h8/joint_p95_deg`、前 8 步夹爪误差，
以及 moving 子集与 hold 基线；也对照训练 split 是否仍拟合不足。
末尾 10% 的划分不是随机划分；记录采集顺序可能影响验证难度，两个实验需保持一致。

部署和离线评估都必须更新到这个 `piper` 版本，并携带完整 `pretrained_model`
目录（config、权重、前/后处理器 JSON 和统计 safetensors）。无需在部署命令手动打开
relative；保存的处理器会自动恢复绝对动作。配置与处理器模式或关节顺序不符时会报错。
先离线评估；硬件控制和碰撞保护没有改动。

## 后续单因素实验

要严格检验 priming 是否有影响，需要在相同清理后数据、训练统计和 12000 总预算上重跑
三阶段组。可继续用旧入口：`FLOW_STEPS=11000 SAVE_STEPS='[5000,8000,11000]'`
加 `--policy.piper_train_normalization=true`、`--policy.use_relative_actions=false`，
运行 `train_piper_retrain.sh 1 expert_last2 0`，使用独立输出组。
这时 750 priming + 250 bridge + 11000 flow = 12000；flow 5000/8000/11000 对应
总预算 6000/9000/12000，旧入口还会管理阶段衔接用的临时 checkpoint。

相对动作组若仍有较大短期误差，再单独试 `CHUNK_SIZE=16 N_ACTION_STEPS=8`，
保持其余配置不变。不要同时改变 horizon、学习率和解冻层数，否则无法归因。

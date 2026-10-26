# Piper：history20 + 完整 action head 微调

用于检查扩大动作侧可训练容量是否改善真实抓取。history20 未成功不能单独证明是参数量不足，
还需要比较留出示范的动作误差和相同初始布置下的真机结果。本组保留 128+128 prompt、
history20、绝对动作、chunk16/执行8、同一 base、seed 和 episode 划分，不从失败的 Piper 模型续训。

## 训练范围

| 模块 | 本组状态 | 默认结构的可训练参数 |
| --- | --- | ---: |
| 视觉编码器、VLM 主干及视觉投影 | 冻结 | 0 |
| VLM prompt（128×2048）+ action prompt（128×1024） | 训练 | 393,216 |
| Action expert 全部 18 个 block（含各层 AdaRMS） | 训练 | 424,783,872 |
| Expert 最后的 AdaRMS norm | 训练 | 3,148,800 |
| 两个时间 MLP | 训练 | 2,099,200 |
| Action 输入、输出投影 | 训练 | 66,592 |
| 未使用的 expert 文本 LM head、旧 inpainting role embedding | 冻结 | 0 |
| **合计** | | **430,491,680（约 430.49M）** |

计数来自 gemma_2b/gemma_300m 默认结构，不分配真实权重的 meta-device 实例；实际训练日志会重新
统计。这里是**冻结 VLM、完整微调连续动作 head**，不再是轻量 prompt-only。旧 `last18` 模式
仍冻结 final norm 和 time MLP，本组显式解冻它们。没有增加新权重；history 的 7+20+5 输入结构
不变，其复用的 `action_in_proj` 现在也参与学习。

新增配置 `train_action_head=true`，必须同时设置 `train_action_projections=true`、
`train_action_expert_last_n_layers=0`、`cabo_enabled=false`、`use_peft=false`。
`last_n_layers=0` 表示不使用局部解冻模式，**不表示本组 expert 冻结**。
旧模型缺省 `train_action_head=false`，旧 prompt-only/lastN 训练范围保持原样。

## 训练命令

训练机更新 `piper`，沿用已经验证的本地数据集和 tokenizer 路径：

```bash
git pull --ff-only origin piper
export WORK_ROOT=/data1/wyn/piper-work
export PRETRAINED_PATH=/data1/wyn/piper-work/models/pi05_base
: "${DATASET_BASE:?请设置本机 May-pick-and-place 的实际完整路径}"
: "${TOKENIZER_PATH:?请设置本机 paligemma tokenizer 的实际完整路径}"
export DATASET_BASE TOKENIZER_PATH
export RUN_GROUP="piper-task1-full-head-t128-history20-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"

GPU_IDS=0,1 MAIN_PROCESS_PORT=30328 BATCH_SIZE=32 \
PROMPT_LR=0.0001 PROMPT_FINAL_LR=0.00001 ACTION_EXPERT_LR_SCALE=0.1 \
GRIPPER_HISTORY_STEPS=20 FLOW_STEPS=12000 SAVE_STEPS='[6000,9000,12000]' DRY_RUN=false \
  bash examples/training/train_piper_action_head.sh 1 128 0
```

两个参数组都使用 warmup + cosine：prompts 为 `1e-4 -> 1e-5`，**整个 action head**（包括
投影、时间 MLP、final norm）为 `1e-5 -> 1e-6`。可用以上三个学习率变量覆盖。
原 `train_piper_prompt_only.sh` 默认学习率没有改变。

默认 batch32 是每卡，双卡总 batch64；显存允许时可显式改为 batch64，与之前双卡 history20
的全局 batch128 一致。完整 head 额外占用梯度/Adam 状态/通信内存，不沿用 prompt-only 显存估计。
如果比较不同 batch，要同时记录全局 batch 和处理过的样本数，不将差异全部归因于训练容量。
12,000 步 direct flow，无 priming/bridge/CABO；全量有效示范中留出 10% episode 验证。
`DRY_RUN=true` 只预检查并打印命令；默认端口 30328，同时启动另一组应换端口和输出目录。

输出目录为：

```text
$OUTPUT_ROOT/full-action-head-t128-b32-history20/
  pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/
    checkpoints/012000/pretrained_model/
```

`dual_prompt_only` 是底层选择 direct flow/no CABO 的历史 variant 名字；本组训练范围以
`train_action_head=true`、上层 `full-action-head` 目录标签和实际参数报告为准。
换 batch 时标签中的 b32 随之改变；新目录不会覆盖先前的 prompt-only 输出。

## 如何判断与部署

W&B/action_eval 保留，另记录 expert final norm、time MLP 的梯度和更新量。启动参数报告应显示
`action_expert_blocks`、`action_expert_norm`、`action_time_mlp`、`action_projection` 都大于零，
`other=0`。`training_diagnostics.jsonl` 记录完整 head 开关、history 长度和实际 trainable 数量。

主要比较留出集前 8 步关节 MAE、夹爪 MAE、运动/夹爪变化子集及预测曲线，再做真机抓取验证。
训练 flow loss 下降（包括低于 0.03）不保证闭环成功。历史仍是训练时的示范历史、部署时的自身
命令，二者的分布差异仍存在。

真机电脑也要 `git pull --ff-only origin piper`，使 PI05Config 识别新字段。完整复制
`pretrained_model` 下配置、权重、processor JSON 和统计量。沿用原 `deploy_piper_wyn.py` 命令，
先加 `--dry_run` 验证加载和合成推理，再测试真实输入；本次不修改关节限位或碰撞/CAN 保护。

# Piper：通用 pi05_base 与可训练范围对照

入口：`bash examples/training/train_piper_base.sh TASK [last2|last4] [SEED]`。
先在 task 1 跑 `last2`，与已有 absolute / horizon 16 的 LIBERO 基座结果比较；
再单独跑 `last4` 检验扩大动作专家训练范围的收益。这是新的实验，不保证抓取成功。

使用 LeRobot 已转换的 `lerobot/pi05_droid` 做 Franka/DROID 基座对照，见
[DROID 下载与训练说明](README_DROID_BASE.md) 和 `train_piper_droid.sh`。

已有基座上扩大到最后 8 层或全部 18 个专家层，见
[容量对照入口](README_CAPACITY.md)；该入口要求显式指定同一个初始 `PRETRAINED_PATH`。

## 基座和训练范围

官方区分通用预训练基座和进一步适配任务/机器人后的权重：

| 模型 | 来源与用途 | 本次选择 |
| --- | --- | --- |
| `lerobot/pi05_base` | 通用 π0.5 基座，包含机器人示范与多模态预训练，用于继续微调 | 下载其 LeRobot 格式权重，作为新的初始化 |
| `lerobot/pi05_libero_base` | 进一步针对 LIBERO 训练的 π0.5 权重 | 之前的对照组；不能理解成仅用仿真从零预训练 |
| OpenPI `pi05_droid` | 针对 DROID / Franka 的适配权重 | 不是 Piper 专用权重，也不能直接将 OpenPI 目录交给本加载器 |

来源：[LeRobot PI05 文档](https://huggingface.co/docs/lerobot/pi05)、
[pi05_base 模型卡](https://huggingface.co/lerobot/pi05_base)、
[OpenPI checkpoints](https://github.com/Physical-Intelligence/openpi#model-checkpoints)。
π0 和 π0-FAST 属于不同配置/动作建模方式，不能作为本 PI05 配置的直接替换。

新入口复用 direct-flow 方法，保留原始 action 标签，不做时间平移。按名称匹配 Piper
两路图像和 7 维状态/动作，从**训练 split** 重算统计；不套用基座的相机键或归一化统计。
基座只提供初始权重。默认训练配置如下：

| 配置 | `last2` | `last4` |
| --- | --- | --- |
| 动作 | absolute，预测 16 / 执行 8 | 相同 |
| 预算 / 保存 | 12000 次 flow 更新；6000、9000、12000 保存 | 相同 |
| Priming / bridge / CABO | 0 / 0 / 关闭 | 相同 |
| 数据划分 | 全部有效 episode；末尾 10% 留作验证；seed 0 | 相同 |
| 训练模块 | 两组 prompt、动作输入/输出投影、expert 末 2 层 | expert 改为末 4 层，其余相同 |
| 可训练参数 | 47,313,952（47.31M） | 94,512,160（94.51M） |
| 峰值 LR | prompt/投影 1e-4；expert 1e-5 | 相同 |
| 精度 / compile | FP32 / false | 相同 |

参数量对应 `gemma_2b` + `gemma_300m`、16+16 prompt、内部动作维度 32。
其中 prompt 为 49,152，投影为 66,592，每个 expert block 为 23,599,104；
最终以启动日志 `Trainable parameter report` 的实测计数为准。
此前的 direct `absolute` / `relative` 也训练了 47.31M，已经不是 prompt-only。
VLM、其余 expert block、最终 norm 和时间 MLP 仍冻结；`last4` 不等于全模型微调。
本分支显式控制可训练模块，不能用 `freeze_vision_encoder=false` 来绕过这一配置。

## 下载与训练

使用训练机现有环境，保留匹配的 Transformers 5.5.4；换权重不需要升级环境。
以下下载固定官方版本，约 14.5 GB，和旧权重使用不同目录。

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper

export WORK_ROOT=/data1/wyn/piper-work
export PI05_BASE_PATH="$WORK_ROOT/models/pi05_base"
HF_HUB_OFFLINE=0 python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/pi05_base",
    revision="b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
    local_dir=os.environ["PI05_BASE_PATH"],
)
PY

export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export GPU_IDS=2,3
export NUM_PROCESSES=2
export BATCH_SIZE=16
export FIT_EPISODES=all
export EVAL_SPLIT=0.1
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'
export CHUNK_SIZE=16
export N_ACTION_STEPS=8
export MASKED_STEPS=12
export RUN_GROUP="piper-task1-pi05-base-last2-h16-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DRY_RUN=true \
  bash examples/training/train_piper_base.sh 1 last2 0
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DRY_RUN=false \
  bash examples/training/train_piper_base.sh 1 last2 0
```

`PI05_BASE_PATH` 专门指定本实验的基座，会替换终端里残留的 `PRETRAINED_PATH`。
它必须包含官方 LeRobot 的 `config.json`、`model.safetensors` 和两个处理器 JSON。
仅将旧模型目录改名为 `pi05_base` 不会改变权重来源。正式训练严格检查权重键兼容性；
dry run 只验证路径/元信息和参数，不能证明模型加载或 GPU 训练成功。

两张卡的 `BATCH_SIZE=16` 是每卡 16、全局 32；不会把 12000 步再乘以卡数。
默认先跑 task 1；将任务编号改成 2/3/4，并更换 RUN_GROUP、OUTPUT_ROOT、LOG_ROOT
即可训练其他任务；`all` 会顺序训练四个独立模型。

`last2` 结束后，如需比较训练范围，保持上述环境，只更改 profile 和输出目录：

```bash
export RUN_GROUP="piper-task1-pi05-base-last4-h16-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DRY_RUN=false \
  bash examples/training/train_piper_base.sh 1 last4 0
```

两组都从同一 `pi05_base` 重新初始化；不要把 `last2` 训练后的权重当作 `last4` 的基座，
否则比较中还混入了额外训练预算。扩大范围可能改善拟合，也可能过拟合或增加显存占用。

## 输出与比较

任务 1 的最终模型为：

```text
$OUTPUT_ROOT/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

任务目录中的 `dual_prompt_only` 是复用的阶段配方名字，不表示只训练 prompt。
同目录保存 `action_eval.jsonl`、`action_eval_samples.json`、`training_diagnostics.jsonl`、
`piper_normalization.json`；W&B 默认记录曲线，项目 `piper-action-fit`。
`best_joint` / `best_gripper` 指向验证误差最小的已保存 checkpoint，不额外保存权重。

比较三组：LIBERO-last2-h16 → base-last2-h16 → base-last4-h16。保持数据、split、
seed、horizon、batch、预算和推理环境一致，核对 `action_eval_samples.json`。
关注前 1/8 步关节 MAE、P95、moving 子集和夹爪误差，同时检查各模块梯度/更新是否非零。
使用相同数据统计时的 loss 可以辅助判断，不能单靠最终 loss 判断抓取成功。
不能把旧 horizon 50 的结果当作只更换基座的严格对照。

如果训练集误差随 `last4` 降低、验证误差也降低，支持增加动作适配容量；
若离线误差已低但真机仍偏离，需要比较下发目标和实际关节反馈、相机视角及执行时序。
若扩大动作专家后视觉定位仍差，再单独设计 VLM/视觉侧适配实验。
本改动只涉及训练入口，不改变真机驱动、动作发送或碰撞保护。

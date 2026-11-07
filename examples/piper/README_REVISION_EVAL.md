# PrimingVLA Piper：配对真机试验计划与记录

`revision_trials.py` 只使用 Python 标准库，提供 `plan / record / summary` 三个命令。
它生成评测分配表、接收操作者对**已经执行的试验**的判定、汇总结果；不会连接相机或机器人，
也不会根据离线动作误差推断闭环成功。日志使用文件锁、追加写入和 `fsync`，适用于 Linux/macOS。

## 1. 先固定比较条件

主组默认是 `priming,direct_cpc,flow_warmup,priming_no_cpc`，每任务每模型固定 20 次试验。
可通过 `--methods` 增加 `direct_no_cpc,no_bridge,fixed_lr_0p1` 等组；这些名称只用于记录，
不会创建、训练或验证对应算法。模型必须来自实际完成并核验的训练。

- 统一初始化、示教集合、归一化、prompt 容量、累计更新次数和部署配置。
- 本轮 `flow_warmup` 沿用仿真修订实验：前 750 次为**无观察条件、50/50 全掩码的 action-only flow warm-up**；
  SAP 使用 40/50 掩码，保留部分干净动作作为轨迹上下文。二者后续 Bridge/Adaptation 保持一致。
  不应把该对照写成带视觉条件的 action-only warm-up。
- 4000 次累计更新不能只按 checkpoint 文件夹后缀判断：完整方法 `750+250+3000=4000`，
  Direct 为 4000 次直接训练。`--cumulative-updates` 默认 4000，所有被绑定模型必须一致。
- 当前 full action-head/history20 训练结果需与论文的 prompt-only 结果分开；方法名称不是参数预算核验。
- 固定输入是否有 proprioception、历史长度、相机/裁剪、action normalization、控制频率、chunk50、
  每次执行多少步后重规划、采样步数、时间限制和保护阈值。逐方法改变这些设置会引入额外因素。

任务与成功判据应在开始前固定。建议使用下表，并预先规定稳定保持时间、最长执行时间和目标区域边界：

| 任务 | 目标 | 不能算完整成功的情况 |
|---|---|---|
| T1 | 苹果放到黄色盘，释放并稳定停留 | 悬空、仍被夹爪抓住、掉到盘外 |
| T2 | 长方体离开蓝盘，落在预先规定的合法区域 | 只抓起、悬空、放到未规定的区域 |
| T3 | 网球由黄盘转到蓝盘，释放并稳定停留 | 转移中掉落、滚出目标盘 |
| T4 | 红方块放到黄色盘，释放并稳定停留 | 放到盘外或没有释放 |

`layout_id` 是人工场景编号。为每任务预先安排 20 个摆放，用台面标记、物体/盘坐标与朝向、初始照片来复位。
相同数值随机种子**不会自动重现真实物体位置**。保留全部失败及保护停止的视频和原因。

## 2. 生成固定计划

在仓库根目录运行；以下仅生成安排，没有执行记录：

```bash
EVAL_DIR=outputs/piper_revision_eval
uv run --no-project python examples/piper/revision_trials.py plan \
  --output "$EVAL_DIR/plan_seed0.json" \
  --methods priming,direct_cpc,flow_warmup,priming_no_cpc \
  --tasks T1,T2,T3,T4 --train-seeds 0 --trials-per-task 20 \
  --policy-seed 20261107 --order-seed 20261108
```

没有 `uv` 时可直接用 `python`，无需安装 LeRobot、PyTorch 或机器人 SDK。
此计划共 320 个有效试验名额。计划文件已存在时命令拒绝覆盖。
如正式比较三个独立训练 seed，在开始该计划前使用 `--train-seeds 0,1,2`，对应 960 个名额。
一个训练模型用多个推理随机种子重复执行，不能代替独立训练 seed。

按 `trials[].order` 执行。每个布局内的方法与训练 seed 组合顺序经过确定性随机化；
同一布局在所有方法、所有训练 seed 下共用一个 `policy_seed`。
`train_seed`、`policy_seed`、`order_seed` 分别表示训练、推理噪声和执行顺序，互不替代。
更改 `order_seed` 只改变执行顺序，不改变各布局的推理随机种子。

### 绑定实际模型

计划可以在训练前生成，此时每行 `model` 为 `null`。**首次录入某个 method/task/train_seed 的试验时必须提供实际模型身份**，
之后同组试验自动复用。一个计划中的同组不能中途换 checkpoint 或训练 manifest。

也可在生成计划时使用 `--models models.json`。文件为 JSON 数组，每项结构如下；
下面的占位文本必须替换为真实路径与数值，不能原样用于正式试验：

```json
[
  {
    "method": "priming",
    "task_id": "T1",
    "train_seed": 0,
    "checkpoint": "/实际模型目录或导出的模型文件",
    "checkpoint_sha256": "实际模型载荷的64位SHA256",
    "model_manifest": "/实际训练目录/revision_manifest.json",
    "model_manifest_sha256": "该训练manifest文件的64位SHA256",
    "cumulative_updates": 4000
  }
]
```

允许只预绑定已经完成的模型，其余模型在首次记录时绑定。SHA256 缺省时工具会尝试读取本地文件计算：
checkpoint 是文件则读取该文件，是目录则读取其中 `model.safetensors`。对于 compact bundle 等其他格式，
使用训练导出的准确载荷摘要，例如 `revision_manifest.json` 的 `checkpoints[].model_sha256`。
显式提供的摘要被当作已核验的身份记录，不会每次重新读取数 GB 权重。
在评测前保存最终 manifest 的不可变副本；保持哈希与实际部署的模型一致。

## 3. 只记录已观察到的结果

从计划取本次实际执行的 `trial_id`。以下是**命令模板**：`ACTUAL_*` 必须先填入本次真实结果和路径，
不能把示例值当作执行证据：

```bash
uv run --no-project python examples/piper/revision_trials.py record \
  --plan "$EVAL_DIR/plan_seed0.json" --log "$EVAL_DIR/trials_seed0.jsonl" \
  --trial-id "$ACTUAL_TRIAL_ID" --outcome "$ACTUAL_OUTCOME" \
  --duration-seconds "$ACTUAL_DURATION_SECONDS" \
  --checkpoint "$ACTUAL_CHECKPOINT" \
  --checkpoint-sha256 "$ACTUAL_MODEL_SHA256" \
  --model-manifest "$ACTUAL_TRAINING_MANIFEST" \
  --cumulative-updates 4000 \
  --video-path "$ACTUAL_VIDEO" --initial-photo-path "$ACTUAL_INITIAL_PHOTO" \
  --initial-pose "$ACTUAL_POSE_DESCRIPTION"
```

本地 manifest 可读时自动计算其 SHA256，否则另加 `--model-manifest-sha256`。
同组的后续记录可以省略所有模型身份参数。每次写入会校验整份已有日志，不会忽略损坏或冲突行。

| `--outcome` | 计入有效分母 | 额外要求 |
|---|---:|---|
| `success` | 是，成功 | 实际满足预先规定的完整任务判据 |
| `failure` | 是，失败 | `--failure-category missed_grasp/drop_in_transport/wrong_target/release_failure/timeout/other`；other 还需 `--reason` |
| `policy_stop` | 是，失败 | `--reason` 描述策略导致的保护停止；自动分类为 policy_safety_stop |
| `invalid_hardware` | 否，保留待完成名额 | `--reason` 描述独立的设备故障；不能用它替代策略失败 |

时长必须有限、非负；有效试验必须大于 0 秒。策略失败、超时和保护停止不允许重新录入覆盖。
工具不能自动判断故障责任，操作者须根据实际视频/日志区分设备故障与策略行为。

### 硬件无效后的显式重试

每条记录会打印 `attempt_id`，例如 `T1-L001-s0-priming.a1`。
只有最近一次结果为 `invalid_hardware`，才能在**重新实际执行后**录入重试，并添加：

```bash
--retry-of "$ACTUAL_PREVIOUS_INVALID_ATTEMPT_ID"
```

重试保留同一 trial/layout/model/policy seed，得到 `.a2` 等新尝试编号，并链接旧尝试。
旧记录保留；硬件无效不会凭空变成成功，也不会增加预定有效试验名额。
重复首次记录、同一试验重复成功、重试非最近一次无效尝试都会被拒绝。

## 4. 汇总与论文口径

```bash
uv run --no-project python examples/piper/revision_trials.py summary \
  --plan "$EVAL_DIR/plan_seed0.json" --log "$EVAL_DIR/trials_seed0.jsonl" \
  --output "$EVAL_DIR/summary_seed0.json"
```

日志尚未生成时，全部显示 pending。输出包括：

- 计划名额、有效完成数、未开始数、硬件无效待重试数、实际尝试次数、保护停止次数。
- 每 method/task/train_seed 的 `successes/completed`、`completed/planned`、成功率和 Wilson 95% 区间。
- `per_method_seed` 的四任务等权宏平均和单独的 pooled success rate。
- `per_method` 的宏平均、pooled rate，以及所有计划完成且至少两个独立训练 seed 时的均值和样本标准差。

pending 既不计成功，也不计失败；已完成部分的 SR 始终连同完成进度展示。
宏平均只在对应 seed 的全部计划任务试验完成后报告，避免只平均已完成的容易任务。
跨 seed 均值/标准差只在所有计划 seed 完成后报告，避免选择性漏掉未完成的 seed。
Wilson 区间描述单任务/训练 seed 内的试验结果，不代表训练 seed 之间的变异，且未校正同布局或连续执行造成的相关性。
四任务试验数相同时，等权宏平均与 pooled rate 数值一致；两者仍分别命名。

原始 `plan*.json` 与 `trials*.jsonl` 是记录依据，不能用汇总文件覆盖。
工具拒绝损坏的 JSONL、冲突 success/outcome 和计划校验值变化；发生误录时应保留原文件，
按明确的人工更正记录核对后另建经过审计的数据版本，不应删除失败行后继续计算。

## 本地验证

```bash
uv run --no-project python tests/scripts/test_piper_revision_trials.py
```

测试只使用临时文件和明确标记的模型占位数据，不会生成真实实验结果，也不会连接机器人。

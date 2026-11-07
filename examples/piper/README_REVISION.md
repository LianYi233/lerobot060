# PrimingVLA 真机补实验：单张 A800、300G 数据盘

> **当前 reality 数据的长训练请使用 [四任务长训练指南](README_REALITY_LONG.md)**：`train_piper_revision_long.sh` 默认按 **2 → 3 → 4 → 1** 为每个任务独立训练 `priming` 模型，累计 **50,000** 步。本文保留审稿受控实验的 **core / 4,000 步默认设置**；两种预算的结果分别记录。

本入口针对审稿中的三个主要问题补充可比较的真机训练：提升是否来自额外更新次数、SAP 是否优于普通动作侧预热、CPC 是否只是较小的动作侧学习率。最终证据仍须来自固定协议下的闭环成功率；离线动作误差和训练完成本身不等于真机成功。

入口为：

```bash
bash examples/training/train_piper_revision.sh GROUP SEEDS --tasks TASKS
```

`GROUP` 可为 `core / extended / cpc_sweep / all`，也可为逗号分隔的方法名称；`SEEDS` 为独立训练种子，如 `0` 或 `0,1,2`；任务编号为 `1,2,3,4`。队列在一张 GPU 上逐个训练，不启动机器人，也不使用 W&B。

## 1. 方法与审稿问题对应

默认总预算为 **4000 次累计优化更新**。准备阶段的 1000 次已经包含 750 次 SAP 和 250 次 Bridge，不额外再加 250 次。

### 首先完成的 core 四组

| 方法名称 | Stage I | Bridge | Adaptation / 直接训练 | CPC | 回答的问题 |
|---|---:|---:|---:|---|---|
| `priming` | SAP 750 | 250 | 3000 | 联合阶段开启 | 完整方法 |
| `direct_cpc` | 0 | 0 | 直接训练 4000 | 开启 | 整体分阶段流程是否优于同预算直接训练 |
| `flow_warmup` | 全掩码动作 flow 750 | 250 | 3000 | 联合阶段开启 | SAP 的轨迹上下文是否优于普通动作侧预热 |
| `priming_no_cpc` | SAP 750 | 250 | 3000 | 全程关闭 | 在相同课程下隔离 CPC 的作用 |

**`flow_warmup` 沿用仿真修订对照：Stage I 不输入观察条件，50/50 个动作步全部掩码，使用 action-only flow 目标。** SAP 为 40/50 掩码，保留部分干净动作作为轨迹上下文。两组 Stage I 都只训练 action prompt；后续 Bridge/Adaptation 设置一致。不要把这个对照写成带视觉条件的 action-only warm-up。

`direct_cpc` 与完整方法比较的是整个训练流程，包含阶段和优化器状态策略；不能仅凭这一对就把差异全部归因于 SAP。SAP→Bridge 延续 AdamW 和 scheduler；准备阶段→Adaptation 重置二者。本入口没有实现四种 optimizer/scheduler carry/reset 组合。

### 其余六组与组名

| 方法名称 | 分阶段更新数 | 联合阶段 action-prompt 更新控制 | 所属组 |
|---|---|---|---|
| `direct_no_cpc` | 直接 flow 4000 | CPC 关闭，倍率 1 | `extended` |
| `no_bridge` | SAP **1000** + Bridge 0 + Adaptation 3000 | CPC 开启 | `extended` |
| `fixed_lr_0p5` | SAP 750 + Bridge 250 + Adaptation 3000 | CPC 关闭，固定倍率 **0.5** | `cpc_sweep` |
| `fixed_lr_0p1` | 同上 | CPC 关闭，固定倍率 **0.1** | `cpc_sweep` |
| `fixed_lr_0p03` | 同上 | CPC 关闭，固定倍率 **0.03** | `cpc_sweep` |
| `fixed_lr_0p01` | 同上 | CPC 关闭，固定倍率 **0.01** | `cpc_sweep` |

`all` 包含上述十种方法。`no_bridge` 将移除的 250 次 Bridge 返还给 SAP，保持总预算；它不是少训练 250 次的版本。

固定学习率倍率**只作用于 Bridge/Adaptation 的 action prompt**，Stage I 保持原始学习率；VLM prompt 的基础学习率与其他设置不变。CPC 对应代码中的 `cabo_enabled`，使用 `rho=2` 的自适应相对更新控制。这里没有解冻 action head，因此不应将对照描述为“降低已训练 action head 的学习率”。固定倍率应在验证数据上预先选择，再进入同一套真机测试；不能按最终测试成功率挑倍率。

**这十组都不是 LoRA 或 PhaseLoRA 的实现。** 如有经过初始化、参数量和训练配置核验的真实 LoRA/PhaseLoRA baseline，可作为额外比较加入论文。已有 full-action-head/history20 实验应单列为容量对照，其结果不能充当本入口约 0.05M 参数的主结果。

## 2. 本入口固定的训练口径

| 项目 | 实际设置 |
|---|---|
| 每任务模型 | 四个任务分别训练；任务数据目录中各含一个任务 |
| 可训练参数 | VLM prompt `16×2048` + action prompt `16×1024` = **49,152** |
| 冻结部分 | 视觉/VLM、整个 action expert、动作投影、时间 MLP、norm；不启用 LoRA |
| 观察条件 | 联合阶段使用两路 RGB、语言、当前状态的离散 token；**不启用 history20** |
| 动作 | 7 维 absolute action；预测 **50** 步，默认执行 **8** 步后重规划；flow 推理 **10** 步 |
| Batch | 默认单卡 **8**；所有比较组必须相同 |
| 数值/内存设置 | `policy.dtype=float32`；gradient checkpointing 开启；compile 关闭 |
| 基础优化设置 | AdamW；基础 LR `2.5e-5`，配置 warmup 1000、decay steps 30000、decay LR `1e-5`；实际阶段长度会触发原 scheduler 的自动缩放 |
| 数据划分 | 每任务最后 `ceil(episode数×0.1)` 个完整 episode 留作验证，其余训练 |
| 归一化 | 仅使用训练 split 重新计算状态/动作 QUANTILES 统计；图像增强关闭 |
| 保存与评估 | 默认保存累计 **3000 / 4000** 更新的 checkpoint；固定验证样本做离线动作评估，不按真机测试结果挑 checkpoint |

episode 数以服务器的 `meta/info.json` 和训练实际输出的 `dataset_split.json` 为准，不预设每任务恰好 100 条。训练/验证按完整轨迹划分，准备阶段和 Adaptation 使用同一划分；本入口还会核对各比较组的实际划分和归一化记录一致。

在默认预算下，原 scheduler 将准备阶段的 warmup/decay 缩放到 **33/1000**，Adaptation 缩放到 **100/3000**，Direct 缩放到 **133/4000**。这保留原代码的阶段优化策略，并不表示 Direct 与分阶段方法拥有逐步相同的 LR 轨迹。`priming`、`flow_warmup`、`priming_no_cpc` 和固定倍率组使用相同阶段长度，适合在相同优化日程下比较对应机制；不同阶段的实际 LR 可从训练日志核查。

在默认 4000 总预算下，有准备阶段的方法其 flow 目录 `002000 / 003000` 对应累计 `3000 / 4000`；Direct 的 flow 目录 `003000 / 004000` 对应累计 `3000 / 4000`。恢复和比较时优先使用统一命名的 `bundles/total-003000 / total-004000`，避免按 flow 目录后缀误算训练量。

新定量实验的状态输入、50/8 执行配置应如实写进论文。旧定性动作图中“不含 proprioception、每次执行 10 步”的描述不适用于本入口。真实控制频率沿用对应数据 metadata 的 fps，不能把相机采集帧率当成动作执行频率。

## 3. 新 AutoDL 服务器：先填写本机路径

使用已经安装好本分支依赖的 **Python 3.12+ LeRobot 环境**，在该仓库的 `piper` 分支根目录运行。下面所有模型、数据和 tokenizer 路径都是配置示例，必须替换为这台新服务器上真实存在的路径；不会从另一台服务器自动复制数据。

```bash
git pull --ff-only origin piper

export WORK_ROOT=/root/autodl-tmp/piper-revision
export DATASET_BASE=/替换为本机数据路径/May-pick-and-place
export PRETRAINED_PATH=/替换为本机原始模型路径/pi05_base
export TOKENIZER_PATH=/替换为本机tokenizer路径/paligemma-3b-pt-224

export RUN_GROUP=piper-review-4k-v1
export OUTPUT_ROOT="$WORK_ROOT/runs/$RUN_GROUP"
export GPU_ID=0
export NUM_PROCESSES=1
export BATCH_SIZE=8
export TOTAL_UPDATES=4000
export CUMULATIVE_CHECKPOINTS='[3000,4000]'
export EVAL_SPLIT=0.1
export N_ACTION_STEPS=8
```

`DATASET_BASE` 下应为已有的四个任务目录：

| 编号 | 目录 |
|---|---|
| 1 | `1-put_the_apple_on_the_yellow_plate` |
| 2 | `2-remove_the_cuboid_from_blue_plate` |
| 3 | `3-move_the_tennis_from_yellow_plate_to_blue_plate` |
| 4 | `4-pick_the_red_cube_into_the_yellow_plate` |

`PRETRAINED_PATH` 必须显式设置，指向所有方法共同使用的**原始**本地 `pi05_base` 或明确选择的 `pi05_libero_base`。应包含模型权重、config 和两份 processor 配置；不能指向已经训练过 Piper head/history/prompt 的模型。若比较两种初始化，分别使用新的 `RUN_GROUP` 和 `OUTPUT_ROOT`，在每一种初始化内部完成同条件比较。

本入口强制单 GPU、单训练进程、离线 HF 模式、关闭 W&B。临时文件与缓存放到 `WORK_ROOT`，训练日志放到 `OUTPUT_ROOT/logs`。单卡仍需反向传播经过冻结网络，因此 49,152 个可训练参数不意味着显存只有 prompt 权重大小；以实际 A800 型号/容量和训练峰值为准。

## 4. 先检查 T1，再串行展开

### 第一步：仅检查路径、数据元信息与将要执行的命令

```bash
DRY_RUN=true bash examples/training/train_piper_revision.sh core 0 --tasks 1
```

它打印四组的训练参数、预计保存点、实际可用磁盘和空间估计，不加载模型、不执行 GPU 训练、不创建实验输出。这个检查不能证明显存足够或训练能够完成。

### 第二步：正式运行 T1 的四组

```bash
DRY_RUN=false bash examples/training/train_piper_revision.sh core 0 --tasks 1
```

先核对训练正常、49,152 个参数审计通过、数据/归一化一致、模型能恢复和离线推理。T1 的比较仍按预定预算执行；不依据测试成功率只延长某一种方法。

### 第三步：四任务 seed0

保持同一代码、数据、初始化及环境设置，在 T1 队列完成后运行：

```bash
DRY_RUN=false bash examples/training/train_piper_revision.sh core 0 --tasks 1,2,3,4
```

共 **16 个任务模型**。已完成的 T1 四组在核对实验签名、共享重建文件和 bundle 摘要后显示 `SKIP completed`，继续剩余任务；不会仅凭目录存在就跳过。

需要三次独立训练时，同一协议下执行：

```bash
DRY_RUN=false bash examples/training/train_piper_revision.sh core 0,1,2 --tasks 1,2,3,4
```

共 **48 个任务模型**，已核验的 seed0 结果可继续跳过。不要同时在两个终端启动同一输出组的训练队列。

扩展机制组可随后加入：

```bash
DRY_RUN=false bash examples/training/train_piper_revision.sh extended 0 --tasks 1,2,3,4
DRY_RUN=false bash examples/training/train_piper_revision.sh cpc_sweep 0 --tasks 1,2,3,4
```

若只执行指定组，第一位置参数也支持 `priming,fixed_lr_0p1` 等逗号分隔的已实现方法。

### 进度、异常与状态

启动终端同时显示原始训练输出，每约 30 秒补充当前队列位置、阶段、累计进度、已用时间和按当前阶段速度估计的 ETA。SAP 与联合训练的速度可能不同，因此 ETA 是动态估算。

另一个终端设为同一 `OUTPUT_ROOT` 后，可查看记录：

```bash
bash examples/training/train_piper_revision.sh --status
tail -f "$OUTPUT_ROOT/logs/task1-priming-seed0-total4000.log"
```

`--status` 的累计更新列是该 run 的**目标预算**；实际运行进度看启动终端或 `training_diagnostics.jsonl`。状态表另有 `latest_adaptation_loss / trailing_mean / window_updates / target_met` 列，定义见 [长训练指南的 loss 监控说明](README_REALITY_LONG.md#3-如何判断-loss-是否接近-1e-4)。完成记录包含训练进程 wall-clock、训练 step 期间的最大 CUDA allocated 显存、参数审计、数据划分、归一化、源代码/模型身份和 checkpoint 摘要。该显存指标不包含模型初始化和离线验证，也不是 `nvidia-smi` 的总占用；压缩和完成后审计时间不包含在 `training_wall_seconds` 中。没有预先承诺一张 A800 需要多少小时，先以 T1 的实测吞吐安排剩余队列。

**当前启动器不会自动恢复未完成的 run。** 中断或失败的目录、日志和 checkpoint 会保留；再次遇到该 run 会停止，避免覆盖或把部分训练误算为完成。需要先检查失败原因与保留状态，再明确安排恢复或使用新的 `RUN_GROUP`/`OUTPUT_ROOT` 重做。更改代码、数据、batch 或预算后，也不能继续混入已有输出组。

### 如果 4000 次不足以适配真机

4000 次是受控比较的起始预算，不保证足以学会 Piper 的动作语义。若训练/验证表明需要更长预算，可将**整组**统一改为 12000 次，并使用新的输出组；完整方法仍为准备 1000 + Adaptation 11000，Direct 为 12000。

```bash
export RUN_GROUP=piper-review-12k-v1
export OUTPUT_ROOT="$WORK_ROOT/runs/$RUN_GROUP"
export TOTAL_UPDATES=12000
export CUMULATIVE_CHECKPOINTS='[4000,12000]'

DRY_RUN=true bash examples/training/train_piper_revision.sh core 0 --tasks 1
DRY_RUN=false bash examples/training/train_piper_revision.sh core 0 --tasks 1,2,3,4
```

显式设置过 `OUTPUT_ROOT` 后，仅更改 `RUN_GROUP` 不会自动改写该路径，所以示例同时设置两者。额外保存点也会增加下一次训练的峰值磁盘需求。不能用 12000 次的完整方法直接对比 4000 次的 baseline 后，将差异全归因于方法。

## 5. 300G 磁盘：共享文件只用于无损重建

启动前检查的是 `OUTPUT_ROOT` 所在文件系统的**实际剩余空间**，不是把总容量 300G 当成可用 300G。每个 run 启动前的保守估计为：

```text
(2 × 原始 model.safetensors 大小 + 64 MiB)
× (累计 checkpoint 数量 + 是否有准备阶段 checkpoint + 1)
+ 30 GiB 余量
```

其中 2 倍用于覆盖输入 BF16、保存 FP32 的情况，额外一份覆盖打包/共享文件。默认余量可由 `RESERVE_GIB` 设置；磁盘不足时队列拒绝开始下一 run，不会删旧实验腾空间。训练期间也会检查可用空间，低于 5 GiB 时停止当前训练并保留文件。增加保存点或保留更多全量文件时要重新核对实测空间。

默认输出结构的重要位置如下：

| 位置 | 用途 |
|---|---|
| `$OUTPUT_ROOT/protocol.json` | 共同初始化、batch、预算、推理设置与代码记录 |
| `$OUTPUT_ROOT/task_protocols/` | 各任务数据身份及实际 split/normalization 协议 |
| `$OUTPUT_ROOT/records/` | 每个 run 的 running/completed/failed 状态 |
| `$OUTPUT_ROOT/storage_base/model.safetensors` | **仅用于重建的共享权重文件** |
| `task1-priming-seed0-total4000/revision_manifest.json` | 此模型的审计、耗时和各 checkpoint 身份 |
| `task1-priming-seed0-total4000/bundles/total-003000` | 累计 3000 次的无损存储包 |
| `task1-priming-seed0-total4000/bundles/total-004000` | 累计 4000 次的无损存储包 |
| 同一 run 的 `bundles/preparation-001000` | 有准备阶段时，保存其结束点以便核查 |

共享 `storage_base` 来自**第一个训练完成的 run 的已保存 checkpoint**，默认是其累计 3000 点。它可能已经包含某任务学到的 prompt；它的角色只是压缩其他 checkpoint 时的字节参照。

**所有新训练始终从原始 `PRETRAINED_PATH` 开始。绝不能把 `storage_base` 作为训练初始化。** 存储共享不表示后面的任务继承前面任务的训练权重。

每个 bundle 保存相对共享文件的差异和该 checkpoint 的配置/处理器等文件。打包时实际流式重建模型并核对原始 SHA256；不假定只有 prompt 字节发生变化。默认只有在本次 run 的所有保留点都通过重建校验后，才移除**本次新 run 自己生成的**重复 `model.safetensors`，留下 `COMPACT_CHECKPOINT.json` 指向恢复位置。配置、processor、optimizer/RNG 状态、日志和原始训练基座均保留。其他实验的 checkpoint 不在清理范围内。

因此 compact bundle 不是可直接传给部署脚本的完整模型目录，必须先 `materialize`。它也不能替代训练恢复所需的 `training_state`。不要丢失或独立覆盖共享 `storage_base`；其 SHA256 是所有相关 bundle 的恢复条件。若设置 `KEEP_FULL_CHECKPOINTS=true`，会同时保留本轮全量权重，空间占用明显增大，需在新的输出组中统一配置。

## 6. 恢复一个模型并先做无硬件推理

以 T1、`priming`、seed0、累计 4000 次为例，在保存这些结果的机器上执行：

```bash
PIPER_DEPLOY_MODEL="$WORK_ROOT/materialized/task1-priming-seed0-total4000"

python examples/piper/revision_checkpoint.py materialize \
  --bundle "$OUTPUT_ROOT/task1-priming-seed0-total4000/bundles/total-004000" \
  --base "$OUTPUT_ROOT/storage_base" \
  --output "$PIPER_DEPLOY_MODEL"
```

恢复到新的目录，重新核对完整模型 SHA256，并带回对应的配置、processor 和 `dataset_info.json`。目标目录已存在时拒绝覆盖。需要在机器人电脑恢复时，复制一次对应 `storage_base`、所需 bundle 和训练 manifest，并将上面的路径换成该电脑的实际路径；不要把另一台服务器的绝对路径直接用于部署。

在具备本分支推理环境的机器上，先执行以下**不连接硬件**的检查。`TOKENIZER_ON_ROBOT` 应为该机器真实存在的本地目录：

```bash
TOKENIZER_ON_ROBOT=/替换为当前机器的tokenizer路径/paligemma-3b-pt-224

HF_HUB_OFFLINE=1 python deploy_piper_wyn.py \
  --policy_path "$PIPER_DEPLOY_MODEL" \
  --tokenizer_path "$TOKENIZER_ON_ROBOT" \
  --dataset_info "$PIPER_DEPLOY_MODEL/dataset_info.json" \
  --task "put the apple on the yellow plate" \
  --steps_per_chunk 8 --num_inference_steps 10 --no_compile \
  --dry_run
```

`DRY_RUN_OK` 表示完整模型与处理器加载并完成合成观测推理，不是任务成功率。本补实验训练队列不会转入真机执行；真实操作继续按 [已有部署说明](README.md) 和 `piper_deploy_guard.py` 的现场检查/故障处理流程进行，本次没有更改或绕过该保护逻辑。正式试验按计划逐次设置 `--seed`，统一相机映射、控制频率和动作执行步数。

## 7. 闭环结果如何补入论文

使用 [配对试验记录说明](README_REVISION_EVAL.md) 的 `revision_trials.py` 固定布局、训练 seed、推理 seed 和方法顺序，逐次记录真实 outcome、视频和模型身份。四任务、core 四方法、每任务每模型 20 次，对应 **320 次闭环试验/训练 seed**；三个独立训练 seed 为 **960 次**。

每任务报告成功数/尝试数及区间；总体分别报告任务等权宏平均与 pooled rate；训练 seed 的均值/标准差单列。没有执行的 trial 保持 pending，策略导致的保护停止计失败，独立硬件故障须保留原因及显式重试链接。不能用离线验证误差替代这些记录，也不能把同一个训练模型的重复 rollout 当成多个训练 seed。

论文至少应交代真实示教数与划分、初始化身份、49,152 个参数、累计更新预算、部署设置、实际训练时长/显存、各任务闭环次数和失败类型。LoRA/PhaseLoRA 的额外结果只有在实际实现并完成同条件训练/评测后才能填入；本入口不会生成这些结果。

### 验证范围

配套测试覆盖 CPU 上的配置/预算构造、固定 action-prompt LR 行为、checkpoint 无损存储与摘要核对、配对试验记录和统计。它们不等于在用户的 A800 上完成了训练，也不等于机器人闭环验证。本次新增代码没有执行实际训练或机器人动作。

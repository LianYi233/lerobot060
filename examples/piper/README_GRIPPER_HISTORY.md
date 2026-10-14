# Piper：当前状态 + 过去 20 步夹爪 action 条件

这是可选的输入条件实验，不能保证解决 loss 平台或抓取失败。建议与相同 base、数据划分、seed、
batch、学习率和 prompt 数量的无历史版本比较。

## 维度与参数量

对当前观测时刻 `t`，按以下顺序拼成一个向量：

| 索引 | 内容 | 维度 |
| --- | --- | --- |
| `0:7` | 当前实测状态：6 个关节 + 1 个夹爪，顺序为保存的 `state_feature_names` | 7 |
| `7:27` | `t-20, ..., t-1` 的夹爪 action，从旧到新 | 20 |
| `27:32` | 归一化后再补入的字面值零 | 5 |

张量为 `[B,32]`。复用现有冻结的 `action_in_proj`（默认结构为 `32 -> 1024`），
得到**一个** `[B,1,1024]` 条件 token，拼在 action expert 输入前面：

`[状态/历史条件 token, action prompt tokens, noisy action tokens]`。

这是 1 个条件 token，不是增加 20 个可训练 prompt。它不加 flow 噪声、不作为预测输出、不计入
action loss。它只读取自身；action prompt 和 action 查询可以读取它。VLM 前缀不读取它，
所以仍可缓存。原本语言 token 中的当前状态也保留。

没有新增权重，严格 prompt-only 启动器的可训练参数仍为：

- VLM prompt：`128 × 2048 = 262,144`。
- Action prompt：`128 × 1024 = 131,072`。
- 合计 **393,216**；VLM、action expert、action 输入/输出投影、time MLP 均冻结。

冻结的 action 投影并非专门为状态/历史向量预训练，因此这种复用是否有效需要实验验证。

## 训练与部署的数据约定

- 训练仅取同一 episode 中当前帧之前的 `action[..., gripper_index]`，不含当前或未来标签。
  不修改未来 action 标签窗口，不做标签平移。只缓存数值列，不额外解码历史视频，也不改数据集文件。
- episode 开头不足 20 步时，用**该 episode 首帧实测夹爪状态**向左补齐，不用 `action[0]` 或零。
  不跨 episode、不跨训练/验证集；支持打乱 batch。
- 当前状态使用保存的 STATE 统计量；夹爪历史使用训练集 ACTION 统计量中的夹爪坐标。
  新处理器把统计量随 checkpoint 保存，加载时核对它与目标 action 的归一化是否一致。
- 数据列叫 `action` 并不证明它是采集时的真实命令。如果采集程序保存的是实测关节位置，
  这里就是历史夹爪位置，不能重建原本未记录的命令。应核实采集语义后再归因。
- 数据集 10 Hz 下，20 步覆盖 2 秒。部署按策略控制步记录，推理等待会延长实际墙钟时间。

## Task 1：128+128、reference 学习率训练命令

在训练机的 `piper` 分支、训练环境中执行：

```bash
git pull --ff-only origin piper

export WORK_ROOT=/data1/wyn/piper-work
export PRETRAINED_PATH=/data1/wyn/piper-work/models/pi05_base
# 沿用本机已经通过预检查的数据集和 tokenizer 路径；不能从 /data1 推断更深层目录。
: "${DATASET_BASE:?请设置本机 May-pick-and-place 的实际完整路径}"
: "${TOKENIZER_PATH:?请设置本机 paligemma tokenizer 的实际完整路径}"
export DATASET_BASE TOKENIZER_PATH
export RUN_GROUP="piper-task1-t128-history20-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"

GPU_IDS=0,1 MAIN_PROCESS_PORT=30228 BATCH_SIZE=64 \
GRIPPER_HISTORY_STEPS=20 FLOW_STEPS=12000 SAVE_STEPS='[6000,9000,12000]' \
PROMPT_LR=0.0001 PROMPT_FINAL_LR=0.00001 DRY_RUN=false \
  bash examples/training/train_piper_prompt_only.sh 1 128 0
```

12,000 次 direct flow 更新，无 priming/bridge/CABO；绝对 action，chunk 16 / 执行 8，
按 episode 留出 10% 验证集。Batch 为**每卡 64**，双卡全局 128。保存 6000、9000、12000。
W&B 和已有 action 误差评估继续启用。命令中显式指定 reference 学习率 `1e-4 -> 1e-5`；
若不指定，原启动器的默认值仍为 `5e-5 -> 5e-6`。

`WORK_ROOT` 同时指定默认临时/缓存目录：`tmp`、`cache/torchinductor`、`cache/triton`、
`cache/huggingface`。`OUTPUT_ROOT` 只指定模型输出位置，不能代替它。
Piper 启动链默认使用本地服务器 `/data1/wyn/piper-work`。统一初始化脚本
`examples/training/piper_runtime_paths.sh` 会将遗留的 `/root/autodl-tmp` 工作目录和
临时/缓存环境变量切换到此目录，并打印迁移提示；其他显式自定义路径保留。
启动时会打印最终路径，dry-run 也会打印但不创建目录。

最终 checkpoint 在：

```text
$OUTPUT_ROOT/prompt-only-t128-b64-history20/
  pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/
    checkpoints/012000/pretrained_model/
```

用原始 base 和新输出目录训练，不把旧无历史训练状态当作同一实验 resume。
`GRIPPER_HISTORY_STEPS=0` 关闭此功能，保留原模型行为和输出目录格式。
`DRY_RUN=true` 仅检查并打印启动命令，不训练。

## 离线评估与真机部署

两台电脑都更新到此分支，沿用 `eval_piper_offline.py`、`deploy_piper_wyn.py` 命令，
把 `--policy_path` 换成新模型的完整 `pretrained_model` 目录即可，不需新增部署参数。
配置中的 `piper_gripper_history_steps=20` 与新处理器自动生效。
复制模型时必须连同 processor JSON 和统计量 safetensors 一起复制，不能只给旧配置补字段。

训练中的 action_eval 和离线评估都使用**示范记录的历史 action（teacher forcing）**，
并非用预测递推的闭环成功率。比较已有前 8 步关节误差、夹爪误差及实际闭合/抓取表现，
不能只以总 flow loss 判断成败。

部署每个 episode 重建历史，用首次实测夹爪状态初始化。只有成功提交的策略控制步才入队，
并应用与驱动一致的夹爪限幅/整数舍入；不把 chunk 中尚未执行的动作放进去。
提交成功不代表机械臂已达到目标。100 Hz 保活不推进 10 Hz 策略历史；启用该实验时，
保活重复上一条已发送的夹爪目标，避免用尚未闭合到位的实测开度打断闭合。
关节保位、故障保护继续保留。合成 `--dry_run` 显式提供零历史，不连接机械臂。

部署用模型自身已发送的命令，而离线用示范历史，两者有分布差异。若数据集 action 实际是
实测位置，还存在位置/命令语义差异，需要在真机结果解释中考虑。

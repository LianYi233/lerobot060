# 原始 PI05 路径上的 VLM-only prompt 实验

分支：`vlm-only`。分别从 `pi05_libero_base`、`pi05_libero_finetuned_v044` 开始，
只训练 VLM prompt。默认把“一个 prompt”解释为 **1 个可学习 token**；
如果指的是一组 prompt，可设置 `VLM_PROMPT_TOKENS=16` 或其他正整数。

| 设置 | 两个来源共用 |
| --- | --- |
| VLM prompt | 默认 1 × 2048，共 2048 个可训练参数 |
| action prompt | 0，不在 action 序列中插入 token |
| 冻结 | 视觉编码器、VLM、action expert、action 输入/输出投影、时间条件模块 |
| 目标 | 原 observation-conditioned flow matching，3000 次更新 |
| 关闭 | CABO、priming、bridge、LoRA、action projection 训练 |
| 精度 | 默认 bfloat16，可用 `DTYPE=float32`；三组评测跟随该 run 保存的 dtype |
| 优化器 | AdamW，峰值 2.5e-5，weight decay=0.01，不启用额外梯度裁剪 |
| 调度 | 沿用 ablation 的配置：warmup=1000、decay=30000、终值=1e-5；3000 步时调度器自动缩放为 warmup=100、decay=3000 |
| W&B | 开启，project=`prompt-learning`，online，disable_artifact=true |

VLM prompt 位于图像/语言/状态组成的 observation prefix 之后。Observation 保持原来的互相可见关系，
VLM prompt 读取 observations 和自己的 prompt 块，action tokens 读取完整 prefix 与 action 块。
因此梯度经冻结的 action expert、VLM 回到 prompt，冻结不等于对整条路径使用 `no_grad`。
没有新 action head。`vlm_prompt_only=true` 会拒绝启用其他适配路径；优化器再次检查
唯一可训练张量必须是 `model.vlm_prompt_tokens.weight`。

## AutoDL 启动

已有 checkout，且本地修改已妥善保存，第一次创建本地 `vlm-only` 分支：

```bash
git fetch origin vlm-only
git switch -c vlm-only FETCH_HEAD
```

这也适用于之前用 `--single-branch` 克隆的仓库。
若本地已存在 `vlm-only`，则执行 `git switch vlm-only` 和 `git pull --ff-only origin vlm-only`。

新机器可直接：

```bash
git clone --branch vlm-only --single-branch https://github.com/LianYi233/lerobot060.git
cd lerobot060
```

激活已有 LeRobot 环境并安装此 checkout（`pip install -e .`）。默认路径：

- dataset：`/root/autodl-tmp/datasets/libero`
- base：`/root/autodl-tmp/models/pi05_libero_base`
- v044：`/root/autodl-tmp/models/pi05_libero_finetuned_v044`
- tokenizer：`/root/autodl-tmp/models/google/paligemma-3b-pt-224`

模型目录必须包含完整权重、config 和 pre/postprocessor 文件，v044 还需要它们引用的统计量文件。
缺少模型时可以沿用 `HF_HUB_DISABLE_XET=1 hf download ... --local-dir ... --max-workers 1` 下载。
base 使用本地数据集 `meta/stats.json` 的 state/action `q01/q99`；启动前检查缺失项。
可用 `BASE_PRETRAINED_PATH`、`FINETUNED_PRETRAINED_PATH`、`DATASET_ROOT`、`TOKENIZER_PATH` 指定其他位置。

默认**两组并行，每张卡一个模型，各自 global batch=32**：

```bash
# 先检查路径、模型、统计量、目标目录和两条训练命令
GPU_IDS=0,1 DRY_RUN=true bash examples/training/train_pi05_vlm_only_pair.sh 0

# GPU 0：base；GPU 1：v044
GPU_IDS=0,1 BATCH_SIZE=32 bash examples/training/train_pi05_vlm_only_pair.sh 0
```

如果希望两卡共同训练一个模型，先 base、后 v044，使用 DDP。
`BATCH_SIZE` 始终指每卡 batch；每卡 16 × 2 卡仍为 global batch=32：

```bash
TRAIN_LAYOUT=ddp GPU_IDS=0,1 BATCH_SIZE=16 \
bash examples/training/train_pi05_vlm_only_pair.sh 0
```

两种方式是替代方案，不要对同一输出目录重复启动。并行模式使用不同端口
29531/29532、不同训练目录、日志和编译缓存。任一模型失败不会取消另一个独立训练，
最终退出码会报告失败；只重跑失败来源时使用单模型入口：

```bash
GPU_IDS=1 bash examples/training/train_pi05_vlm_only.sh finetuned 0
```

已有输出会被拒绝覆盖；保留有价值的 checkpoint，选择新 `RUN_NAME` 重跑，
或用训练器的 `--config_path=.../train_config.json --resume=true` 恢复已有 run。

更改 token 数或学习率：

```bash
VLM_PROMPT_TOKENS=16 GPU_IDS=0,1 BATCH_SIZE=32 \
bash examples/training/train_pi05_vlm_only_pair.sh 0 --policy.optimizer_lr=1e-5
```

默认输出根目录：`/root/autodl-tmp/chkpt/2601-lerobot/vlm-only`。
run 名称：`pi05-base-vlm1-seed0`、`pi05-finetuned-vlm1-seed0`。
日志在 `/root/autodl-tmp/logs/vlm-only`。保存 `000000`、`001000`、`002000`、`003000` 的完整模型；
两组共 8 份 checkpoint，需预留相应磁盘空间。`SAVE_FREQ=3000` 可只保存初始化和最终模型。

## 三组对照与评测

每个来源均对比：

| 条件 | 权重与处理器 |
| --- | --- |
| baseline | 原始 source 权重、零 prompt、与该 run 的 step-0 相同的处理器和 dtype |
| initial | 该 run 保存的真实 step-0 随机 VLM prompt |
| trained | 该 run 的 step-3000 学习后 VLM prompt |

base 官方 processor 的 normalizer `features` 为空、未附统计量，直接原样加载的处理流程
与之前 ablation 训练时不同。因此 base 训练沿用 dataset QUANTILES；v044 保留其官方
MEAN_STD 统计量。**每个来源的三组对照内部保持数据处理一致**。
由原权重和 step-0 处理器组合生成的 baseline 是数据处理一致的零 prompt 对照，
不能直接沿用之前 base 原样评测的 2% 作为它的结果。两个来源之间仍有归一化方式差异，
绝对成功率的差异不能全部归于初始权重。

baseline 生成不加载/修改 source 权重，只在该 run 的 `matched_baseline/pretrained_model`
创建指向原权重的符号链接并复制小型配置/处理器文件；原模型目录需要保留。
baseline 和 initial 评测需要该 run 已保存 `000000`；trained 还需要所选训练步的 checkpoint。
评测前检查 prompt 张量、源路径、配置、处理器及统计量是否匹配。

训练完成后，分别在两个终端评测，使用两张卡：

```bash
# 终端 1
GPU_ID=0 bash run_eval_libero_vlm_only.sh base all all 0

# 终端 2
GPU_ID=1 bash run_eval_libero_vlm_only.sh finetuned all all 0
```

也可以 `GPU_ID=0 bash run_eval_libero_vlm_only.sh both all all 0` 单卡顺序测完两来源的六组条件。
默认每个模型四套 LIBERO、每任务 10 次，共 400 episodes；环境 seed=1000，执行动作数=10。
输出根目录：`/root/autodl-tmp/eval/vlm-only`，每个来源/run/condition 独立保存并支持原评测续跑。
修改 token 数后，评测也传入同一个 `VLM_PROMPT_TOKENS`；不会在评测时修改学习好的 prompt 大小。

```bash
VLM_PROMPT_TOKENS=16 GPU_ID=0 bash run_eval_libero_vlm_only.sh base all all 0
CHECKPOINT_STEP=1000 GPU_ID=0 bash run_eval_libero_vlm_only.sh base trained all 0
```

主要报告 `trained − baseline`；`initial − baseline` 刻画插入随机 prompt 的影响，
`trained − initial` 刻画学习带来的改善。随机或全零 prompt 都不保证保持原函数。
不要将同环境 seed 理解成不同模型拥有完全相同的去噪噪声流。
先固定最终 3000 步作为主比较点，再把中间 checkpoint 作为曲线分析，避免挑最高测试结果。

官方 base 处理器：
https://huggingface.co/lerobot/pi05_libero_base/blob/main/policy_preprocessor.json

# Piper task 1：改用 PI05 DROID / Franka 初始化

GPU 0、1 低学习率／GPU 2、3 原学习率的并行对照，见 [学习率实验说明](README_DROID_LR.md)。

`lerobot/pi05_droid` 是 LeRobot 提供的 PI05 DROID 权重，适合作为新的微调初始化。
DROID 平台采用 Franka Panda 7-DoF 机械臂；这是在 DROID 真机数据上适配的权重，
不是另一种模型架构，也不是 Piper 的专用控制器。真机抓取偏差是否改善仍需实验验证。

官方资料：
- [DROID 平台](https://droid-dataset.github.io/)
- [OpenPI 模型列表](https://github.com/Physical-Intelligence/openpi#model-checkpoints)
- [LeRobot 格式权重](https://huggingface.co/lerobot/pi05_droid/tree/72824c0a93f00ce5bb8bedb7feb58953ba1da364)
- [模型配置](https://huggingface.co/lerobot/pi05_droid/blob/72824c0a93f00ce5bb8bedb7feb58953ba1da364/config.json)

## 这次只更换初始化

新增 `train_piper_droid.sh 1 last2 0`，复用已有 base/direct 训练入口：

| 项目 | 配置 |
| --- | --- |
| 初始权重 | `lerobot/pi05_droid`，固定 revision `72824c0a93f00ce5bb8bedb7feb58953ba1da364` |
| 可训练模块 | 16+16 prompt、动作输入/输出投影、expert 末两层，47,313,952 参数 |
| 目标 / 时间窗口 | Piper absolute；预测 16、执行 8 |
| 预算 / 保存点 | 12000 次 flow 更新；6000、9000、12000 |
| Priming / bridge / CABO | 0 / 0 / 关闭 |
| 数据 / 验证 | 全部有效 episode；末尾 10% 留作验证，seed 0 |
| 精度 / LR | FP32；prompt/投影 1e-4，expert 1e-5 |

两种基座声明的骨干都是 `gemma_2b` / `gemma_300m`，内部 state/action 最大维度 32。
DROID checkpoint 保存的 horizon 为 15；本次从权重初始化、重新训练 horizon 16，
以匹配上一组实验。模型中没有依赖 horizon 长度的可训练位置表，动作投影仍为 32 维。
正式加载仍执行原有严格的权重键/形状检查，不忽略缺失或不匹配的骨干权重。

DROID 的 state 是 7 个关节加夹爪；Piper 是 6 个关节加夹爪。我们保留 Piper 的
两路相机名称、关节顺序、标签单位和原始 action 标签，只用 DROID 提供初始模型权重。
按 Piper 的训练 split 重算统计，重新构建处理器，不加载 DROID 的机器人动作映射或统计。
没有增加标签移位、强制闭爪、驱动修改或碰撞保护修改。

## 下载（现有训练环境即可）

```bash
cd /home/wyn/VLAA/lerobot060
git pull --ff-only origin piper
export WORK_ROOT=/data1/wyn/piper-work
export PI05_DROID_PATH="$WORK_ROOT/models/pi05_droid"

python examples/piper/download_pi05_droid.py \
  --output_dir "$PI05_DROID_PATH" \
  --endpoint https://hf-mirror.com
```

这里使用第三方镜像以应对训练机无法连接 Hugging Face 的情况；脚本不发送登录 token。
能访问官方站时，将 endpoint 改为 `https://huggingface.co`。脚本固定模型版本，逐个下载
四个必需文件，并核对官方权重 SHA256。保留已有下载，不自动删除文件，不升级依赖。
约 16.6 GB 权重下载后会再读一遍做校验；出现 `DROID_FILES_OK` 表示下载校验完成，
不表示已经通过完整模型加载或 GPU 训练。也可在其他联网电脑下载后复制这四个文件。

只检查已下载文件，不联网：

```bash
python examples/piper/download_pi05_droid.py --output_dir "$PI05_DROID_PATH" --check_only
```

## 训练 task 1

沿用 task 1 上次的 GPU 2、3；如果它们被占用，改用空闲 GPU 并保持两卡全局 batch 32。
task 2 若还在卡 0、1 上训练，不要同时在同一组卡上启动本任务。
训练机和部署机保持原来匹配的 Transformers 版本，不安装 OpenPI 或替换环境。

```bash
export DATASET_BASE=/data/datasets/May-pick-and-place
export TOKENIZER_PATH=/data/models/paligemma-3b-pt-224
export GPU_IDS=2,3
export NUM_PROCESSES=2
export BATCH_SIZE=16
export FIT_EPISODES=all
export EVAL_SPLIT=0.1
export CHUNK_SIZE=16
export N_ACTION_STEPS=8
export MASKED_STEPS=12
export FLOW_STEPS=12000
export SAVE_STEPS='[6000,9000,12000]'
export DTYPE=float32
export MIXED_PRECISION=no
export COMPILE_MODEL=false

export RUN_GROUP="piper-task1-pi05-droid-last2-h16-12k-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_ROOT="/data1/wyn/chkpt/2601-lerobot/${RUN_GROUP}"
export LOG_ROOT="/data1/wyn/logs/${RUN_GROUP}"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DRY_RUN=true \
  bash examples/training/train_piper_droid.sh 1 last2 0
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DRY_RUN=false \
  bash examples/training/train_piper_droid.sh 1 last2 0
```

`PI05_DROID_PATH` 覆盖残留的 `PI05_BASE_PATH` 和 `PRETRAINED_PATH`，以免仍读到旧基座。
脚本默认的 RUN_GROUP 包含 `pi05-droid`，手工指定 OUTPUT_ROOT/LOG_ROOT 时也应使用新的组名。
启动日志应显示 `Loading model from: .../pi05_droid` 和 `Trainable total: 47,313,952`。
`last4` 也可用，但首轮先保留 last2，避免同时改变基座与可训练范围。

最终模型：

```text
$OUTPUT_ROOT/pi05-may-1-put_the_apple_on_the_yellow_plate-dual_prompt_only-seed0/checkpoints/012000/pretrained_model
```

W&B 和训练/验证动作评估继续开启。输出中 `dual_prompt_only` 是阶段配方名，
并不表示只训练 prompt。分析方法和文件见 [基座对照说明](README_BASE_MODEL.md)。
重点比较相同样本的前 1/8 步关节 MAE、P95、moving 子集和夹爪误差。
若离线误差改善但真机仍系统性偏移，应进一步对照发送动作与实际反馈、相机视角和时序；
这时不能仅用更换基座来解释或解决问题。

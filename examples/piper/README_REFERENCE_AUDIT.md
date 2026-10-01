# 与参考真机代码对照：先区分拟合、环境和执行问题

检查日期：2026-10-01。参考材料是用户提供的 `lerobot-main.zip`，本分支对照起点
为 `b97116a`。未获得这次四个任务的 checkpoint、实际训练日志、录像或关节跟踪记录；
下文区分代码事实与待验证的解释，不声称已找到唯一根因或改善了抓取成功率。

## 后续实测：苹果模型的数据选择已确认，部署环境应对齐 5.5.4

用户随后在两台机器上运行了审计工具，报告同一个苹果任务 checkpoint 的
`episodes=null`、`eval_split=0.0`、`configured_flow_steps=3000`、
`per_process_batch_size=16`。结合用户明确设置 `FIT_EPISODES=all`，本次不再把
单条示范训练作为该模型的解释。配置中的 3000 是正式 flow 阶段设置，并非根据
目录名称推断出的 12000；实际保存步数另看报告的 `saved_step`。

两台机器的探针结果为：

| 环境 | Transformers | 图像单位输出 | 文字单位权重输出 | embedding 类 |
| --- | --- | --- | --- | --- |
| 真机电脑 | 5.3.0 | 0.25 | 1.0 | Embedding |
| 训练电脑 | 5.5.4 | 1.0 | 4.0 | GemmaTextScaledWordEmbedding |

这确认了当前环境的计算差异。按用户提供的训练环境信息，应先在真机电脑使用
5.5.4，保留同一份权重做离线复测，无需立即重训。若训练后升级过训练环境，仍应
以当时的版本记录为准。对齐版本不能保证消除所有抓取问题，也不会让部署代码
自动具备任务完成检测。

在真机电脑的 `lerobot` 环境中执行（安装步骤需要能访问包源）：

```bash
conda activate lerobot
python -m pip freeze > "piper-env-before-554-$(date +%Y%m%d-%H%M%S).txt"
python -m pip install "transformers==5.5.4"
python -m pip check
python deploy_piper_wyn.py --check_env

python piper_checkpoint_audit.py \
  --policy_path /absolute/path/to/the/same/pretrained_model \
  --probe_embeddings \
  --output piper_robot_audit_554.json
```

安装时让 pip 解析 Transformers 自身的依赖，不使用 `--no-deps`，也不顺带升级
PyTorch/CUDA。这里的精确版本来自这批 checkpoint 的训练环境，不是给所有历史
checkpoint 指定统一版本。探针应变为图像 1.0、文字 4.0；随后在两台机器用同一
checkpoint、数据集、episode、seed 运行离线动作评估，重点比较 h1/h8 的误差。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python eval_piper_offline.py \
  --policy_path /absolute/path/to/the/same/pretrained_model \
  --dataset_root /absolute/path/to/May-pick-and-place/1-put_the_apple_on_the_yellow_plate \
  --tokenizer_path /absolute/path/to/paligemma-3b-pt-224 \
  --episodes 0,1,2 --stride 8 --max_samples 64 --execution_steps 8 \
  --seed 0 --plots 3 --output_dir /absolute/path/to/new/eval-apple-transformers554
```

若已有完全相同样本、seed、设置的 5.3.0 离线结果，可直接对比升级前后误差和
动作曲线；不要用不同采样设置的结果归因版本影响。离线脚本直接使用
`piper_policy_utils.py` 的只读辅助函数，不再导入部署入口或硬件保护模块。

## 最优先：训练与推理的 Transformers 行为是否一致

参考包的 `deploy_piper_wx.py` **额外覆盖**了 `embed_image` 和 `embed_language_tokens`，
其注释明确说明要复现 Transformers 5.5.4 的训练行为。覆盖只针对参考作者核对过的
quantile checkpoint，不能把整个脚本或它的采样选项套用到我们的 flow checkpoint。

官方源码确认了以下区别：

| HF 原语 | Transformers 5.3.0 | Transformers 5.5.4 |
| --- | --- | --- |
| `PaliGemmaModel.get_image_features` | 投影输出除以 `sqrt(hidden_size)` | 直接使用投影输出 |
| `GemmaModel` 的 token embedding 模块 | 普通 `nn.Embedding` | `GemmaTextScaledWordEmbedding`，内部乘以 `sqrt(hidden_size)` |

本分支直接调用这两个入口；`PiGemmaModel` 继承 HF 的 embedding 构造，没有再统一
embedding 尺度。因此相同代码、相同权重，跨这两个版本也可能得到不同前缀特征。
默认 VLM 宽度 2048 时，两类观测 embedding 从旧版到新版名义上均放大
`sqrt(2048) ≈ 45.25` 倍；这不是最终 action 的放大倍数，也不是图文相对比例变化。
prompt、残差和后续非线性使模型行为不能仅凭权重 key 全部匹配来保证一致。

用户此前真机环境报告过 Transformers 5.3.0，当前分支依赖声明是 `>=5.4,<5.6`。
**初次检查时尚不知道实际训练环境；后续实测见上节。** 对其他历史 checkpoint，如果训练也用了 5.3.0，不能仅因
新版依赖声明就认定必须升级。先复现训练环境，再比较同一 checkpoint 在固定训练样本上的
预测；也不能仅凭基础模型目录名认定它的历史依赖版本。

官方代码依据：

- [5.3.0 PaliGemma](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/paligemma/modeling_paligemma.py)
- [5.5.4 PaliGemma](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/paligemma/modeling_paligemma.py)
- [5.3.0 Gemma](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/gemma/modeling_gemma.py)
- [5.5.4 Gemma](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/gemma/modeling_gemma.py)

## 第二项：这次到底用了多少条示范

`train_piper_fit.sh` 默认 `FIT_EPISODES='[0]'`。即使对四个任务分别执行一次，每个任务
也只使用本任务第一条示范；增加 batch 或迭代数不会增加场景、物体位置的覆盖范围。
目录中出现 `no_cabo` 不足以判断是否用了 projections；查看 config 中
`train_action_projections` 的实际值。要确认训练数据选择，读取 checkpoint 内的
`train_config.json`，不能只看当前 shell 的变量。

- `dataset.episodes=[0]`：单条拟合实验。离线应该先评估同一条示范，不能据此承诺一般场景抓取。
- `dataset.episodes=null`：没有显式 episode 子集；仍需检查 `eval_split`，留出部分没有用于训练。
- 缺少 `train_config.json`：信息未知，不能假定用了全部数据。

## 第三项：预测正确不代表实际轨迹跟得上

| 项目 | 参考包 | 当前 Piper 部署 |
| --- | --- | --- |
| 关节单位 | SDK 毫度转弧度，再逆变换发送 | 相同 |
| 夹爪 | 原始开度除以 70000；命令乘以 70000 | 相同 |
| 彩色图像 | RealSense BGR8，驱动转 RGB | 默认按驱动返回 RGB 使用 |
| 默认相机映射 | high=213622075951，wrist=213222078968 | 取本机 PiperConfig，可覆盖；需核对实景与训练视角 |
| MOVE J 速度参数 | 原驱动硬编码 100% | 默认 20%，故障监测开启 |
| 原 `deploy_piper.py` 循环 | 实际也是先等待推理，再执行 chunk | 同类串行循环；默认执行 8 步 |
| 新 `deploy_piper_wx.py` 入口 | 安装 `wx_continuous_control.py`：10 Hz 策略时间、50 Hz 插值发送、提前推理、按观测年龄选动作 | 未启用这套连续控制 |

原参考脚本的“双缓冲”说明与实际 `run_episode` 的串行执行不同，不能只按文件注释判断。
参考新增连续控制逻辑确实处理了推理等待，但没有附成功率或跟踪测试结果。

当前每 8 步按 10 Hz 发送，之后约 0.22 秒推理，加起来至少 1.02 秒，平均约
7.84 个策略动作/秒，还未计算采图等开销。推理前又将实测当前位置作为保位目标，
若 MOVE J 尚未到达上一目标，可能产生反复停走。需要同时记录
`raw_prediction`、裁剪后/发送的 `command`、反馈 `measured_state`、时间戳和故障，
才能判断偏差来自模型、限位裁剪还是动作跟踪。

用户已发生压桌和碰撞保护；本次不提高默认速度、不绕过保护，也不直接移植参考脚本
中的失能退出逻辑。先确认离线预测和环境一致性，再单独测试控制改动。

## “放好后仍在动”不能单独用来判断视觉识别

当前脚本和参考脚本都没有物体检测、成功分类器、完成概率阈值或 success/done 停止分支，
PI05 输出的是动作序列。循环按 `steps_per_episode` 继续运行，除非人工中断或发生故障。
它可能学到示范末尾的保位动作，但没有独立的“任务完成”输出。

人手把物体放入盘子，还会引入与示范不同的物体/手臂组合和遮挡。
在无手遮挡、机械臂状态与示范接近的完成场景比较预测动作，比要求模型立即自动停止更能
隔离问题。需要自动结束时，应另行验证完成检测模块及连续多帧判定，而非简单把“小动作”
当作成功，因为卡住、停滞也会产生小动作。

## loss 与可训练范围

我们的正式目标是随机时间点的 flow velocity MSE（`noise - normalized_action`），
不是最终预测角度的 MSE。参考包主要 PI05 配置默认 regression/quantile 分支，
其 pinball/L1/L2 目标与 flow loss 数值不可直接比较。参考包还包含多个 policy 版本，
没有其实际训练配置，不能断言作者用了哪一个或哪种配置获得了成功。

我们的 projections 配置仍只训练 115,744 个参数，两大 transformer 冻结；参考包主
PI05 代码允许全量或 action expert 训练，不能把两者看作相同的适配能力。是否需要
扩大 action expert 的适配范围，要由**训练环境内的离线动作误差**决定，不能仅凭 loss>0.2。

## 现在可以执行的检查

先更新 `piper`。在训练机器、真机电脑各运行一次；`--policy_path` 指向各自机器上的
**同一份模型**，`--dataset_info` 可省略。输出使用新文件名，脚本不会覆盖既有文件。

```bash
python piper_checkpoint_audit.py \
  --policy_path /absolute/path/to/pretrained_model \
  --probe_embeddings \
  --output piper_train_audit.json
```

真机电脑输出改为 `piper_robot_audit.json`。若复制了训练报告到真机，可加
`--compare /path/to/piper_train_audit.json`。无需连接机械臂、相机或加载权重；可选探针只
在 CPU 上构造一个宽度 16 的小 Gemma，并以单位 vision 输出测试 HF 原语。
已核对的探针期望值：5.3.0 图像 0.25/文字 1；5.5.4 图像 1/文字 4。
探针导入失败会写明异常，同时保留其他审计信息；不是兼容性通过。

报告包含 checkpoint 的 episode 选择、正式训练步数、policy 配置、处理器及其统计文件
指纹、当前依赖版本和关键源码指纹。**当前训练机器环境不等于历史训练环境**：若升级过，
还需要当时的环境记录或 W&B metadata。报告不读取模型大权重，不能证明权重传输完整。

接着在原训练环境运行同一模型的 `eval_piper_offline.py`，固定 episode、seed、stride，
再在真机电脑以相同设置运行。优先比较 h1/h8 关节 MAE（度）、夹爪误差、运动片段和
静止基线。误差在训练机小、部署机明显大，优先定位环境/处理器；两处都大，才进一步
排查拟合和适配容量；两处都小但真机失败，则重点检查视角、初始场景与实际跟踪。

分析训练曲线还需要各任务输出目录的 `action_eval.jsonl`、`training_diagnostics.jsonl`，
这些没有包含在本次参考代码压缩包中。`action_train` 是固定示范观测下的离线拟合指标，
不是闭环成功率；仅单条拟合时也不能代表未见场景。

## 本次验证范围

审计工具的五项无硬件测试通过，涵盖缺失训练配置、单条/全量选择、处理器统计指纹、
报告比较、探针失败保留诊断和不修改 checkpoint。另在独立 CPU 环境分别安装
Transformers 5.3.0、5.5.4（PyTorch 2.7.1+cpu），实际运行探针，结果与上表预期一致。
没有在用户的模型权重、数据集、GPU 或真机上验证抓取改善；本次没有修改模型前向、
控制速度或碰撞保护。

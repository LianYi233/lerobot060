# LIBERO 在线评测注意力视频

在 `prompt-ablation` 分支中，激活现有 LeRobot/LIBERO 环境后，在仓库根目录运行：

```bash
GPU_ID=0 bash run_eval_libero-full_with_attention.sh full_reference all 0
```

参数沿用 `run_eval_libero-full.sh`：模型 variant、`all`/`no10`、训练 seed。
默认评测 `003000/pretrained_model`，每个 task 10 个 episode、单环境、每 10 步预测一个动作块。
保留原脚本的 CUDA 预检、整体进度、成功率统计和按 task 断点续测。新脚本自动使用当前仓库的 `src`。
默认结果单独保存到 `/root/autodl-tmp/eval/2601-lerobot-attention`。

如果模型目录采用之前 NTK 实验的命名，显式指定 checkpoint 根目录：

```bash
RUN_DIR=/root/autodl-tmp/chkpt/2601-lerobot/prompt-ablation/pi05-full-reference-ntk-seed0
BASE_CKPT="$RUN_DIR/checkpoints" GPU_ID=0 \
  bash run_eval_libero-full_with_attention.sh full_reference all 0
```

`BASE_CKPT` 下应有 `003000/pretrained_model`。这里 3000 是主训练阶段的 checkpoint 步数；
对应先前热图中的 4000 累计步数。路径不同可直接修改 `BASE_CKPT`，无需修改训练配置或重新训练。

## 视频内容与文件

视频横向展示三个面板：实时环境、产生当前动作块的模型输入、该输入上的注意力叠加。
热图低值呈蓝紫色，高值过渡到亮红、橙黄和亮黄色，默认颜色覆盖强度为 0.82。
所有文字统一使用 Times New Roman；程序自动查找并核对字体名称，不会静默替换为其他字体。
如果服务器没有安装该字体，上传已有的 Times New Roman 常规字体文件，并指定实际路径：

```bash
export ATTENTION_FONT_PATH=/root/fonts/times.ttf
```

热图使用实际传入视觉编码器的像素，已包含 LIBERO 方向变换、缩放和 padding，不再额外翻转。
执行队列中的后续动作时，输入图与热图保持在上次预测时刻；左侧实时画面继续更新。
底部标注输入步数、queued-action age、注意力来源、层号、相机、该相机的注意力总质量及颜色范围。
第一帧尚未进行预测，会明确显示等待状态。

每个成功和失败 episode 都保存，目录形式为：

```text
$BASE_OUTPUT/libero060-all-<run>-003000-<suite-tag>-resume/
  videos/<suite>_<task_id>/
    eval_episode_0_TRUE_attention.mp4
    eval_episode_0_TRUE_attention.attention.npz
    eval_episode_0_TRUE_attention.attention.json
  task_results/
  resume_manifest.json
  resume_summary.json
  eval_info.json
```

NPZ 保存每次重新预测的原始 patch 概率 `maps`（预测次数 × 网格高 × 网格宽）、
`input_steps`、`denoise_pass_counts`、`image_attention_mass`。JSON 记录来源、实际层号、相机 feature key 等。
标准 224×224 输入、14×14 patch 对应 16×16 网格；程序依据实际视觉配置推导并核对 token 数。
视频插值只用于显示，不改变 NPZ 数据。

## 已有视频直接重绘

在现有 LeRobot 环境、仓库根目录运行以下命令，无需加载 checkpoint 或使用 GPU：

```bash
PYTHONPATH=src python -m lerobot.scripts.replot_libero_attention \
  --input-dir /root/autodl-tmp/eval/2601-lerobot-attention \
  --output-dir /root/autodl-tmp/eval/2601-lerobot-attention-restyled
```

输入目录可以是整个评测目录或单个任务目录，须保留视频及同名 `.attention.npz`、
`.attention.json`。工具递归处理视频，在新目录中保留相对路径和三个面板的排列，
重绘全部文字与热图，原视频不变。再次运行相同命令会跳过已完成的视频。
字体可通过上面的 `ATTENTION_FONT_PATH` 或 `--font-path /实际路径/times.ttf` 指定。
如需更强的颜色覆盖，可加 `--alpha 0.9`。

默认保留原始颜色数值上限，NPZ 原样复制，注意力概率不变；仅颜色和字体变化。
模型输入画面来自原视频的中间面板，重新编码会带来轻微压缩差异。
输入和输出目录须相互独立；修改重绘设置时使用新的输出目录。
若重新运行在线评测，也请设置新的 `BASE_OUTPUT`（如 `...-attention-v2`），
避免与旧版本的断点续测 manifest 冲突。

## 注意力定义

默认 `ATTENTION_SOURCE=action`：最后一个 action-expert Transformer 层中，
即将执行的前 `n_action_steps` 个动作 token 对图像 patch token 的注意力。
对全部注意力 head 和所有 flow denoising passes 取算术平均。
softmax 的分母包含所有可见 key（其他相机、语言、prompt、动作等），然后才选出目标相机；
**不会只在图像 token 上重新归一化**。因此 `image_attention_mass` 可以反映分配给该相机的概率质量。

`ATTENTION_SOURCE=vlm_prompt`：VLM prompt token 对图像 patch token 的注意力。
VLM 前缀每次预测只运行一次，因而不涉及多次 denoising 的聚合。
没有 VLM prompts 的模型会报错，而不会生成虚假的空图。

当前模型的 action prompts 被 attention mask 限制，不能直接读取图像；
所以 action 热图查询的是 **动作 token**，不能将它称为 action-prompt-to-image attention。
图像 key 本身也包含上下文信息，空间热图是 attention routing 的诊断，而非对象分割或完整因果归因。

程序在指定层读取真实 RoPE 后的 Q 和包含前缀缓存的 K，额外计算所需的少量注意力行。
模型输出仍由原来的 SDPA/eager 后端计算，不替换动作、不消耗额外随机数、不进行额外 policy forward。
仅支持单环境、顺序 task 评测；需关闭 compile 和 RTC。新启动脚本已关闭 compile。
记录和视频编码有额外开销，运行速度可能低于普通评测。

## 可调整项

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `ATTENTION_SOURCE` | `action` | `action` 或 `vlm_prompt` |
| `ATTENTION_LAYER` | `-1` | 层号从 0 开始，负数从末尾计数 |
| `ATTENTION_CAMERA` | `0` | 模型图像顺序中的索引；通常 0 外部相机，1 手腕相机，JSON 保存实际 key |
| `ATTENTION_DENOISE` | `mean` | `mean`、`first`、`last`；VLM 前缀只有一次 |
| `ATTENTION_ALPHA` | `0.82` | 颜色覆盖强度，0 到 1，越大颜色越明显 |
| `ATTENTION_VMAX` | `0` | 0 按当前预测的最大值显示；正值固定颜色上限 |
| `ATTENTION_FONT_PATH` | 自动查找 | Times New Roman 字体文件路径；缺少字体时立即报错 |
| `BASE_OUTPUT` | `/root/autodl-tmp/eval/2601-lerobot-attention` | 独立输出根目录 |

例如观察 VLM prompts 与手腕相机：

```bash
ATTENTION_SOURCE=vlm_prompt ATTENTION_CAMERA=1 \
BASE_OUTPUT=/root/autodl-tmp/eval/2601-lerobot-attention-vlm-wrist \
  bash run_eval_libero-full_with_attention.sh full_reference all 0
```

更改注意力设置、代码或模型后，使用新的 `BASE_OUTPUT`；manifest 会拒绝混合不同配置的旧结果。
断点恢复还会检查视频与 NPZ/JSON 是否存在，缺失时明确报错。

默认相对颜色范围适合看同一张图中的空间分布，不能直接比较两段视频的颜色强度。
跨方法比较应使用相同层、查询定义、相机、评测 seeds 和固定 `ATTENTION_VMAX`，
并结合原始概率、成功率与消融结果。注意力落在目标物体上可作为定性证据，单独不能证明方法有效。

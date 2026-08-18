#!/usr/bin/env bash
# Evaluate a local PI0.5 / flow-mask checkpoint on LIBERO-plus.
#
# Defaults match the wyn workstation layout:
#   repo:       /home/wyn/myStudy/2601-data_in_lerobot_format
#   conda env:  lerobot5
#   checkpoint: /media/wyn/data/10-EmbodiedAI/2601-lerobot/chkpt/fmm816-12k
#
# Hugging Face's LIBERO-plus page still documents the vanilla 10-task table and
# `n_episodes=10` (400 episodes). That is **not** the complete plus benchmark.
# Official plus (sylvestf/LIBERO-plus) is ~10,030 perturbation tasks × 1 trial.
#
# Usage:
#   bash examples/libero_plus/eval_flow_mask_checkpoint.sh smoke
#   bash examples/libero_plus/eval_flow_mask_checkpoint.sh full-plus
#   MODE=smoke TASK=libero_spatial N_EPISODES=1 bash examples/libero_plus/eval_flow_mask_checkpoint.sh

set -euo pipefail

MODE="${1:-${MODE:-smoke}}"
REPO_HINT="${REPO_HINT:-/home/wyn/myStudy/2601-data_in_lerobot_format}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot5}"
CHKPT_ROOT="${CHKPT_ROOT:-/media/wyn/data/10-EmbodiedAI/2601-lerobot/chkpt/fmm816-12k}"
EVAL_ROOT="${EVAL_ROOT:-/media/wyn/data/10-EmbodiedAI/2601-lerobot/eval}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${HOME}/myStudy/LIBERO-PLUS/LIBERO-plus}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BRANCH="${BRANCH:-flow-mask}"
GPU_ID="${GPU_ID:-0}"
CONTROL_MODE="${CONTROL_MODE:-relative}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"

if [[ "${MODE}" == "full-plus" ]]; then
  TASK="${TASK:-libero_spatial,libero_object,libero_goal,libero_10}"
  N_EPISODES="${N_EPISODES:-1}"
  TASK_IDS="${TASK_IDS:-}"
elif [[ "${MODE}" == "full" ]]; then
  TASK="${TASK:-libero_spatial,libero_object,libero_goal,libero_10}"
  N_EPISODES="${N_EPISODES:-10}"
  TASK_IDS="${TASK_IDS:-}"
else
  TASK="${TASK:-libero_spatial}"
  N_EPISODES="${N_EPISODES:-1}"
  TASK_IDS="${TASK_IDS:-[0]}"
fi

log() { printf '[eval-libero-plus] %s\n' "$*"; }
die() { printf '[eval-libero-plus] ERROR: %s\n' "$*" >&2; exit 1; }

find_repo() {
  local hint="$1"
  if [[ -d "${hint}/.git" ]]; then
    printf '%s\n' "${hint}"
    return
  fi
  if [[ -d "${hint}/lerobot060/.git" ]]; then
    printf '%s\n' "${hint}/lerobot060"
    return
  fi
  local found
  found="$(find "${hint}" -maxdepth 3 -type d -name .git -printf '%h\n' 2>/dev/null | head -n 1 || true)"
  [[ -n "${found}" ]] || die "no git repo under ${hint}"
  printf '%s\n' "${found}"
}

source_conda() {
  local candidate
  for candidate in \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/root/anaconda3/etc/profile.d/conda.sh" \
    "/root/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return
  fi
  die "conda not found"
}

resolve_pretrained_dir() {
  local root="$1"
  [[ -e "${root}" ]] || die "checkpoint path does not exist: ${root}"
  if [[ -f "${root}/config.json" && -f "${root}/model.safetensors" ]]; then
    printf '%s\n' "${root}"
    return
  fi
  if [[ -f "${root}/pretrained_model/config.json" ]]; then
    printf '%s\n' "${root}/pretrained_model"
    return
  fi
  local latest
  latest="$(ls -1d "${root}"/checkpoints/*/pretrained_model 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "${latest}" ]] || die "could not find config.json + model.safetensors under ${root}"
  printf '%s\n' "${latest}"
}

choose_mujoco_gl() {
  if python -c "import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library('EGL') else 1)"; then
    printf 'egl\n'
  else
    printf 'osmesa\n'
  fi
}

log "mode=${MODE} task=${TASK} n_episodes=${N_EPISODES} task_ids=${TASK_IDS:-all}"

REPO="$(find_repo "${REPO_HINT}")"
log "repo=${REPO}"
cd "${REPO}"

if [[ "${SKIP_GIT:-0}" != "1" ]]; then
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  if [[ -n "$(git status --porcelain)" ]]; then
    log "warning: working tree is dirty; not fast-forwarding ${BRANCH}"
  else
    git merge --ff-only "origin/${BRANCH}" || log "could not fast-forward ${BRANCH}; continuing on current HEAD"
  fi
fi
log "git HEAD=$(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

source_conda
conda activate "${CONDA_ENV_NAME}"
log "python=$(command -v python) $(python -V)"
log "conda env=${CONDA_PREFIX}"

export HF_ENDPOINT
export PYTHONPATH="${REPO}/src:${LIBERO_PLUS_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Keep the editable install pointed at this checkout.
if ! python -c "import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve())" >/dev/null 2>&1; then
  log "installing lerobot editable into ${CONDA_ENV_NAME}"
  pip install -e ".[pi]"
fi

PRETRAINED="$(resolve_pretrained_dir "${CHKPT_ROOT}")"
log "policy.path=${PRETRAINED}"
ls -lh "${PRETRAINED}/config.json" "${PRETRAINED}/model.safetensors"

python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${PRETRAINED}/config.json").read_text())
print("policy.type=", cfg.get("type"))
print("n_action_steps=", cfg.get("n_action_steps"))
print("chunk_size=", cfg.get("chunk_size"))
print("training_stage=", cfg.get("training_stage"))
print("dtype=", cfg.get("dtype"))
PY

# LIBERO-plus must replace hf-libero so `import libero` resolves to the fork.
need_libero_plus=0
python - <<'PY' || need_libero_plus=1
from pathlib import Path
import libero
from libero.libero import get_libero_path
file_ = getattr(libero, "__file__", None)
root = Path(file_).resolve().parent if file_ else Path(next(iter(libero.__path__)))
assets = Path(get_libero_path("assets"))
print("libero=", root)
print("assets=", assets, "exists=", assets.exists())
if not assets.exists():
    raise SystemExit(2)
print("libero import ok")
PY

if [[ "${need_libero_plus}" != "0" ]]; then
  log "installing LIBERO-plus into ${LIBERO_PLUS_ROOT}"
  pip install "robosuite==1.4.1" bddl easydict mujoco wand scikit-image gym
  if [[ ! -d "${LIBERO_PLUS_ROOT}/.git" ]]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git "${LIBERO_PLUS_ROOT}"
  fi
  pip install --no-deps -e "${LIBERO_PLUS_ROOT}"
  pip uninstall -y hf-libero || true
  python - <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path
import os, zipfile, shutil
root = Path("${LIBERO_PLUS_ROOT}") / "libero" / "libero"
assets = root / "assets"
if assets.exists() and any(assets.iterdir()):
    print("assets already present", assets)
else:
    z = hf_hub_download(
        repo_id="Sylvest/LIBERO-plus",
        repo_type="dataset",
        filename="assets.zip",
        local_dir="/tmp/libero-plus-dl",
    )
    extract = Path("/tmp/libero-plus-dl/extract")
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(extract)
    found = next(p for p in extract.rglob("assets") if p.is_dir())
    if assets.exists():
        shutil.rmtree(assets)
    shutil.move(str(found), str(assets))
    print("extracted assets to", assets)
cfg_dir = Path.home() / ".libero"
cfg_dir.mkdir(exist_ok=True)
(cfg_dir / "config.yaml").write_text(
    "\n".join(
        [
            f"assets: {root / 'assets'}",
            f"bddl_files: {root / 'bddl_files'}",
            f"datasets: {root.parent / 'datasets'}",
            f"init_states: {root / 'init_files'}",
            "",
        ]
    )
)
print("wrote", cfg_dir / "config.yaml")
PY
fi

if [[ -d "${HOME}/.libero_plus" ]]; then
  export LIBERO_CONFIG_PATH="${HOME}/.libero_plus"
else
  export LIBERO_CONFIG_PATH="${HOME}/.libero"
fi
MUJOCO_GL_VALUE="${MUJOCO_GL:-$(choose_mujoco_gl)}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
if [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM=osmesa
  unset MUJOCO_EGL_DEVICE_ID || true
fi
log "MUJOCO_GL=${MUJOCO_GL}"

STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ "${MODE}" == "full-plus" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ROOT}/fmm816-12k-libero-plus-full}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ROOT}/fmm816-12k-libero-plus-${MODE}-${STAMP}}"
fi
mkdir -p "${OUTPUT_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"

TASK_IDS_ARG=()
if [[ -n "${TASK_IDS}" ]]; then
  TASK_IDS_ARG=(--env.task_ids="${TASK_IDS}")
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

if [[ "${MODE}" == "full-plus" ]]; then
  set -x
  python "${REPO}/examples/libero_plus/run_full_plus_eval.py" \
    --policy.path="${PRETRAINED}" \
    --output_dir="${OUTPUT_DIR}" \
    --env.task="${TASK}" \
    --env.control_mode="${CONTROL_MODE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_amp=false \
    --policy.device=cuda
  set +x
else
  set -x
  lerobot-eval \
    --policy.path="${PRETRAINED}" \
    --output_dir="${OUTPUT_DIR}" \
    --env.type=libero_plus \
    --env.task="${TASK}" \
    --env.control_mode="${CONTROL_MODE}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED:-10}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_amp=false \
    --policy.device=cuda \
    ${TASK_IDS_ARG[@]+"${TASK_IDS_ARG[@]}"}
  set +x
fi

python - <<PY
import json
from pathlib import Path
p = Path("${OUTPUT_DIR}") / "eval_info.json"
print("eval_info=", p, "exists=", p.exists())
if p.exists():
    info = json.loads(p.read_text())
    print(json.dumps(info, indent=2, default=str)[:4000])
PY

log "done. results in ${OUTPUT_DIR}"

#!/usr/bin/env bash
# Source before constructing any Piper output paths. This branch targets the local /data1 server.
# Only the former AutoDL runtime defaults are migrated; other explicit paths are preserved.
# No directories are created here, including during DRY_RUN.

piper_runtime_path() {
  local name="$1" default="$2" current
  current="${!name:-}"
  if [[ -z "${current}" || "${current}" == /root/autodl-tmp || "${current}" == /root/autodl-tmp/* ]]; then
    if [[ -n "${current}" ]]; then
      printf 'Piper local paths: %s=%s -> %s\n' "${name}" "${current}" "${default}" >&2
    fi
    printf -v "${name}" '%s' "${default}"
  fi
  export "${name}"
}

piper_runtime_path WORK_ROOT /data1/wyn/piper-work
piper_runtime_path TMPDIR "${WORK_ROOT}/tmp"
piper_runtime_path TMP "${TMPDIR}"
piper_runtime_path TEMP "${TMPDIR}"
piper_runtime_path TORCHINDUCTOR_CACHE_DIR "${WORK_ROOT}/cache/torchinductor"
piper_runtime_path TRITON_CACHE_DIR "${WORK_ROOT}/cache/triton"
piper_runtime_path HF_HOME "${WORK_ROOT}/cache/huggingface"
unset -f piper_runtime_path

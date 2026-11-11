#!/usr/bin/env bash
# Source before constructing Piper output paths. Usually targets the /data1 local server.
# Optional PIPER_RUNTIME_LAYOUT=autodl preserves explicit /root/autodl-tmp paths for
# the AutoDL A800 machine. With the default local layout, legacy path migration
# is unchanged. Do not create any directories during DRY_RUN.
case "${PIPER_RUNTIME_LAYOUT:-local}" in
  local|autodl) ;;
  *) echo "Unknown PIPER_RUNTIME_LAYOUT: ${PIPER_RUNTIME_LAYOUT}" >&2; return 2 ;;
esac

piper_runtime_path() {
  local name="$1" default="$2" current
  current="${!name:-}"
  if [[ "${PIPER_RUNTIME_LAYOUT:-local}" == "autodl" ]]; then
    # Respect every explicitly selected path, including /root/autodl-tmp.
    if [[ -z "${current}" ]]; then
      printf -v "${name}" '%s' "${default}"
    fi
  elif [[ -z "${current}" || "${current}" == /root/autodl-tmp || "${current}" == /root/autodl-tmp/* ]]; then
    if [[ -n "${current}" ]]; then
      printf 'Piper local paths: %s=%s -> %s\n' "${name}" "${current}" "${default}" >&2
    fi
    printf -v "${name}" '%s' "${default}"
  fi
  export "${name}"
}

if [[ "${PIPER_RUNTIME_LAYOUT:-local}" == "autodl" ]]; then
  piper_runtime_path WORK_ROOT /root/autodl-tmp/piper-partial-ft-work
else
  piper_runtime_path WORK_ROOT /data1/wyn/piper-work
fi
piper_runtime_path TMPDIR "${WORK_ROOT}/tmp"
piper_runtime_path TMP "${TMPDIR}"
piper_runtime_path TEMP "${TMPDIR}"
piper_runtime_path TORCHINDUCTOR_CACHE_DIR "${WORK_ROOT}/cache/torchinductor"
piper_runtime_path TRITON_CACHE_DIR "${WORK_ROOT}/cache/triton"
piper_runtime_path HF_HOME "${WORK_ROOT}/cache/huggingface"
unset -f piper_runtime_path

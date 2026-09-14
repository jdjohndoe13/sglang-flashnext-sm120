#!/usr/bin/env bash
# A/B TEST PROFILE: serve_best.sh with MTP/spec decoding OFF — Qwen3.8-Flash-Next NVFP4,
# 8x RTX 5090 (sm120, 32 GB/card), TP8 + EP8.
#
# Purpose: measure what native NEXTN/MTP speculative decoding (3 steps, FR-Spec 64K hot-token
# draft head) actually contributes to decode throughput on this deployment. Baseline context:
# on the vLLM GLM-5.3-Flash kit the equivalent MTP path gave NO measurable speedup, so it is
# worth checking the same here. Compare `logs/serve.log` decode tok/s (and acceptance-length
# lines, absent in this profile) against a serve_best.sh run on identical prompts.
#
# Only change vs serve_best.sh: SPEC=0 (drops --speculative-algorithm NEXTN + acceptance
# thresholds + --speculative-token-map) and the MTP-specific GDN_MTP_CACHE_MODE. Everything
# else is identical (HICACHE off, MEMFRAC 0.80, fp8 KV + fp8 stack, MAMBA_CACHE=48, 8-way) so
# the comparison isolates MTP alone. Note: without spec decoding, the "MAMBA_CACHE must be
# ~6x MAXREQ" rule no longer binds (it exists for the speculative CUDA graphs) — 48 is kept
# anyway so only the MTP variable differs between the two profiles.
#
# FOREGROUND MODE: the server runs attached to this terminal. Ctrl+C stops it cleanly
# (SIGINT to the whole process group; afterwards a sweep reaps any leftover SGLang GPU
# processes). Output goes to the terminal AND logs/serve.log — run inside tmux/screen if
# you need it to survive a disconnect. No systemd unit, no auto-start.
#   ./serve_best_no_mtp.sh              # start;  Ctrl+C to stop
#   ./serve_best_no_mtp.sh --port 8002  # extra args are appended to the sglang CLI (last-wins)
# Needs the sm120 patches in sglang-official:  bash scripts/apply_patches.sh
#
# Pre-flight: refuses to start while the GPUs are busy (any other LLM server occupying
# the cards — stop it first; offending PIDs are printed).
# Endpoint: http://localhost:1025/v1  (model name: qwen-3.8-flash-next). Thinking is ON by
# default (tokens stream in delta.reasoning_content); pass chat_template_kwargs
# {"enable_thinking": false} to disable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '$2+0 > 4096 {printf "gpu%s=%sMiB ", $1, $2}' || true)"
if [[ -n "$busy" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: GPU memory in use: $busy"
  echo "Stop the occupying server first. Offending processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  echo "(override with FORCE=1)"
  exit 1
fi

mkdir -p logs
echo "Starting sglang in the foreground (TP=${TP:-8}, EP=${EP_SIZE:-8}, 8-way, MTP OFF). Ctrl+C to stop. Logging to logs/serve.log"

export TP="${TP:-8}" EP_SIZE="${EP_SIZE:-8}" MEMFRAC="${MEMFRAC:-0.80}" CTX="${CTX:-262144}" MAXREQ="${MAXREQ:-8}" \
  LINEAR_BACKEND=flashinfer SSM_DTYPE=bfloat16 MAMBA_RADIX=extra_buffer \
  KVDTYPE=fp8_e4m3 SPEC=0 HICACHE=0 \
  SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1 \
  CUDAGRAPH_MAXBS=8 MAMBA_CACHE=48 CPU_OFFLOAD_GB=0 \
  AUTOTUNE=1 MAX_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=2
# SPEC=0 drops the NEXTN/MTP speculative stack entirely (no --speculative-algorithm, no
# FR-Spec draft head, no acceptance-threshold knobs). Patches 0007/0011 (the FR-Spec TP8
# fixes) stay applied but are inert without spec decoding.

trap 'true' INT   # wrapper survives Ctrl+C so the cleanup sweep below still runs
set +e

# Post-ready warmup (background subshell, dies with this wrapper): polls /health, then
# sends one tiny + one long request so the lazy Triton/FlashInfer kernels device-load
# inside the startup window (see serve_best.sh for the full rationale).
if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
{
  for _ in $(seq 1 720); do
    curl -sf "http://127.0.0.1:${PORT:-1025}/health" >/dev/null 2>&1 && break
    kill -0 $PPID 2>/dev/null || exit 0
    sleep 5
  done
  curl -sf "http://127.0.0.1:${PORT:-1025}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-3.8-flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
    >/dev/null 2>&1 \
    && echo "[warmup] short warmup request OK (pool kernels + JIT loaded)" \
    || echo "[warmup] warmup request failed (server still starting or stopped)"
  longfiller="$(tr -dc 'a-z0-9 ' < /dev/urandom 2>/dev/null | head -c 200000)"
  printf '{"model":"qwen-3.8-flash-next","messages":[{"role":"user","content":"%s"}],"max_tokens":1}' "$longfiller" \
    | curl -sf "http://127.0.0.1:${PORT:-1025}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @- >/dev/null 2>&1 \
    && echo "[warmup] long-prefill warmup request OK (sparse GQA + model kernels loaded)" \
    || echo "[warmup] long-prefill warmup request failed"
} &
warmup_pid=$!
fi

bash scripts/serve.sh "$@" 2>&1 | tee logs/serve.log
rc=${PIPESTATUS[0]}
set -e

# Server exited or Ctrl+C — make sure no SGLang processes survived on the GPUs.
sleep 2
sglang_pids() {
  { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
      | awk -F', ' 'tolower($2) ~ /sglang/ {print $1}' || true
    pgrep -f "$ROOT/sglang-official/.venv/bin/sglang serve" 2>/dev/null || true
  } | sort -u
}
pids="$(sglang_pids)"
if [[ -n "$pids" ]]; then
  echo "SGLang left GPU processes behind; terminating: $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r kill 2>/dev/null || true
  sleep 5
  pids="$(sglang_pids)"
  if [[ -n "$pids" ]]; then
    echo "still alive, SIGKILL: $(echo "$pids" | tr '\n' ' ')"
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
fi

echo "sglang stopped (exit $rc)."
exit "$rc"

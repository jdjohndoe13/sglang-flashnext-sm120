#!/usr/bin/env bash
# HiCache PROFILE: serve_best.sh + KV cache offloading to host RAM (128 GB total).
# Qwen3.8-Flash-Next NVFP4, 8x RTX 5090 (sm120, 32 GB/card), TP8 + EP8.
#
# Identical to serve_best.sh except HICACHE=1: sglang's hierarchical cache (HiCache) keeps
# a second cache tier in pinned host RAM and the UnifiedRadixCache evicts/ restores between
# the tiers. For this hybrid model the branch routes automatically through the hybrid-mamba
# stack and attaches host pools for BOTH parts:
#   * the QSA/full-attention KV (QSATokenToKVPool = HybridLinearKVPool), and
#   * the GDN/mamba state checkpoints (MambaPoolHost; the extra_buffer track states).
# MTP/spec decoding stays ON — the MTP+HiCache+hybrid-mamba combo is CI-validated upstream
# on this model lineage (Qwen3-Next / Qwen3.5 hicache suites) and the draft pools have
# explicit host sidecars.
#
# SIZING — --hicache-size is PER RANK: 16 GiB/rank x 8 ranks = 128 GB pinned host RAM total
# (this script's default, matching its name). The launcher splits it proportionally between
# the KV and mamba host pools. Override per-rank size with e.g.:
#   HICACHE_SIZE=32 ./serve_best_kv_128.sh    # 256 GB total
# The box has ~1 TB RAM; keep 8 x HICACHE_SIZE well under it. No NIXL or other extra deps
# are needed (nixl is only for L3 storage backends; the host tier uses the kernel IO backend).
#
# What to expect in logs/serve.log on a good boot:
#   "Tree cache initialized: ... impl=UnifiedRadixCache hybrid_ssm=True hicache_attached=True"
#   "Attached hybrid mamba pool stack ... pools=KV + MAMBA"
#   host-hit metrics appear once evictions start (mamba host hit lengths in request stats).
#
# Known upstream caveats at this branch (sgl-project/sglang ~v0.5.19, all OPEN):
#   #33714 — long prompts only back up their first chunked-prefill window (~4096 tokens with
#            this config) to the host tier at prefill time; the rest is device-tree only.
#            Short prefixes still reach the host tier normally.
#   #36743 — a restored mamba slot can theoretically be consumed before its H2D copy
#            finishes (deferred-COW race). Watch for corrupted-prefix symptoms after
#            host restores; /flush_cache clears the state if seen.
#   #37613 — under host-pool pressure a KV backup can lose its mamba companion (all-or-
#            nothing discard). Sizing here is generous for the device pool; watch for the
#            "host KV pool ... smaller than the device pool" warning if you shrink it.
#
# FOREGROUND MODE: the server runs attached to this terminal. Ctrl+C stops it cleanly
# (SIGINT to the whole process group; afterwards a sweep reaps any leftover SGLang GPU
# processes). Output goes to the terminal AND logs/serve.log — run inside tmux/screen if
# you need it to survive a disconnect. No systemd unit, no auto-start.
#   ./serve_best_kv_128.sh              # start;  Ctrl+C to stop
#   ./serve_best_kv_128.sh --port 8002  # extra args are appended to the sglang CLI (last-wins)
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
echo "Starting sglang in the foreground (TP=${TP:-8}, EP=${EP_SIZE:-8}, 8-way, HiCache ${HICACHE_SIZE:-16} GiB/rank = $(( ${HICACHE_SIZE:-16} * 8 )) GB total host tier). Ctrl+C to stop. Logging to logs/serve.log"

export TP="${TP:-8}" EP_SIZE="${EP_SIZE:-8}" MEMFRAC="${MEMFRAC:-0.80}" CTX="${CTX:-262144}" MAXREQ="${MAXREQ:-8}" \
  LINEAR_BACKEND=flashinfer SSM_DTYPE=bfloat16 MAMBA_RADIX=extra_buffer \
  KVDTYPE=fp8_e4m3 SPEC=1 HICACHE=1 HICACHE_SIZE="${HICACHE_SIZE:-16}" \
  GDN_MTP_CACHE_MODE=none \
  SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1 \
  CUDAGRAPH_MAXBS=8 MAMBA_CACHE=48 CPU_OFFLOAD_GB=0 \
  AUTOTUNE=1 MAX_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=2
# MEMFRAC journey: 0.90 -> capture OOM (<100 MiB free, 2026-09-10 23:50); 0.85 -> text clean
# (245,109-token prompts at bs 1/2/4/8, user-validated) but on-GPU image preprocessing in the
# tokenizer process OOM'd with <100 MiB free; user-validated default now 0.80. With the pil
# image backend (CPU-only preprocessing, default in serve.sh) 0.85 likely works again — try it
# if you want the bigger KV pool.

trap 'true' INT   # wrapper survives Ctrl+C so the cleanup sweep below still runs
set +e

# Post-ready warmup (background subshell, dies with this wrapper): polls /health, then
# sends one tiny + one long request so the lazy Triton/FlashInfer kernels device-load
# inside the startup window (same rationale as serve_best.sh).
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

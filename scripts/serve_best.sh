#!/usr/bin/env bash
# DEFAULT PROFILE — Qwen3.8-Flash-Next NVFP4 on 8x RTX 5090 (sm120, 32 GB/card).
# TP8 + EP8 (expert parallel MoE): pure TP8 is IMPOSSIBLE for this checkpoint —
# moe_intermediate_size 640 sharded 8-way = 80/rank requires NVFP4 w13-scale swizzle
# padding, unsupported for gated experts (load-time assert). EP8 keeps the full 640
# per expert (512 experts -> 64/rank). Interactive + agents: 8-way concurrency, fp8 KV,
# fp8 dense stack, native 262144 ctx.
# Fallback (env-overridable; uses 4 cards, the other 4 idle — untested):
#   TP=4 EP_SIZE=4 ./serve_best.sh    # 4-way + EP4 (shared expert fuses into the EP MoE)
#
# FOREGROUND MODE: the server runs attached to this terminal. Ctrl+C stops it cleanly
# (SIGINT to the whole process group; afterwards a sweep reaps any leftover SGLang GPU
# processes). Output goes to the terminal AND logs/serve.log — run inside tmux/screen if
# you need it to survive a disconnect. No systemd unit, no auto-start.
#   ./serve_best.sh                     # start;  Ctrl+C to stop
#   ./serve_best.sh --port 8002         # extra args are appended to the sglang CLI (last-wins)
# For one huge session (786,432-token context): ./serve_single.sh
# Needs the six sm120 patches in sglang-official:  bash scripts/apply_patches.sh
#
# Pre-flight: refuses to start while the GPUs are busy (any other LLM server occupying
# the cards — stop it first; offending PIDs are printed).
# Endpoint: http://localhost:1025/v1  (model name: qwen-3.8-flash-next). Thinking is ON by
# default
# (tokens stream in delta.reasoning_content); pass chat_template_kwargs {"enable_thinking":
# false} to disable.
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
echo "Starting sglang in the foreground (TP=${TP:-8}, EP=${EP_SIZE:-8}, 8-way). Ctrl+C to stop. Logging to logs/serve.log"

export TP="${TP:-8}" EP_SIZE="${EP_SIZE:-8}" MEMFRAC="${MEMFRAC:-0.80}" CTX="${CTX:-262144}" MAXREQ="${MAXREQ:-8}" \
  LINEAR_BACKEND=flashinfer SSM_DTYPE=bfloat16 MAMBA_RADIX=extra_buffer \
  KVDTYPE=fp8_e4m3 SPEC=1 HICACHE=0 \
  GDN_MTP_CACHE_MODE=none \
  SGLANG_SM120_LOWM_FP8_WEIGHT=1 SGLANG_SM120_LM_HEAD_FP8=1 \
  CUDAGRAPH_MAXBS=8 MAMBA_CACHE=96 CPU_OFFLOAD_GB=0 \
  AUTOTUNE=1 MAX_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=2
# MEMFRAC journey: 0.90 -> capture OOM (<100 MiB free, 2026-09-10 23:50); 0.85 -> text clean
# under the OLD image backend but DISPROVEN 2026-09-14 with HiCache + pil: device pool leaves
# only ~1.0 GB and the first long prefill CUDA-OOMs (logs/serve-hicache-80.log). 0.80 is the
# only validated value — do not raise it.

trap 'true' INT   # wrapper survives Ctrl+C so the cleanup sweep below still runs
set +e

# Post-ready warmup (background subshell, dies with this wrapper): polls /health, then
# sends one tiny chat request. This forces the lazy Triton pool kernels (alloc_extend,
# assign_req_to_token_pool) and first-token JIT paths to load INSIDE the startup window —
# otherwise they device-load on the user's first real request ("Triton kernel ...
# device-loaded after serving started" warnings) and also pollute the first bench.
# SKIP_WARMUP=1 disables it (used by scripts/repro_crash.sh for deterministic replays).
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
  # Second request: long prompt (~40-50k tokens) so the long-context model kernels
  # (_sparse_gqa_chunk_prefill, _sparse_gqa_prefill, _fused_slot_copy, ...) device-load
  # now too — those only trigger on long prefill, not on the tiny request above.
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

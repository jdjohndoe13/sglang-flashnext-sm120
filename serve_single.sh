#!/usr/bin/env bash
# SINGLE-SESSION LONG-CONTEXT PROFILE — Qwen3.8-Flash-Next NVFP4, TP8+EP8, 8x RTX 5090 (sm120).
# (TP8 + EP8 because pure TP8 is impossible for this checkpoint: moe_intermediate 640 -> 80/rank
#  needs NVFP4 swizzle padding, unsupported for gated experts. Fallback: TP=4 EP_SIZE=0.)
# Trades the fp8 dense-weight copies (~20% decode speed) for the biggest possible KV pool,
# and extends rope with YaRN (factor = CTX/262144) -> 786,432-token context window default.
# With TP8+EP8 the aggregate KV pool is ~8x the single-96GB-card ceiling, so larger CTX (e.g.
# CTX=1048576, YaRN x4) should fit — unvalidated; walk up gradually and re-run the needle gates.
# Use serve_best.sh instead for the max-throughput / 8-way concurrency profile.
#
# FOREGROUND MODE: the server runs attached to this terminal. Ctrl+C stops it cleanly
# (SIGINT to the whole process group; afterwards a sweep reaps any leftover SGLang GPU
# processes). Output goes to the terminal AND logs/serve.log — run inside tmux/screen if
# you need it to survive a disconnect. No systemd unit, no auto-start.
#   ./serve_single.sh                   # start;  Ctrl+C to stop
#   CTX=1048576 ./serve_single.sh       # bigger window (YaRN x4, unvalidated)
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
CTX="${CTX:-786432}"
FACTOR="$(awk "BEGIN{printf \"%.4f\", ${CTX}/262144}")"
ROPE="{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":${FACTOR},\"original_max_position_embeddings\":262144}}}"

echo "Starting sglang in the foreground (single-session, TP=${TP:-8} EP=${EP_SIZE:-8}, ctx=$CTX YaRN x$FACTOR). Ctrl+C to stop."

export TP="${TP:-8}" EP_SIZE="${EP_SIZE:-8}" MEMFRAC="${MEMFRAC:-0.92}" CTX="$CTX" MAXREQ="${MAXREQ:-2}" \
  LINEAR_BACKEND=flashinfer SSM_DTYPE=bfloat16 MAMBA_RADIX=extra_buffer \
  KVDTYPE=fp8_e4m3 SPEC=1 HICACHE=0 \
  GDN_MTP_CACHE_MODE=none \
  SGLANG_SM120_LOWM_FP8_WEIGHT=0 \
  CUDAGRAPH_MAXBS=2 MAMBA_CACHE=12 CPU_OFFLOAD_GB=0 \
  ROPE_OVERRIDE="$ROPE" \
  AUTOTUNE=1 MAX_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=2

trap 'true' INT   # wrapper survives Ctrl+C so the cleanup sweep below still runs
set +e
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

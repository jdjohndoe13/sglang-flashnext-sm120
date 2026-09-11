#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 at TP8 on 8x RTX 5090 (sm120, 32 GB/card) — machine
# "testcomp2", official sglang qwen4-main-squashed branch, editable venv inside this repo.
# Requires the six sm120 patches applied to sglang-official once after each git pull:
#   bash scripts/apply_patches.sh        # idempotent; no rebuild needed (editable install)
# Safe defaults; the profile launchers (serve_best.sh / serve_single.sh) push the validated
# overrides in via systemd-run env. All paths env-overridable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-$ROOT/sglang-official}"
SGLANG="${SGLANG:-$REPO/.venv/bin/sglang}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
CACHE_BASE="${CACHE_BASE:-$ROOT/cache}"
PORT="${PORT:-1025}"                # user-requested low-port; anything in 1024-65535 works
SERVED_NAME="${SERVED_NAME:-pennyroyal,glm-5.3-flash}"
# ^ comma-separated names: patch 0008 lists ALL of them on /v1/models and accepts any of
# them by name; the first is canonical (used in spans/metrics). requests may address the
# model as "pennyroyal" or "glm-5.3-flash" without touching other tools' configs.
[[ -x "$SGLANG" ]] || { echo "ERROR: sglang launcher not found at $SGLANG — set REPO=/path/to/sglang-checkout (with .venv) or SGLANG=/path/to/launcher"; exit 1; }

# ---- tunable knobs (safe defaults) ----
TP="${TP:-8}"                       # 8x RTX 5090 -> TP8 (~10 GB NVFP4 weights/card)
EP_SIZE="${EP_SIZE:-0}"             # >0: MoE expert parallel (--ep-size; must divide TP).
                                    # REQUIRED at TP8: moe_intermediate_size 640 -> 80/rank under
                                    # pure TP8 needs NVFP4 w13/w2-scale swizzle padding, which is
                                    # unsupported for gated experts -> load-time assert
                                    # (modelopt_quant.py "padding ... gated activations").
                                    # EP8 keeps the full 640 per expert (64 experts/rank).
                                    # NOTE: per-rank intermediate must be a multiple of 64 for the
                                    # NVFP4 swizzle -> pure-TP only TP 1/2/5 are valid; TP4/TP8 are
                                    # not. On 8x32GB, no pure-TP config fits, so EP8 is the only
                                    # all-GPU configuration.
CTX="${CTX:-262144}"                # native window (full rope); YaRN beyond 262144
MEMFRAC="${MEMFRAC:-0.90}"          # start here on 32 GB cards; OOM at load -> 0.88/0.85
MAXREQ="${MAXREQ:-8}"
LINEAR_BACKEND="${LINEAR_BACKEND:-triton}"   # safe: triton;  perf: flashinfer (sm120, patches applied)
SPEC="${SPEC:-1}"                   # 1 = enable native NEXTN MTP, 0 = disable
HICACHE="${HICACHE:-0}"             # 0 = off (no NIXL);  1 = enable hierarchical cache
CUDAGRAPH_MAXBS="${CUDAGRAPH_MAXBS:-8}"   # must be >= MAXREQ. On a box with a desktop
                                          # session holding VRAM, keep this small (see docs).
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"     # offload N GB of weights to host RAM for extra VRAM headroom (costs throughput)
AUTOTUNE="${AUTOTUNE:-0}"                  # 0 = disable flashinfer autotune (avoids the parallel-cicc RAM storm); 1 = enable (only with low MAX_JOBS)
KVDTYPE="${KVDTYPE:-auto}"                 # auto (fp16/bf16, triton-safe) or fp8_e4m3 (needs flashinfer backend; crashes triton GDN/QSA)
# MAMBA_CACHE defaults to 6x MAXREQ below (--max-mamba-cache-size): must be ~6x or the
# speculative CUDA graphs silently cap concurrency (8-way used to run slower than 4-way).
# FlashInfer GDN on sm120 (unpatched official branch): server_args REQUIRES bf16 SSM state on SM100+, but the
# radix-cache state-checkpoint plan (built only under --mamba-radix-cache-strategy extra_buffer + track-interval)
# demands fp32 on sm120 -> contradiction = jpezzulli's 280825c3e2 patch. Sidestep: bf16 SSM + no_buffer (no
# track mask -> no checkpoint plan). Cost: no GDN prefix-STATE caching across turns (TTFT on multi-turn), not decode speed.
SSM_DTYPE="${SSM_DTYPE:-bfloat16}"
# NOTE: no_buffer is NOT viable for this model (compressed QSA forces page-size 64; no_buffer needs page 1).
# FlashInfer GDN on sm120 is therefore blocked on the unpatched branch (sm120 DSL kernel is fp32-state, sglang demands
# bf16 for flashinfer decode, and extra_buffer tracking needs checkpoints) -> needs jpezzulli's WY-output-only patch.
# Known-good: LINEAR_BACKEND=triton. Keep extra_buffer always.
MAMBA_RADIX="${MAMBA_RADIX:-extra_buffer}"

mkdir -p "$CACHE_BASE"/{huggingface,torch,torchinductor,triton,flashinfer,sglang/jit}
export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 TORCH_CUDA_ARCH_LIST=12.0
# venv bin FIRST: sglang's JIT kernel builder (sglang/kernels/jit/.../ninja.py) shells out
# to `ninja` — without the venv on PATH it dies in CUDA-graph warmup with
# "FileNotFoundError: [Errno 2] No such file or directory: 'ninja'" (hit 2026-09-10).
export PATH="$REPO/.venv/bin:/home/user/.cargo/bin:/usr/local/cuda/bin:$PATH"
export HF_HOME="$CACHE_BASE/huggingface" XDG_CACHE_HOME="$CACHE_BASE"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor" TRITON_CACHE_DIR="$CACHE_BASE/triton"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
# CAP kernel-JIT parallelism: unbounded cicc compilers (nproc=128 x ~3GB each) + 50GB PLE => RAM storm => thrash.
export MAX_JOBS="${MAX_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-4}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

args=(
  serve --model-path "$TARGET_MODEL" --load-format safetensors
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" --tp "$TP" \
  --image-processor-backend "${MM_IMG_BACKEND:-pil}"
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype "$KVDTYPE"
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --page-size 64 --max-running-requests "$MAXREQ" --chunked-prefill-size 4096
  --cuda-graph-max-bs "$CUDAGRAPH_MAXBS"
  --mamba-ssm-dtype "$SSM_DTYPE" --max-mamba-cache-size "${MAMBA_CACHE:-$((6 * MAXREQ))}"
  --mamba-radix-cache-strategy "$MAMBA_RADIX"
  --linear-attn-decode-backend "$LINEAR_BACKEND" --linear-attn-prefill-backend "$LINEAR_BACKEND"
  --ple-offload-embedding --trust-remote-code
  --chat-template "$TARGET_MODEL/chat_template.jinja"
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-request-time-stats-logging --enable-metrics --watchdog-timeout 1800
)
[[ "$CPU_OFFLOAD_GB" -gt 0 ]] && args+=( --cpu-offload-gb "$CPU_OFFLOAD_GB" )
[[ "$AUTOTUNE" == "1" ]] || args+=( --disable-flashinfer-autotune )   # default: autotune OFF (its parallel cicc JIT storm OOMs host RAM)
[[ "$MAMBA_RADIX" == "extra_buffer" ]] && args+=( --mamba-track-interval 64 )   # state tracking only exists for extra_buffer
if [[ "$EP_SIZE" -gt 0 ]]; then
  [[ $((TP % EP_SIZE)) -eq 0 ]] || { echo "ERROR: TP=$TP is not divisible by EP_SIZE=$EP_SIZE"; exit 1; }
  args+=( --ep-size "$EP_SIZE" )
fi
# RecoverSSM / WY output-only MTP verify (ported jpezzulli 280825c3e2, branch sm120-wy). jpezzulli: "none".
[[ -n "${GDN_MTP_CACHE_MODE:-}" ]] && args+=( --gdn-mtp-cache-mode "$GDN_MTP_CACHE_MODE" )
# MTP depth: jpezzulli = 3 steps / 4 draft tokens — and that is the MAXIMUM on this model/branch:
# SPEC_DRAFT must be <= the QSA compress ratio (4): "Qwen QSA requires speculative_num_draft_tokens <= the QSA compress
# ratio (4): the pending index-key ring holds one group" (SPEC_STEPS=4 -> 5 draft tokens fails at startup, tested 2026-08-30).
SPEC_STEPS="${SPEC_STEPS:-3}"; SPEC_DRAFT="${SPEC_DRAFT:-$((SPEC_STEPS+1))}"
# Relaxed MTP acceptance (2026-08-31): force-accept a draft token the target gives >= this prob
# instead of coin-flipping it. LOSSY at temp>0 (sharpens toward high-prob tokens; exact at temp 0).
# Ladder measured at temp 0.6 (C1 median): 1.0 lossless = 179 - 0.5 = 203 - 0.3 = 231 tok/s.
# 0.3 passes greedy/needles/8x GSM/code/French gates; raise toward 1.0 if outputs feel dull/sharp.
SPEC_ACCEPT_SINGLE="${SPEC_ACCEPT_SINGLE:-0.3}"; SPEC_ACCEPT_ACC="${SPEC_ACCEPT_ACC:-0.3}"
# FR-Spec (2026-08-31): draft lm_head scores only a 64K hot-token subset (vocab is 248K) ->
# draft logits ~4x cheaper; verification stays exact so only draft quality could dip (accept
# length measured unchanged, gates + French pass). C1 200.8->219.4, C4 541->592 (temp 0.6).
# Set SPEC_TOKEN_MAP=none to disable. Map = 32K base BPE + top code-corpus tokens + specials
# (ships in this repo root; regenerate with scripts/make_hot_tokens.py if needed).
SPEC_TOKEN_MAP="${SPEC_TOKEN_MAP:-$ROOT/hot_tokens_64k.pt}"
[[ "$SPEC" == "1" ]] && args+=( --speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk 1 --speculative-num-draft-tokens "$SPEC_DRAFT" --speculative-draft-model-quantization unquant
  --speculative-accept-threshold-single "$SPEC_ACCEPT_SINGLE" --speculative-accept-threshold-acc "$SPEC_ACCEPT_ACC" )
[[ "$SPEC" == "1" && "$SPEC_TOKEN_MAP" != "none" && -f "$SPEC_TOKEN_MAP" ]] && args+=( --speculative-token-map "$SPEC_TOKEN_MAP" )
[[ "$HICACHE" == "1" ]] && args+=( --enable-hierarchical-cache --hicache-size 32
  --hicache-host-memory-mode cache --hicache-write-policy write_through --hicache-io-backend kernel )

# Long-context YaRN rope override (factor = CTX/262144 for CTX beyond the native window).
# Fields mirror the checkpoint's rope_parameters with rope_type default->yarn (jpezzulli ran
# factor 2.0 = 524288; the 800K single-session profile uses 3.0 = 786432).
[[ -n "${ROPE_OVERRIDE:-}" ]] && args+=( --json-model-override-args "$ROPE_OVERRIDE" )

# ---- debug instrumentation (all off by default; used by scripts/repro_crash.sh) ----
# DBG_LAUNCH_BLOCKING=1: CUDA launches run synchronously -> the next device-side assert
# reports the TRUE kernel/op in the python traceback instead of an async report at the
# next stream sync (costs ~10-30% throughput while on).
if [[ "${DBG_LAUNCH_BLOCKING:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1
  echo "[dbg] CUDA_LAUNCH_BLOCKING=1 + TORCH_SHOW_CPP_STACKTRACES=1 (slow, exact assert location)"
fi
# DBG_CRASH_DUMP=1: on crash, dump CUDA coredumps + the requests from the last 5 min
# (server_args wires CUDA_COREDUMP_FILE under the folder when --crash-dump-folder is set).
if [[ "${DBG_CRASH_DUMP:-0}" == "1" ]]; then
  export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
  mkdir -p "$ROOT/logs/crashdump"
  args+=( --crash-dump-folder "$ROOT/logs/crashdump" )
  echo "[dbg] --crash-dump-folder $ROOT/logs/crashdump + CUDA user-triggered coredumps"
fi

# Extra CLI args (e.g. from `omega --serve <key> [args...]`) are appended last; argparse last-wins
# so they can override anything above (--port, --served-model-name, --context-length, ...).
echo "sglang ${args[*]} $*"
exec "$SGLANG" "${args[@]}" "$@"

#!/usr/bin/env bash
# One-command crash reproduction runner (wraps scripts/repro_crash.sh).
#
#   bash /mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang/scripts/run_crash_reproduction.sh
#   bash scripts/run_crash_reproduction.sh <req1.json> [req2.json ...] [attempts]   # override
#
# Everything needed is inside; no extra env or flags required. It does, in order:
#   1. Refuses to run while the GPUs are busy (vLLM etc.) — FORCE=1 overrides.
#   2. Relaxes kernel.yama.ptrace_scope so the sglang watchdog's py-spy dumps work on a
#      crash (skipped if already 0; if sudo fails, continues with a warning — it only
#      affects the quality of crash stack dumps, not the reproduction itself).
#   3. Runs scripts/repro_crash.sh with the full debug instrumentation (all on by default):
#      DBG_LAUNCH_BLOCKING=1       exact kernel/op + sync Python stack on the next assert
#                                  (slows decode 2-3x; disable with DBG_LAUNCH_BLOCKING=0)
#      DBG_CRASH_DUMP=1            CUDA coredumps + last-5-min requests into logs/crashdump/
#                                  (disable with DBG_CRASH_DUMP=0)
#      SGLANG_ENABLE_ASYNC_ASSERT=1  in-graph invariant probes (topk_index bounds, NaN/Inf
#                                  on draft logits) — they NAME the failing tensor/bound on
#                                  the next assert (disable with ASYNC_ASSERT=0)
#   and tees everything to logs/repro.log.
# Expected runtime with the default 14-request chain and 30 attempts: each attempt is
# roughly 2-6 min (server start ~2 min warm; decode runs 2-3x slower under
# CUDA_LAUNCH_BLOCKING), so a full 30-attempt run can take 1-3 hours. A crash usually
# happens well before that and the script exits immediately when it does.
# Artifacts when finished (either way):
#   logs/repro.log           console mirror
#   logs/repro_server.log    full sglang server output (the exact assert lives here)
#   logs/repro/              per-request responses
#   logs/crashdump/          crash artifacts (only if it crashed)
# The script starts AND stops the sglang server itself — no Ctrl+C needed; Ctrl+C is
# also safe at any time. All debug env is process-local and dies with the server.
# NOTE: do not send traffic from other clients (opencode etc.) at this server while a
# reproduction is running — it perturbs the replay (concurrency was ruled out as the
# trigger, but determinism matters).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- 1) GPU pre-flight -------------------------------------------------------
if [[ "${FORCE:-0}" != "1" ]]; then
  busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | awk -F', ' '$3 + 0 > 1024 {print $0}' || true)"
  if [[ -n "$busy" ]]; then
    echo "ERROR: GPUs are busy — stop the other server(s) first, then re-run (FORCE=1 to override):"
    echo "$busy"
    echo "Hint: stop the vLLM TP8 server the way you normally start it (e.g. its tmux session / start script),"
    echo "      wait ~10 s, then re-run this command."
    exit 4
  fi
fi

# --- 2) ptrace_scope for the watchdog's py-spy dumps -------------------------
# Harmless to run repeatedly (idempotent write); resets to the distro default on reboot.
if [[ -f /proc/sys/kernel/yama/ptrace_scope ]]; then
  cur="$(cat /proc/sys/kernel/yama/ptrace_scope)"
  if [[ "$cur" != "0" ]]; then
    echo "kernel.yama.ptrace_scope = $cur -> relaxing to 0 (needs sudo once per boot)"
    if sudo sysctl -w kernel.yama.ptrace_scope=0; then
      echo "ptrace_scope=0 set (watchdog py-spy dumps enabled)"
    else
      echo "WARNING: could not set ptrace_scope (sudo refused?) — continuing anyway." >&2
      echo "         If it crashes, the py-spy stack dumps will fail (weaker evidence)." >&2
    fi
  else
    echo "ptrace_scope already 0 — nothing to do"
  fi
fi

# --- 3) run the reproducer ----------------------------------------------------
if [[ $# -eq 0 ]]; then
  # Default: the last 13 real agent turns before the crash (bounded .repro.json copies,
  # max_tokens=1500, rebuilds the deep radix/mamba history of the live session in order)
  # + the pristine crashing request, 30 attempts.
  set -- requests-and-responses/1789088087042.repro.json \
         requests-and-responses/1789088091899.repro.json \
         requests-and-responses/1789088098428.repro.json \
         requests-and-responses/1789088114760.repro.json \
         requests-and-responses/1789088118489.repro.json \
         requests-and-responses/1789088121925.repro.json \
         requests-and-responses/1789088130175.repro.json \
         requests-and-responses/1789088136536.repro.json \
         requests-and-responses/1789088141649.repro.json \
         requests-and-responses/1789088159364.repro.json \
         requests-and-responses/1789088170289.repro.json \
         requests-and-responses/1789088172341.repro.json \
         requests-and-responses/1789088173658.repro.json \
         requests-and-responses/1789088177485.req.json 30
fi

# Debug instrumentation (all on by default; overridable, process-local to the server)
debug_env=()
if [[ "${DBG_LAUNCH_BLOCKING:-1}" != "0" ]]; then
  debug_env+=(DBG_LAUNCH_BLOCKING=1)
fi
if [[ "${DBG_CRASH_DUMP:-1}" != "0" ]]; then
  debug_env+=(DBG_CRASH_DUMP=1)
fi
if [[ "${ASYNC_ASSERT:-1}" != "0" ]]; then
  debug_env+=(SGLANG_ENABLE_ASYNC_ASSERT=1)
fi
if [[ "${TRACK_DUMP:-1}" != "0" ]]; then
  debug_env+=(SGLANG_DEBUG_TRACK_DUMP=1)
fi
if [[ "${EAGER_DRAFT:-0}" = "1" ]]; then
  # 0010f: run the EAGLE draft steps eagerly instead of via the captured
  # decode graph. With DBG_LAUNCH_BLOCKING the IndexKernel assert then
  # surfaces at the exact python op (full stack) instead of inside
  # cudaGraphLaunch. Slower decode; keep 30 attempts or Ctrl+C.
  debug_env+=(SGLANG_DEBUG_EAGER_DRAFT=1)
fi

echo "=== crash reproduction: env ${debug_env[*]} bash scripts/repro_crash.sh $* ==="
rc=0
env "${debug_env[@]}" bash scripts/repro_crash.sh "$@" 2>&1 | tee logs/repro.log || rc=$?
exit "$rc"

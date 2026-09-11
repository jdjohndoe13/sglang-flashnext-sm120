#!/usr/bin/env bash
# One-command crash reproduction runner (wraps scripts/repro_crash.sh).
#
#   bash scripts/run_crash_reproduction.sh
#   bash scripts/run_crash_reproduction.sh <req1.json> [req2.json ...] [attempts]   # override
#
# It does, in order:
#   1. Refuses to run while the GPUs are busy (vLLM etc.) — FORCE=1 overrides.
#   2. Relaxes kernel.yama.ptrace_scope so the sglang watchdog's py-spy dumps work on a
#      crash (skipped if already 0; if sudo fails, continues with a warning — it only
#      affects the quality of crash stack dumps, not the reproduction itself).
#   3. Runs scripts/repro_crash.sh with the debug instrumentation:
#      DBG_LAUNCH_BLOCKING=1 (exact kernel/op on the next device-side assert)
#      DBG_CRASH_DUMP=1      (CUDA coredumps + last-5-min requests into logs/crashdump/)
#      and tees everything to logs/repro.log.
# Artifacts when finished (either way):
#   logs/repro.log           console mirror
#   logs/repro_server.log    full sglang server output (the exact assert lives here)
#   logs/repro/              per-request responses
#   logs/crashdump/          crash artifacts (only if it crashed)
# The script starts AND stops the sglang server itself — no Ctrl+C needed; Ctrl+C is
# also safe at any time. The debug env is process-local and dies with the server.
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
  # Default: the last successful agent turn (KV warmup) + the crashing request, 3 attempts.
  set -- requests-and-responses/1789088173658.req.json \
         requests-and-responses/1789088177485.req.json 3
fi

echo "=== crash reproduction: DBG_LAUNCH_BLOCKING=1 DBG_CRASH_DUMP=1 bash scripts/repro_crash.sh $* ==="
rc=0
DBG_LAUNCH_BLOCKING=1 DBG_CRASH_DUMP=1 bash scripts/repro_crash.sh "$@" 2>&1 | tee logs/repro.log || rc=$?
exit "$rc"

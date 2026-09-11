#!/usr/bin/env bash
# Replay saved OpenAI chat requests (requests-and-responses/*.req.json) against an
# automatically-started sglang server to reproduce a crash under debug instrumentation.
#
# Usage:
#   DBG_LAUNCH_BLOCKING=1 DBG_CRASH_DUMP=1 \
#     bash scripts/repro_crash.sh <req1.json> [req2.json ...] [attempts]
#
# - The request files are POSTed in order per attempt — later requests find earlier
#   ones' KV in the radix tree, like the live multi-turn session. `attempts` (default 3)
#   replays the whole sequence repeatedly: attempt N>1 starts from attempt N-1's cached
#   KV, which is the closest proxy for live conditions.
# - If a crash kills the server, the script reports "SERVER DIED" and stops — the
#   evidence is in logs/repro_server.log (exact kernel stack with DBG_LAUNCH_BLOCKING=1)
#   and logs/crashdump/ (with DBG_CRASH_DUMP=1).
# - The server is started via scripts/serve_best.sh — the validated crash profile
#   (TP8+EP8, MEMFRAC 0.80, fp8 stack) — and restarted automatically if it is not
#   running when an attempt begins. Warmup requests are skipped (SKIP_WARMUP=1) so
#   the replay is deterministic.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-1025}"
LOGDIR="$ROOT/logs/repro"
mkdir -p "$LOGDIR"

# Split args: last numeric arg = attempts, the rest = request files.
reqs=(); attempts=3
for a in "$@"; do
  if [[ "$a" =~ ^[0-9]+$ ]]; then attempts="$a"; else reqs+=("$a"); fi
done
[[ ${#reqs[@]} -gt 0 ]] || { echo "usage: bash scripts/repro_crash.sh <req.json> [...] [attempts]"; exit 1; }
for f in "${reqs[@]}"; do [[ -f "$f" ]] || { echo "missing request file: $f"; exit 1; }; done

health() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

start_server() {
  echo "[repro] patch check, then starting sglang via serve_best.sh (the validated crash profile: TP8+EP8, MEMFRAC 0.80, fp8 stack) ..."
  bash "$ROOT/scripts/apply_patches.sh" || true
  extra_args=()
  if [[ "${NO_OVERLAP:-0}" = "1" ]]; then
    extra_args+=(--disable-overlap-schedule)
  fi
  ( cd "$ROOT" && SKIP_WARMUP=1 bash scripts/serve_best.sh "${extra_args[@]}" ) > "$ROOT/logs/repro_server.log" 2>&1 &
  srv=$!
  echo "[repro] server pid $srv; waiting for /health (up to ~35 min; first start = load + autotune + graph capture) ..."
  local i
  for i in $(seq 1 420); do
    health && { echo "[repro] server ready after ~$((i * 5))s"; return 0; }
    kill -0 "$srv" 2>/dev/null || { echo "[repro] server process died during startup — see logs/repro_server.log"; return 1; }
    sleep 5
  done
  echo "[repro] server not ready after 35 min — giving up"
  return 1
}

stop_server() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "$ROOT/sglang-official/.venv/bin/sglang" 2>/dev/null || true
  sleep 3
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "$ROOT/sglang-official/.venv/bin/sglang" 2>/dev/null || true
  # Safety net for orphaned scheduler/detokenizer children (proctitle "sglang::...");
  # cannot match vLLM ("VLLM::...").
  sleep 2
  pkill -9 -f "sglang::" 2>/dev/null || true
}
trap 'stop_server' EXIT

# Pre-flight: this script must control the server itself (the debug env lives in serve.sh).
if health; then
  echo "ERROR: something already answers /health on port $PORT."
  echo "Stop the running server first — it was not started by this script and may lack the debug instrumentation."
  exit 3
fi

for ((att = 1; att <= attempts; att++)); do
  echo "[repro] ===== attempt $att/$attempts ====="
  if ! health; then
    echo "[repro] server not running — starting it"
    stop_server
    start_server || exit 1
  fi
  n=0
  for f in "${reqs[@]}"; do
    n=$((n + 1))
    out="$LOGDIR/resp.att${att}.seq${n}.json"
    echo "[repro] POST $f (seq $n) ..."
    code=$(curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' --data-binary "@$f" \
      --max-time 3600 -o "$out" -w '%{http_code}' 2>"$LOGDIR/curl.att${att}.seq${n}.err" || true)
    if [[ "${code:-000}" == "200" ]]; then
      echo "[repro]   -> 200 OK, $(wc -c <"$out") bytes -> $out"
    else
      echo "[repro]   -> HTTP ${code:-ERR}; body in $out"
    fi
    if ! health; then
      echo "[repro] SERVER DIED after this request — crash reproduced (attempt $att, seq $n)."
      echo "[repro] ---- tail of logs/repro_server.log ----"
      tail -n 8 "$ROOT/logs/repro_server.log" 2>/dev/null || true
      echo "[repro] exit code 2: pull logs/repro_server.log + logs/crashdump/ for analysis"
      stop_server
      exit 2
    fi
  done
done
echo "[repro] all $attempts attempts passed WITHOUT a crash — rerun with more attempts if desired"

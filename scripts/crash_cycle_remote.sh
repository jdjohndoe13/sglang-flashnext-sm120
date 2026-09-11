#!/usr/bin/env bash
# crash_cycle_remote.sh — remote half of the crash-investigation cycle.
#
# Run via:  ssh testcomp2 "bash /mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang/scripts/crash_cycle_remote.sh"
#
# Steps:
#   1. stop (and if needed kill) every running container of the vllm image
#      -> frees the 8 GPUs for the sglang crash reproduction
#   2. run scripts/run_crash_reproduction.sh (stops on crash, or runs all
#      30 attempts clean); its exit code is echoed and saved to
#      logs/last_repro_exit_code.txt
#   3. restart vllm via /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4.sh
#      (that script is a foreground `docker run`, so it is launched under
#      nohup in the background)
#   4. poll http://127.0.0.1:1025/v1/completions with a "Hi" prompt until it
#      answers 200 OK (model name auto-detected from /v1/models)
#   5. exit -> the ssh session disconnects; the local .bat then notifies
#      the opencode session.

set -uo pipefail

ROOT=/mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang
REPRO="$ROOT/scripts/run_crash_reproduction.sh"
VLLM_SCRIPT=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4.sh
VLLM_PORT=1025
IMAGE="cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1"
POLL_INTERVAL=15
VLLM_TIMEOUT=$((90 * 60))   # seconds to wait for the completions 200 OK

log() { echo "[cycle $(date +%H:%M:%S)] $*"; }

# docker may need sudo depending on group membership
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

# ---- step 1: free the GPU --------------------------------------------------
log "stopping containers of image $IMAGE"
ids=$("${DOCKER[@]}" ps -q --filter "ancestor=$IMAGE" || true)
if [ -n "$ids" ]; then
  log "docker stop: $ids"
  "${DOCKER[@]}" stop $ids >/dev/null 2>&1 || true
fi
sleep 3
ids=$("${DOCKER[@]}" ps -q --filter "ancestor=$IMAGE" || true)
if [ -n "$ids" ]; then
  log "still running after stop, killing: $ids"
  "${DOCKER[@]}" kill $ids >/dev/null 2>&1 || true
fi
sleep 5
ids=$("${DOCKER[@]}" ps -q --filter "ancestor=$IMAGE" || true)
if [ -n "$ids" ]; then
  log "WARNING: containers still up after stop+kill: $ids"
else
  log "vllm container stopped (GPU free)"
fi

# ---- step 2: crash reproduction --------------------------------------------
log "running crash reproduction: $REPRO"
rc=0
bash "$REPRO" || rc=$?
log "run_crash_reproduction.sh exit code: $rc"
echo "$rc" > "$ROOT/logs/last_repro_exit_code.txt"

# ---- step 3: restart vllm ---------------------------------------------------
stamp=$(date +%Y%m%d_%H%M%S)
vllm_log="$ROOT/logs/vllm_start_$stamp.log"
log "starting vllm: $VLLM_SCRIPT (log: $vllm_log)"
nohup bash "$VLLM_SCRIPT" > "$vllm_log" 2>&1 &
vllm_pid=$!

# ---- step 4: poll completions until 200 OK ----------------------------------
log "waiting up to $((VLLM_TIMEOUT / 60)) min for 200 OK on 127.0.0.1:$VLLM_PORT/v1/completions"
deadline=$(( $(date +%s) + VLLM_TIMEOUT ))
ok=0
noted=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  model=$(curl -s --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
  if [ -n "${model:-}" ]; then
    code=$(curl -s --max-time 60 -o /dev/null -w '%{http_code}' \
      -X POST "http://127.0.0.1:$VLLM_PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${model}\",\"prompt\":\"Hi\",\"max_tokens\":1}" || echo 000)
    if [ "$code" = "200" ]; then
      ok=1
      log "completions 200 OK (model=$model) — vllm is serving"
      break
    fi
  fi
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    if [ "$noted" = "0" ]; then
      log "note: vllm launcher pid $vllm_pid exited; continuing to poll (server may be detached)"
      noted=1
    fi
  fi
  sleep "$POLL_INTERVAL"
done

if [ "$ok" != "1" ]; then
  log "ERROR: no 200 OK within $((VLLM_TIMEOUT / 60)) min — last vllm log lines:"
  tail -30 "$vllm_log" 2>/dev/null || true
  exit 3
fi

log "cycle complete"
exit 0

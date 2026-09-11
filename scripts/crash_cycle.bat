@echo off
rem ============================================================
rem  crash_cycle.bat — one-click crash-investigation cycle
rem
rem  1. ssh to testcomp2 and run the remote half
rem     (scripts/crash_cycle_remote.sh on testcomp2):
rem       - stop/kill every container of the vllm image (frees the GPUs)
rem       - run scripts/run_crash_reproduction.sh (crash or 30 clean attempts)
rem       - restart vllm and wait until /v1/completions answers 200 OK to "Hi"
rem     The ssh session ends (disconnect) when the remote script finishes.
rem  2. Notify the opencode session to check the new logs.
rem
rem  Expect this window to stay open for a long time (repro can take hours,
rem  vllm restart several minutes). Exit codes from the remote half:
rem     0 = cycle fully successful (repro ran, vllm back with 200 OK)
rem     3 = vllm did not answer 200 OK within the timeout
rem  To change the vllm wait time, edit VLLM_TIMEOUT in
rem  crash_cycle_remote.sh (on testcomp2 and in the repo scripts folder).
rem ============================================================
setlocal

set HOST=testcomp2
set REMOTE_SCRIPT=/mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang/scripts/crash_cycle_remote.sh
set SESSION=ses_f72b307f3ffeQZQnrN91yNllIw

echo [cycle] connecting to %HOST% — this can run for hours (repro, then vllm restart)...
ssh -o ConnectTimeout=15 -o ServerAliveInterval=60 %HOST% "bash %REMOTE_SCRIPT%"
echo [cycle] disconnected from %HOST% (remote exit code %ERRORLEVEL%).

echo [cycle] notifying opencode session %SESSION% ...
opencode -s %SESSION% --prompt "run_crash_reproduction.sh completed, check the logs"

echo [cycle] done.
endlocal

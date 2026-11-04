# Incident 2026-11-03 19:28–19:30 — SGLang server stall (investigation, 2026-11-03 ~21:30)

All times UTC. Evidence: proxy `stats.jsonl`/`stall.log`/req-dumps
(`b:\intersub\temp\llm-proxy\logs`), docker container metadata, testcomp2
health checks (remote-exec runs `re_20261103-211842-6211`, `re_20261103-212046-1588`).
The old server's own `serve.log` was **lost** (per-boot truncation; the
19:33:33 restart overwrote it) — everything before 19:33 is proxy-side evidence.

## Verdict

1. **The proxy was not the problem.** It forwarded everything it received:
   `stall_s: 0.0` on every entry in the window (the temp guard never fired —
   GPUs were 43–50 °C, far below the 74 °C stall threshold), and the pause
   gate was only engaged ~20:15+ (during the fallback switch, see below).
   The 19:29:05 container recreation was the **user's own restart** (user
   confirmed, thinking the proxy was stuck).
2. **The server was stuck, not dead — stuck in a non-compute path.**
   The 19:28–19:30:43 log silence = scheduler wedged; the first sign of life
   (user's log excerpt) is the 19:30:43 `200 OK` with decode at **0.96 tok/s**
   and `#running-req: 1`.
   **Grafana confirms the user's power observation** (`gpu_power_watts`,
   per-GPU, job `gpu-temp-monitor`, 19:15–19:40 at 1-min steps):
   `sum(gpu_power_watts)` idle ≈ 185–195 W (8× ~24 W), full load ≈ 1200–1650 W.
   The silence window was **mostly idle** — and crucially, the two huge
   requests known to be in the server (188k-tok sent 19:22:50 → done 19:27:10;
   115k-tok sent 19:22:53 → done 19:27:39) sat there while the GPU read
   **194 W / 191 W** at 19:25–19:26. They were not queued behind compute —
   compute was not happening at all. The real prefill of the 189k-tok
   request at 19:27:11 took ~19 s at 539–1311 W (≈10k tok/s), so the
   multi-minute waits were spent in a **non-compute path** — prime suspect
   the HiCache host-tier restore/transfer machinery (the same subsystem
   patch 0014 fixed for correctness), not the GPU.
   The only exceptions to idle: two full-power bursts — **19:28–19:29
   (762→1311 W, the 19:28:31 trio's prefill)** and **19:31 (1188 W)**,
   between which the GPU dropped back to 195 W. So the GPU did compute
   when the scheduler actually ran a batch; the pathological mode was the
   multi-minute idle gaps *while requests were in flight*, ending in the
   0.96 tok/s crawl.
3. **The preceding hour of slowness was load-driven, not hardware.**
   Three long-running OpenCode sessions (kept alive by the subagent-watchdog
   probes — the 4.83 MB/323-message request's last message literally is a
   watchdog probe) grew to 150k–190k-token prompts. Proxy p90 latency:
   15 s (17:00) → 100 s (17:40) → 170–190 s (19:10–19:20); max durations hit
   the 600 s ceiling; concurrency only 3–6. Power corroborates the busy
   hour: 850–1100 W at 17:00–17:25, then bursty 400–1000 W.
4. **No GPU/NCCL/kernel errors**: `dmesg` Xid scan clean (unprivileged
   `dmesg` permitted only partially — no errors visible), journal clean,
   `clocks_event_reasons` idle-only, ECC N/A (consumer cards), no throttling.
5. **The proxy container was recreated 19:29:05** (mid-silence) — the three
   19:28:31 in-flight requests died with it (req dumps, no resp dumps).
   This matches the user's own `restart.sh` action; the clients re-sent the
   same three sessions 19:29:06–19:29:13 and those completed 19:32:25.

## Timeline (UTC)

| time | event |
|---|---|
| 17:00–19:20 | hour-long degradation: prompt median 42k→141k tok; p90 15s→190s; 600 s max durations; first 444s (client timeouts) from 17:50 |
| 19:21:04, 19:23:35 | 4.8 MB requests hit 300 s client timeout → 444 |
| 19:22:50, 19:22:53 | 188k-tok → 260 s, 115k → 285 s, both 200 (server at the edge) |
| 19:27:30, 19:27:39 | two watchdog probes aborted by the client at 59/50 s |
| 19:28:31 | the three big sessions send (1.89/0.74/4.83 MB) → proxy forwards → server **prefills them at full power (762→1311 W, Grafana)** but the log stays silent (no decode-batch lines) until 19:30:43 |
| 19:28:56–19:29:01 | small 200s still complete (old proxy alive) |
| **19:29:05** | **proxy container recreated** (user's restart, confirmed); the 19:28:31 trio lost in-flight (no resp dumps) |
| 19:29:06–19:29:13 | same three sessions re-send to the fresh proxy; GPU idle 19:30 (195 W), one more prefill burst 19:31 (1188 W), then idle 19:32 (194 W) while decode crawls at **0.96 tok/s** |
| 19:30:43 | user's log excerpt: `200 OK` + decode `0.96 tok/s` — server alive, stuck |
| 19:32:25 | the three retries complete (190–200 s) |
| 19:32:36–39 | next retry trio of the three sessions in flight |
| ~19:33 | **user kills/restarts the server** (new boot 19:33:33) → those three abort (444 at 19:33:47/19:37:46/19:37:56) |
| 19:36:32 | new boot **crashes**: patch-0015 MROPE launch bug (fixed later, see AGENTS.md) |
| 19:36–20:17 | **no server**: proxy attempts hang → 600 s (300 s×2) timeout → 444 wave (19:35–20:15, ~14 big requests); dashboard "in-flight" = these held requests |
| ~20:15 | user pauses the proxy (first post-gap records carry `paused_seconds` 110–246 s) |
| ~20:17 | **vLLM docker fallback** grabs GPUs+:1025 → traffic resumes; 11× 400 = vLLM rejecting SGLang-specific request fields |
| 21:00–21:02:58 | patched SGLang boots (gate `pools=KV + MAMBA + INDEXER`), traffic normal |

## Power trace (Grafana `gpu_power_watts`, sum of 8 GPUs, 1-min steps)

Idle ≈ 185–195 W; full load ≈ 1200–1650 W. Selected 5-min averages:

| window | W | what was happening |
|---|---|---|
| 17:00–17:25 | 850–1100 | busy hour (prompt growth start) |
| 18:30–19:15 | 430–700 | bursty, big-prefill spikes |
| 19:20–19:26 | **193–197** | 188k/115k requests IN the server, GPU idle (260–285 s waits) |
| 19:27–19:29 | 539→**1311** | 189k prefill (19 s), then the 19:28:31 trio prefill |
| 19:30 | **195** | the silence — GPU idle |
| 19:31 | **1188** | one more full-power burst |
| 19:33–19:36 | 194–626 | user restart + MROPE crash warmup |
| 19:37–20:00 | 184–190 | server dead (no backend), idle |
| 20:18–20:35 | 1230–1657 | vLLM fallback loading models + serving |
| 20:46+ | 185–297 | fallback idle / patched boot starting |

## What remains unknown

- **Why the huge requests sat idle for minutes before/while computing** —
  its serve.log is gone. The Grafana power trace (see Verdict 2) localizes
  it to a non-compute path: idle GPU while 115k–188k-token requests were
  in the server for 260–285 s, interleaved with full-power bursts when the
  scheduler did run. Prime suspect: the HiCache host-tier restore/transfer
  (patch-0014 subsystem). The leak watcher
  (`scripts/bench/watch_server_leaks.ps1`) is armed for recurrence.
- **Whether the wedge is still latent** — the box has been healthy since
  21:02, but the same three sessions are still growing. If it recurs, the
  per-boot logs (see Recommendations) now capture the server side, and the
  Grafana `gpu_power_watts` trace can again separate "stuck idle" from
  "grinding".

## Recommendations

1. **Session hygiene**: the watchdog probe sessions never compact — the
   4.83 MB/323-msg session was idle-but-probed. Cap/compact long sessions
   (context-length guard) so 150k+ prompts don't pile up.
2. **Preserve serve.log across boots — DONE 2026-11-03 21:34**: all six
   serve scripts (`scripts/serve_best*.sh`, `scripts/serve_single.sh`,
   root `serve_single.sh`) now `tee logs/serve.log logs/serve.<ts>.log`
   — `serve.log` stays the fixed "latest" path (watcher + docs depend on
   it), the per-boot copy survives restarts. Applied to both the local
   `D:\shared` copy and testcomp2's `/mnt/data/shared` copy (verified
   separate copies via probe; `bash -n` clean on both).
3. **vLLM fallback returns 400** to SGLang-shaped requests (agents saw
   400s for ~45 min). Either make the fallback reject with 503 (retryable)
   or gate the fallback behind a flag.
4. Consider `max_prefill_tokens`/`chunked_prefill_size` tuning or a
   per-session concurrency cap so one 190k prompt can't starve decodes.

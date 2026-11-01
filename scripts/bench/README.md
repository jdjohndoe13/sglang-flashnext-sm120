# Hallucination bench — parallel agentic load

`agent_hallucination_bench.mjs` — replays the *shape* of the load that exposed
the HiCache cross-session leakage (see `hallucination-investigation.md`), and
measures the leak class directly.

Default endpoint `http://192.168.2.208:19999/v1` is the LLM proxy that
forwards to the sglang server on testcomp2 (`192.168.2.167:1025`), so every
bench request is also archived under `B:\intersub\temp\llm-proxy\logs\`
for post-run forensics.

## What it does

- **8 lanes × 2 chains** (main + subagent) hit `/v1/chat/completions`
  concurrently — matching the 6–8-parallel-heavy-agent profile that saturated
  the box (`--max-running-requests 32`) during the bad boot.
- **Large shared prefix** (~30 KB system prompt, identical for every lane —
  like the real tool catalog) so the radix cache *shares and offloads* it;
  **session churn** (finish → new session) then forces the prefix to be
  reloaded from the host tier while other lanes are still writing — exactly
  the window the 0012/0013 fences close.
- **Big contexts** (~100 K tokens per session end, grown with synthetic
  tool results) — the leak class only appeared on 100 K+-token requests.
- **Canary detection**: every session carries a unique 8-hex token and
  `/work/<hex>/` paths. A canary appearing in another lane's output =
  `CROSS_SESSION_LEAK` (the phantom-folder class). Any tool name outside the
  declared set = `UNKNOWN_TOOL` (the `not defined in the tools list` class).

## Run

```bash
node scripts/bench/agent_hallucination_bench.mjs --lanes 8 --duration 15
# from Windows with the server on testcomp2:
node scripts\bench\agent_hallucination_bench.mjs --lanes 8 --duration 15 ^
     --base http://192.168.2.208:19999/v1
```

Events → `bench-run.jsonl` (every request, per-lane), summary →
`bench-run.summary.json`. Exit code 1 if any leak or unknown-tool hit.

## Duration

15 min is a good **smoke test** (a 2-lane smoke run produces ~33 reqs /
1.2 M input tokens / 2 min). For the 0012/0013 A/B, use **60 min**: the
`kirill`/`c4b7c840` bursts were ~100 min long and the old HICACHE=1 leak
rate was ~0.24/h — an hour of saturated load is the minimum that gives
the race a realistic number of chances.

## Profiles

- Current daily boot (HICACHE=0): expect 0 leaks — this is the baseline.
- HICACHE=1 + fences (after the next boot): expect 0 leaks if the fences
  work; the old HICACHE=1 boot leaked ~23 under ~4 days of real load.
- Watch server-side too: `grep -c "not defined in the tools list"
  logs/serve.log` during the run (the server-side twin of the client check).

## Two-terminal setup (the A/B run)

Terminal 1 — the bench (prints `!!!` lines the moment a leak is detected,
full summary at the end, exit code 1 on any hit):

```bat
cd /d D:\shared\models\sglang-flashnext-sm120
node scripts\bench\agent_hallucination_bench.mjs --lanes 8 --duration 60
```

Terminal 2 — server-side watcher (tails `logs/serve.log` over ssh; baselines
the pre-existing hit counts on start, then alerts on every NEW hit):

```bat
powershell -ExecutionPolicy Bypass -File D:\shared\models\sglang-flashnext-sm120\scripts\bench\watch_server_leaks.ps1
```

A red `!!! SERVER-SIDE UNKNOWN-TOOL` line means the server parsed a tool
call for a tool that was never declared — the server-side twin of the
client's `UNKNOWN_TOOL`. Yellow `[i]` bad-param lines are the
model-internal confabulation class (present on clean boots too);
informational only.

Both alert **immediately** on detection and summarize at the end.

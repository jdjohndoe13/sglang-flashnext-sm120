# Bug: accepted requests vanish from the scheduler without completing or aborting

**Date:** 2026-11-08 (all timestamps UTC, server clock is UTC)
**Reported by:** llm-proxy (fronting reverse proxy) — forensics done from the proxy's
logs + dashboard and the sglang serve log.
**Status:** reproducible in production traffic; two confirmed incidents (12:47 and
13:36 on 2026-11-08), same fingerprint.

## Summary

Under concurrent load from large-context clients (120k–190k-token prompts,
`stream=false` at the HTTP level), the server **accepts a request, the connection
stays open, but the request then disappears from the scheduler**: it never appears
in `#running-req` / `#queue-req` / a prefill batch beyond its initial admission,
never logs a `ReqTimeStats` completion, and never logs an abort when the client
disconnects. The connection just sits there until the client's proxy gives up
after 300 s and drops it. No WARNING/ERROR/watchdog output anywhere near the
incident window.

The proxy's 300 s timeout + identical resend masks the problem: the resend
completes in 2–5 s with ~99 % prefix-cache hits, so the client eventually gets a
200 — after ~9 minutes. Without the proxy's retry the client would have seen a
hard failure.

Secondary observation (separate issue, same logs): the model frequently enters
repetition loops and generates to the 30 000-token completion cap. That is model
behavior, not a scheduler bug, but the request dumps that show it are in the same
proxy log directory — see [Repeating-nonsense dumps](#repeating-nonsense-dumps).

## Environment

- Server: testcomp2 (host `testcomp`), served model `qwen-3.8-flash-next`
  from `/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4`, port **1025**.
- Launch script: `/mnt/data/shared/models/sglang-flashnext-sm120/scripts/serve_best_kv_offload.sh`
  (this repo). Flags in effect: `--tp 8 --ep-size 8
  --speculative-algorithm NEXTN --enable-hierarchical-cache --hicache-size 55
  --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 192
  --watchdog-timeout 1800 --chunked-prefill-size 4096 --page-size 64`.
- sglang log: `/mnt/data/shared/models/sglang-flashnext-sm120/logs/serve.log`
  (currently `serve.20261106-221544.log`).
- Front proxy: llm-proxy on the same host, `/mnt/data/shared/models/llm-proxy`
  (docker container `llm-proxy-python`, dashboard http://192.168.2.167:9999/dashboard).
  - The proxy **rewrites every client request to `stream=false`** and buffers the
    full upstream response (`resp.read()`), so sglang only ever sees non-streaming
    requests.
  - `UPSTREAM_TIMEOUT=300`, `UPSTREAM_RETRIES=1` — on timeout the proxy drops the
    connection and resends the identical request once.
  - Proxy source address seen by sglang: `172.21.0.2` (docker bridge).
- Clients: `opencode` agents (192.168.2.x / 192.168.250.x), typically 4–8
  concurrent sessions, prompts 110k–190k tokens, heavy prefix-cache reuse.

## Incident A — 13:36:42–13:41:51 (the "5 in-flight, only 1 served" case)

### Timeline

| time (UTC) | event |
|---|---|
| 13:36:42–13:36:51 | 4 huge requests arrive at the proxy and are forwarded to sglang: 159473 / 121307 / 165305 / 124001 prompt tokens (IPs 192.168.250.137, 192.168.2.208 ×3) |
| 13:36:42–13:36:43 | the *previous turns* of the very same 4 sessions complete (`ReqTimeStats` input_len 158570 / 120444 / 164588 / 122865 — note the stuck inputs are just ~800–1100 tokens larger, i.e. same conversation, next turn) |
| 13:39:54 | sglang shows `#running-req: 1, #queue-req: 0` while the proxy holds 5 in-flight (the 4 stuck + 1 genuinely decoding) |
| 13:41:42–13:41:51 | proxy logs `upstream did not respond within 300.0s ... (attempt 2 of 2)` ×4 and drops the connections |
| 13:41:42–13:42:07 | proxy resends the identical requests; sglang `ReqTimeStats` for the resends: `input_len=159473, cached_input_len=158528, output_len=74, forward_duration=2237ms` (and 121307→120384, 165305→164544, 124001→122816) — 2–5 s each, ~99 % cache hit |
| 13:41:42–13:47:00 | **no abort lines at all** in serve.log for the dropped first attempts |

Final stats (proxy `stats.jsonl`, all `status 200`, `upstream_retries: 1`):

```
13:36:42  prompt 159473  total 546.8s  completion 49
13:36:44  prompt 121307  total 511.8s  completion 223
13:36:45  prompt 165305  total 540.5s  completion 232
13:36:51  prompt 124001  total 549.7s  completion 1351
```

### What the sglang log does NOT show (the anomaly)

For the four first attempts (entry_time ≈ 1794145002–1794145011):

- no `ReqTimeStats` with `entry_time` in that window (grep `entry_time=17941450[0-9][0-9]`
  only shows the *previous* turns, e.g. input_len 120444/164588, which completed
  normally);
- no `#queue-req: >0` samples in 13:36–13:42, and `#running-req` distribution over
  13:36–13:42 is 409×1 / 193×2 / 17×3 / 4×4 / 2×5 — never the 4 stuck + others;
- no WARNING/ERROR/CRITICAL lines 13:30–13:45; no hicache/backup/evict/retract
  lines; no watchdog output;
- no abort lines when the proxy dropped the connections at 13:41:42–51;
- no HTTP access line (not even a 499) for the four first attempts.

Yet the resends hit ~99 % prefix cache, so the first attempts **did complete
prefill** (the KV is in the radix cache) — and then the requests vanished from
the scheduler without decoding, without finishing, and without being aborted
when the client connection died.

## Incident B — 12:47 (same pattern, one hour earlier)

- Same 4-session pattern; proxy retries at 12:52:49–12:53:48 (same 300 s timeout
  warnings).
- At 12:55:00 sglang logged exactly one abort, `Abort request ... not found in
  rid_to_state` — i.e. the request id was still registered in the tokenizer
  manager's map but the scheduler had already lost it. That is the strongest
  hint that the request object is dropped on the scheduler side while the
  HTTP/tokenizer-manager side still holds the connection.
- `#running-req` distribution 12:48–12:52: 223×1 / 176×2 / 83×3 / 21×4 / 2×5 —
  again, the four stuck requests never appear.

## Repeating-nonsense dumps

The proxy logs every request/response body to `/mnt/data/shared/models/llm-proxy/logs/`
as `<ms-timestamp>.req.json.xz` / `<ms-timestamp>.resp.json.xz` (recent files are
uncompressed `.json`; a rotation job compresses older ones to `.xz`). The
`stats.jsonl` entries with `completion_tokens: 30000` are the repetition-loop
runs (30 of them so far). The most recent examples:

```
2026-11-07T23:12:30Z  ts 1794093150317  prompt 107028
2026-11-08T04:36:57Z  ts 1794112616877  prompt 107878
2026-11-08T10:05:39Z  ts 1794132338917  prompt 134235
2026-11-08T10:53:43Z  ts 1794135222943  prompt 152018
2026-11-08T11:00:37Z  ts 1794135637431  prompt 154433
2026-11-08T13:52:39Z  ts 1794145959398  prompt 100361
```

Read one with e.g. `xz -dc /mnt/data/shared/models/llm-proxy/logs/1794145959398.resp.json.xz | head -c 200000`
(the repeated text is in `choices[0].message.content`). These are model-quality
evidence, not scheduler evidence — but the same dumps are how you reproduce the
load: the stuck requests' prompts are long agentic conversations, and the loops
show up on the same workload.

## Where the evidence lives (all on testcomp2)

| what | where |
|---|---|
| sglang server log | `/mnt/data/shared/models/sglang-flashnext-sm120/logs/serve.log` |
| proxy access log + warnings | `docker logs llm-proxy-python` (the 300 s timeout warnings, client access lines) |
| proxy per-request stats | `/mnt/data/shared/models/llm-proxy/logs/stats.jsonl` (`upstream_retries: 1` = the stuck+resent requests; `completion_tokens: 30000` = loops) |
| proxy stall log | `/mnt/data/shared/models/llm-proxy/logs/stall.log` |
| full request/response dumps | `/mnt/data/shared/models/llm-proxy/logs/<ts>.req.json.xz` / `<ts>.resp.json.xz` |

The four stuck requests' dumps (ts = proxy entry time, ms; the `.resp.json.xz`
belongs to the successful resend):

```
1794145002209  -> prompt 159473 (13:36:42.209)
1794145003619  -> prompt 121307 (13:36:43.619)
1794145005438  -> prompt 165305 (13:36:45.438)
1794145011237  -> prompt 124001 (13:36:51.237)
```

Useful greps (serve.log):

```bash
SGL=/mnt/data/shared/models/sglang-flashnext-sm120/logs/serve.log
# the four stuck requests: nothing should come back
grep -a 'entry_time=17941450[0-9][0-9]' $SGL | grep -a ReqTimeStats
# the four resends, ~99% cache hits
grep -a 'input_len=159473' $SGL | grep -a '13:4[12]'
# the one abort that DID land (12:55:00)
grep -a 'not found in rid_to_state' $SGL
# aborts in the incident window (none expected)
grep -aE '2026-11-08 13:4[1-7]' $SGL | grep -ai abort
```

## Reproduction

- Replay the four `.req.json.xz` dumps above against `:1025` (set
  `stream: false` and drop `stream_options` — that is what the proxy sends).
  The strongest single scenario: two requests in the same conversation where
  the second (≈ previous prompt + previous output + tool results) is submitted
  within ~1–2 s of the first completing — that is exactly what the four stuck
  requests were (previous turns completed at 13:36:42–43, stuck requests
  arrived 13:36:42–51).
- The proxy's dashboard (http://192.168.2.167:9999/dashboard) now marks retried
  requests with a `↻` badge and shows a `sent` state (request dispatched, no
  response headers yet), so a live recurrence is visible at a glance.

## Hypotheses / pointers for the sglang maintainer

1. The request dies between the tokenizer-manager (`rid_to_state` still holds
   the rid) and the scheduler — the 12:55:00 "not found in rid_to_state" abort
   shows the two sides disagree.
2. The first attempts completed prefill (99 % cache hits on resend) but never
   entered decode — suspect the state transition after prefill for huge
   prompts on this build: hybrid mamba radix cache (`--mamba-radix-cache-strategy
   extra_buffer`, `--max-mamba-cache-size 192`), HiCache (`--enable-hierarchical-cache
   --hicache-size 55`), and NEXTN speculative decoding all interact here.
3. The client disconnect (proxy drop at 13:41:42–51) not producing an abort
   means the request is not in the scheduler's abort path either — a leak:
   the socket stays open forever.
4. All four stuck requests are ≥120k tokens; the incident happened while the
   server was otherwise healthy (300–1000 tok/s decode, no watchdog).

## Questions

1. Is there a known path where a request is silently dropped after prefill
   (e.g. mamba-cache admission failure, retract-to-queue failure) without
   logging and without aborting?
2. Should the `not found in rid_to_state` abort at 12:55:00 be considered
   the smoking gun — i.e. the scheduler already evicted the rid while the
   HTTP layer still waits?
3. Any recommended logging (e.g. `--log-level debug` for the scheduler) to
   catch the next occurrence without disturbing production?

# Zombie request after client disconnect (2026-11-14) — patch 0017

## Symptom

Interrupt a streaming request client-side (Ctrl+C on the tool, no llm-proxy). The
server then prints, for minutes:

    Received output for rid='...' but the state was deleted in TokenizerManager.

2026-11-14 incident: rid `bca912965bfe462d9c4359614f353e15` — 52,254 lines at
~70/s from 19:01:10 to 19:14:21. The spam ended exactly when the request's
`ReqTimeStats` line appeared: `input_len=9492, output_len=252650` (9492+252650 =
262142 ≈ the 262144 context cap), `forward_duration=970414 ms` (16.2 min). The
scheduler had been decoding the abandoned request for 16 minutes — until it hit
the context cap — and the tokenizer manager logged one error per output message
per decode step.

## Root cause (upstream sgl-project/sglang#36333, regression of the #34160 revert)

Streaming path, client disconnect:

1. The HTTP stream generator is cancelled → `asyncio.CancelledError` (a
   `BaseException`) unwinds `TokenizerManager.generate_request`.
2. Its `except BaseException` cleanup calls `_discard_pending_req_states(obj)` →
   `rid_to_state[rid]` is popped immediately.
3. The `create_abort_task` background task (attached to the StreamingResponse)
   fires 2 s later and calls `abort_request(rid)` — but its guard
   `rid not in rid_to_state → return` now silently skips the `AbortReq` to the
   scheduler.
4. The scheduler never learns; the request keeps running as a zombie
   (`#running-req` still counted it; every step's output hits the missing-state
   error branch). It dies only at the context cap (or EOS — which never comes
   in a repetition loop).

GPU load while spamming: the zombie is a real decode — bs=1, ~350 tok/s
(accept rate ~1.0 on the loop), cuda graph on, 8-rank TP traffic. The 16:09
window of the previous boot (ghost alone, `#running-req: 1`) shows the same
~350 tok/s at ~99% GPU utilization. The log line itself is trivial CPU work;
the load is the zombie decode.

Non-streaming requests are NOT affected: the disconnect is detected in
`_wait_one_response` (type 1/3 checks, ≤4 s) which calls `abort_request`
BEFORE raising, so the scheduler abort is delivered. (That is why the old boot
only ever logged single "Abort request ... not found" INFO lines, never spam.)

## Fix — patch 0017 (port of merged upstream PR #35255)

`tokenizer_manager.py`: `ReqState` gains `dispatched`/`abort_sent`; the send
paths mark dispatched; the disconnect/failure cleanup becomes
`_release_req_states_on_failure(request_rids)` — dispatched requests are
**aborted first** (one INFO line) and their state is retained until the
scheduler's abort echo removes it; undelivered states are still dropped.
`abort_request` dedups via `abort_sent`. `create_abort_task` skips vanished
rids. Applies to /generate, /v1/chat/completions, /v1/completions, the
Anthropic shim — all share the same abort path.

Verified: `py_compile` OK; the patch also applies to the pristine HEAD file
(`patch -p1 --dry-run`), so `apply_patches.sh` stays order-independent for
0008. Takes effect on the next `llmqwen` restart. After restart the expected
behavior on a client interrupt: ONE line
`Client disconnected for rid=...; aborting dispatched request.` and the
decode stops within ~2 s (watch `#running-req` drop).

## Related (same day, different issue)

- The 16:10 restart failure: `EADDRINUSE` on the torch.distributed rendezvous
  port (39133) — the previous engine's main process was still tearing down
  (GPU memory was already freed, which is why the busy-GPU pre-flight passed).
  The immediate second retry (16:11) succeeded. Wait ~1 min between llmqwen
  restarts after a Ctrl+C that had a zombie in flight.
- The repetition loop itself is model behavior (the loop shows up as accept
  rate ~1.00 at ~350 tok/s). The server cannot detect it; the client should
  set a sane `max_tokens` (the incident request had no effective cap —
  252,650 tokens) and can use `repetition_penalty`/`frequency_penalty`.

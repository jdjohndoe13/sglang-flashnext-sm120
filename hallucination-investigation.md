# Hallucination Investigation — sglang-flashnext-sm120

Status: **RESOLVED (2026-11-02)** — root cause = the QSA indexer state
gap (§14); fixed by patch 0014; the HICACHE=1 A/B rerun PASSED (§15)
and `serve_best_kv_offload.sh` (HICACHE=1) is the daily profile again.
The heavy-load case passed separately (§11). Post-resolution watch:
`watch_server_leaks.ps1` stays armed for a few days; any user-specific
content (real paths/IPs/machine names) in a response = reopen.
Earlier sections are working notes kept for provenance; §14–§15 supersede
the conclusions in §7–§13.
Written at the point where I was asked whether I was stuck.
This is a working notes file, not a conclusion. Every claim below is either
(a) an observation from a file or command in this repo (with provenance), or
(b) a hypothesis explicitly marked as such.

---

## 1. Scope

Investigating reports of the served model **hallucinating** (fabricated
content / confidently wrong continuations) on the `testcomp2` deployment
(Qwen3.8-Flash-Next-NVFP4, 8x RTX 5090 sm120, TP8+EP8, SGLang
`qwen4-main-squashed` @ 4ccff141 + the 10 patches in `patches/`).

## 2. Environment facts (verified)

- Active launch profile appears to be the daily `serve_best_kv_offload.sh`:
  TP8+EP8, fp8 KV (`kv_cache_dtype=fp8_e4m3`), MTP spec decoding ON, HiCache
  host-RAM tier (800 GB total, per-rank size), ctx 262144,
  `max_running_requests=32`, `page_size=64`, `c128_page_size=16`,
  `MAX_TOTAL_TOKENS=1048576` (the 2026-10-27 OOM fix). Source: `output.txt`
  (server internal_state dump) + `serve_single.sh` / `README.md` env knobs
  (`SPEC=1`, `MAMBA_RADIX=extra_buffer`, `GDN_MTP_CACHE_MODE=none`,
  `MAMBA_CACHE=12`, `TRACK_INTERVAL=2048`).
- `max_total_num_tokens: 1523200` in `output.txt` (pool before the cap; see
  README 32-way note for the cap history).
- The model is a **GDN (gated-delta-net) hybrid**: attention KV + recurrent
  GDN/mamba state. Speculative decoding is EAGLE-style
  (`eagle_worker_v2.py` in the crash traceback) with the FR-Spec 64k
  hot-token map (`hot_tokens_64k.pt`).
- The analysis environment cannot reach the server: `curl` to
  `http://127.0.0.1:1025/health` is refused (exit 7) — the server runs on
  `testcomp2`, not this machine; no live log/health data was available.

## 3. Evidence gathered

### 3.1 crash.log — the 2026-10-27 crash (fixed, and NOT the hallucination)

- At 10:25:52 the scheduler died with a CUDA OOM in `expandable_segments`
  on device 0 during an **extend** step, inside the EAGLE target-verify
  path (`eagle_worker_v2` → `tp_worker` → `eager_runner._execute_extend`).
  Token usage was only **0.18** — the KV pool was NOT the victim; the
  activation headroom was. Matches the README note: at 32-way concurrency
  + MEMFRAC 0.80 the first 16k-token extend batch OOMed in the PLE
  short-conv (78 MiB alloc, 54 MiB device-free).
- Fix already in place: `MAX_TOTAL_TOKENS=1048576` in
  `serve_best_kv_offload.sh` (README, 32-way concurrency section).
- The rest of the log is the NCCL/TCPStore teardown cascade after the
  scheduler died (all ranks report the store closed) — expected, not a
  second fault.
- **Conclusion:** this crash is a known, fixed OOM. If hallucinations were
  observed *during* a crash/teardown, they would be a different (already
  explained) symptom; the ongoing investigation targets hallucinations on a
  *healthy* server.

### 3.2 README — the structural host-tier limit (prime suspect)

README (Validated 2026-09-14 overflow/restore suite):

- Device-tier hits are **bit-identical at temp 0** — no corruption there.
- Host-tier restore works, but the host **mamba** slice is only ~3.6% of
  the tier (~46 slots at 128 GB) while one 121k-token session churns ~240
  checkpoints (1 per ~506 tok at track interval 2048). **Restore only
  works while churn since the session stayed under ~23k tokens** — i.e.
  reliable for same-session next-turn agent reuse, NOT for cross-session
  reuse after heavy traffic.
- Also: an orphaned 512 GB GLM tier file in `/dev/shm` can squat the host
  pool boot budget; `df -h /dev/shm` is the check.

**Hypothesis H1 (current prime suspect):** under real traffic the
GDN/mamba state checkpoints for a session are evicted/recycled from the
host tier while the attention-KV prefix is *still* served (or a stale
checkpoint is reused). The session then resumes with a recurrent state
that does not match the served prefix — the model is conditioned on a
wrong state and continues "confidently" from it. A GDN hybrid is
*structurally* vulnerable to this in a way a pure-transformer is not: the
KV prefix and the recurrent state are two separate caches that must stay
coherent, and the host tier can lose one and not the other.

### 3.3 Canary round-21 — the KV-LOOKUP / offload-tier probe

`tmp/canary-round21-ext.sh` + `tmp/canary-round21-marker2.sh` probe the
**other** served model (`vllm-glm-5.3-flash-nvfp4` container) for the same
class of failure: they scan `KV-LOOKUP` traces for the red-flag signature
`hit=0 cov=0/172` and fire marker-class probes (`M1`, `M1G`) at
`/tmp/soak-M1.json` / `/tmp/soak-M1G.json`.

- **Data gap:** the last run's output is not in this repo (no `out/` dir;
  `prompt-tests/out/` is absent). The marker results are unknown.
- Note the canary targets the vLLM/GLM deployment, not the sglang/Qwen
  one — keep the two evidence lines separate.

### 3.4 Ruled out (with evidence)

- **Quantization stack** (NVFP4 weights + fp8 KV + fp8 dense stack,
  patches 0004-0006): the 2026-09-14 suite shows device-tier hits
  bit-identical at temp 0, and the README's needle gates pass at
  653k-token depth (start / middle / end). A quantization-fidelity
  problem would fail those first.
- **MTP / speculative decoding:** the A/B (2026-09-14,
  `results/mtp-ab-2026-09-14/`) shows accept length holding 2.3–2.6
  across concurrencies and the MTP output matching the no-MTP probes at
  temp 0; the verify step is exact. (The 0007/0011 patches were about
  *assert crashes*, not content drift.)
- **The 2026-10-27 crash** (see 3.1): fixed, different symptom.
- **mamba cache-size / track-interval tuning** (README): those were
  *throughput* and *re-prefill* problems (silently capped graphs;
  6k-token replies churning the checkpoint pool). The track-interval fix
  at 2048 already addresses the next-turn evict-and-reprefill case on the
  device tier; it does not fix the host-tier slot starvation of 3.2.

## 4. Open questions (the ones that would actually discriminate)

1. **Do the hallucinated outputs correlate with a host-tier restore
   event?** The daily profile logs restore lines; a hallucination
   appearing within ~1 turn of a restore (especially after a churny gap
   or on a cross-session prefix) would nail H1.
2. **Replay test:** take the exact request that produced a hallucination
   (if archived) and replay it with `serve_best.sh` (HiCache OFF) and
   with `--disable-radix-cache`. If it stops reproducing, it is a
   cache-state issue; if it reproduces even on a cold cache, it is a
   model/prompt issue.
3. **Is there a hallucinated output in `prompt-tests/` or `results/`?**
   The `*.req.json` files are inputs; the `out/` bodies are absent — the
   actual hallucination samples are not archived in this repo.
4. **Canary results:** was the round-21 `hit=0 cov=0/172` red flag ever
   observed? If yes, the GLM line needs its own investigation file.

## 5. Proposed next steps

1. **Capture the evidence first:** re-run the workload that produced the
   hallucinations on the daily profile and archive the raw outputs
   (`prompt-tests/out/` or a new `results/halluc-<date>/`), plus the
   server log window covering each hallucination (restore/churn lines).
2. **A/B the cache stack:** same workload, three boots —
   (a) daily (HiCache ON), (b) `serve_best.sh` (HiCache OFF),
   (c) `serve_best.sh` + `--disable-radix-cache`. Hallucination rate per
   boot.
3. If (a) > (b) ≈ (c): H1 confirmed. The README's own note says the fix
   is a **code change** (per-session tail-only mamba backup), not a size
   increase — the mamba share is locked to the device-pool ratio.
   Interim mitigation: raise `HICACHE_SIZE` (helps only the KV side, not
   the mamba share), or route long-lived sessions to `serve_single.sh` /
   `MAXREQ=1` where churn cannot evict them, or disable the host tier for
   agent traffic.
4. If the replay test (Q2) still hallucinates on cold cache: investigate
   the prompt/template side (the `qwen3_coder` tool-call parser +
   `reasoning_parser=qwen3` + `strip_thinking_cache` /
   `enable_strict_thinking` flags in `output.txt`) — a parser/templating
   mismatch can also look like "hallucination" to the caller.

## 6. Answer to "are you stuck?"

Not stuck; blocked on **data access**, not on ideas:

- the server is not reachable from this analysis environment (health
  probe refused), and
- the hallucination samples + last canary results are not in the repo.

The decisive experiment is the three-way boot A/B in 5.2; it needs the
`testcomp2` box (or the archived outputs) to run.

(The access block was later resolved: the `remote-exec` binary
(`--host testcomp2`, per the remote_exec skill) is the working channel.
Measurements below were taken through it on 2026-10-29 23:13 UTC.)

## 7. A/B RESULT (2026-10-29) — HICACHE=0 boot: H1 strongly supported

User created `scripts/serve_best_no_kv_offload.sh` = copy of
`serve_best_kv_offload.sh` with only `HICACHE=1` → `HICACHE=0`
(verified as the correct minimal flip: `serve.sh` gates the four
`--enable-hierarchical-cache/--hicache-*` flags on `HICACHE==1` at
serve.sh:140-141; MAXREQ=32 and MAX_TOTAL_TOKENS=1048576 preserved).
Server restarted at 17:35:57 UTC. Boot cmdline verified: NO
`--enable-hierarchical-cache`; `--max-running-requests 32`,
`--max-total-tokens 1048576` present.

**Signature metric:** server-log lines
`Tool '...' is not defined in the tools list` (unknown-tool calls —
the cross-session leakage class) and `Parsed parameter '...' is not
defined in the tool parameters for tool '...'` (wrong-parameter class).

### 7.1 Numbers

| hour (UTC) | chat reqs | hallucinations | rate |
|---|---|---|---|
| 17 | 365 | 39 | 10.7% |
| 18 | 1488 | 79 | 5.3% |
| 19 | 618 | 1 | 0.16% |
| 20 | 190 | 0 | 0% |
| 21 | 178 | 0 | 0% |
| 22 | 186 | 0 | 0% |
| 23 (partial) | 139 | 0 | 0% |

- Overall: 119/3160 = 3.8%, but ~all inside the first 2 hours.
- Baseline (HICACHE=1 boot, measured 16:45 UTC): ~1.35% overall,
  daytime hours 3–5%, peak 132/hour at 13:00.
- The same heavy agent sessions are running (median request 97k
  tokens, ~3000 requests >100k) — they are now clean.

### 7.2 Signature-mix change (the key observation)

- **Unknown-tool calls: 0 occurrences** (baseline: 23, incl. the
  cross-session smoking guns `c4b7c840`,
  `visual_studio_solution_info`, `tools["win-a"].FileSystem`).
  The cross-session leakage class is GONE with the host tier off.
- Remaining 119 are all parameter-name confusion (`paths`/`context`/
  `glob` on `grep`, `old_string`/`edit_new_string` on `edit`,
  `code` on `bash`) — the model calling real tools with wrong
  schema names; clustered in the cold-start window.

### 7.3 Interpretation

- The 17:00–18:00 spike is explained by the restart: the cache tree
  was wiped (no host tier to restore from), so every resumed session
  re-prefilled its full ~100k-token context at once — a churn storm.
  After the cache re-warmed (~19:00) the rate collapsed to ~0.
- Zero cross-session tool leakage with HiCache off, under the same
  resumed heavy sessions, is consistent with H1: the host-tier
  mamba-state eviction/restore incoherence was the corruption
  vector. The device-tier radix cache alone (still ON) did not
  reproduce the leakage class in this window.
- The parameter-confusion class may be model-internal (schema
  confusion, e.g. ast_grep_search's `paths`/`globs` vs grep's
  `path`), amplified by cold-start re-prefill churn; it needs its
  own baseline comparison.

### 7.4 Caveats / what would falsify

- Only ~5.5 h of data; evening traffic (19:00+) is much lighter than
  the 08:00–14:00 daytime peak that produced 132/hour on the old
  boot. The decisive comparison is the SAME daytime load on this
  boot (tomorrow morning).
- Current device pool is NOT under pressure (token usage ~0.10, 1
  running req). If heavy daytime load still yields zero/low
  signatures, H1 is confirmed as the root cause.
- If the unknown-tool class REAPPEARS at daytime peak with HiCache
  off, the device-tier radix cache (or MTP+GDN state) moves to the
  top of the suspect list and the follow-up is
  `--disable-radix-cache` (5.2c).

### 7.5 If confirmed (H1)

README's own note: the fix is a **code change** (per-session
tail-only mamba backup), not a size increase — the mamba share is
locked to the device-pool ratio. Interim options:
(a) keep the no-KV-offload boot as the daily agent profile (the
    ~100k-token cold re-prefills after restart are the cost;
    steady-state next-turn reuse works via the device tier),
(b) route long-lived sessions to serve_single.sh / low MAXREQ,
(c) per-session tail-only mamba backup upstream change.

### 7.6 Day-2 re-measurement (2026-10-30 07:46 UTC, uptime 14h10m)

Boot re-verified: `BOOT=HICACHE-OFF`, health HTTP 200, no crash/
restart in the window. Same heavy multi-agent traffic all night
(3000+ requests >100k tokens/hour-window totals).

| hour (UTC) | chat reqs | >100k reqs | hallucinations |
|---|---|---|---|
| 29th 23 | 669 | 807 | 6 |
| 30th 00 | 805 | 714 | 1 |
| 30th 01 | 582 | 870 | 0 |
| 30th 02 | 342 | 549 | 0 |
| 30th 03 | 525 | 669 | 23 |
| 30th 04 | 455 | 653 | 24 |
| 30th 05 | 194 | 233 | 2 |
| 30th 06 | 430 | 501 | 33 |
| 30th 07 (partial) | 408 | 447 | 36 |

Whole boot (14h): ~7,435 chat requests, 244 signatures —
**unknown-tool class: 4** (all generic confabulations:
`exec` ×3, `ask_user` ×1 — no cross-session names);
**bad-parameter class: 240** (`paths`/`context`/`glob` on `grep`
dominant, `old_string` on `edit`).

**Verdict per class:**
- **Cross-session leakage (the serious symptom): GONE.** 14 hours
  / ~7,400 requests under heavy load, zero occurrences of the
  baseline's signature tools (`c4b7c840`,
  `visual_studio_solution_info`, `tools["win-*"]`). Strongest
  support yet for H1.
- **Parameter-name confusion: persists** (23–36/h in the 03:00+
  hours). Present at the same rate on the old boot (241 `paths`
  alone), so HiCache was never its cause — model-internal schema
  confusion (Claude-Code-style names on real tools), benign: the
  server coerces the value to a string and agents retry.

**Remaining gap:** the user's original symptom (permission requests
for nonexistent folders) is path hallucination — invisible to the
server-side parser. Needs behavioral confirmation from the user.
If it persists on this boot, next A/B is `--disable-radix-cache`
(5.2c).

### 7.7 Daytime-peak comparison (2026-10-30 11:11 UTC, uptime 17h35m)

The decisive window from §7.4 — the 08:00–11:00 daytime peak on the
HICACHE=0 boot:

| hour (UTC) | chat reqs | >100k reqs | hallucinations | rate |
|---|---|---|---|---|
| 30th 07 | 739 | 826 | 38 | 5.1% |
| 30th 08 | 1624 | 819 | 38 | 2.3% |
| 30th 09 | 1240 | 792 | 18 | 1.5% |
| 30th 10 | 997 | 582 | 1 | 0.1% |
| 30th 11 (partial) | 158 | 223 | 0 | 0% |

- Unknown-tool class: +1 since the 07:46 measurement (5 total:
  `exec` ×4, `ask_user` ×1) — still ZERO cross-session names.
  Old boot at the same hours produced the leakage smoking guns
  and peaks of 132/h.
- Bad-parameter class: 298 total (196 `paths` on `grep`), same
  mix as the old boot — rate now 1–2% at peak and falling to ~0.
- Whole boot: 11,785 requests, 2.57% (all cold-start + bad-param).

**Conclusion: the daytime-peak test PASSED.** Heavy load with
HiCache off does not reproduce the leakage class. H1 is confirmed
as the root cause of the cross-session contamination; the
parameter-confusion class is model-internal and unrelated to
HiCache.

## 8. Day-3 recheck (2026-11-01 07:06 UTC, same HICACHE=0 boot, uptime 2d13h)

Server healthy (HTTP 200, zero errors in last 2000 log lines).
Whole boot: 47,941 chat requests, 1.11% overall signature rate.

Per-day signature classes:

| day | reqs | unknown-tool | bad-param |
|---|---|---|---|
| 10-29 | 3,694 | 0 | 125 |
| 10-30 | 17,654 | 10 | 297 |
| 10-31 | 19,931 | 17 | 66 |
| 11-01 (partial) | 6,711 | 1 | 18 |

**All 28 unknown-tool calls are generic confabulations** —
`exec` ×9, `summary` ×7, `Read` ×5 (capital-R Claude-Code name),
`task_update` ×2, `search` ×2, `print`, `ask_user`, `canary.exe`.
**Zero cross-session names in 3.5 days.** Peak concurrency observed
under this boot: 10–11 running requests (Nov 1 03–04h) — heavier
than the first two days but still below the 32-way cap.

**`canary.exe` resolved (not leakage):** proxy-log trace of the
14:25:50 window (req 1793454908754 et al.) shows the user's own
sessions were running the `canary` skill (timer binary
`canary.exe` via fixer subagents) — the word was in the request
context. The model emitted `canary.exe` as a tool NAME where it
should have emitted `bash` with command `canary.exe` —
in-session content confusion, same class as `Read`/`summary`.
(Initial epoch-window math was off by 31 days and scanned an empty
range; corrected window 1793454900000–1793456760000 found 351
req files.)

**Still pending:** the 6–8-parallel-heavy-agent load case (user
had not run it as of 2026-10-30).

## 9. Upstream known-issue investigation (2026-11-01) — ANSWER: YES, known

Question: "is HICACHE=1 → hallucinations/cross-session exchange a
known issue with a fix?" Verified against upstream PRs and the
RUNNING tree (not the .slim clone — the running package is the
editable install at `sglang-official/python/sglang`, HEAD
`a34f194c1a` + local sm120 patches; the clone turned out to be a
different branch — its maintainer's `no_buffer`/`last_mamba_host_pool`
claims do NOT apply to our tree and were discarded).

### 9.1 The exact race is fixed upstream in two PRs

- **PR #36738 "[HiCache] Fence load-back behind the forward
  stream"** — MERGED into main 2026-08-28 (merge commit
  `f611e0c0`, 18 lines, 3 files).
  Mechanism (their words): "Overlap scheduling frees a finished or
  retracted request's KV pages while the round already launched for
  it is still writing them... The hicache load-back H2D has no such
  ordering: it writes the reallocated pages from the transfer
  stream. In-flight writes from the earlier round can therefore
  race the restore and **corrupt a restored prefix**."
  → restored-prefix corruption = exactly our symptom.
- **PR #36743 "[HiCache] Wait for load-back before deferred Mamba
  COW"** — OPEN (not merged) since 2026-08-27.
  Mechanism: "A restored radix slot can be selected as the source
  of deferred Mamba copy-on-write before that layerwise transfer
  has finished. Deferred COW consumes the entire source slot at
  once... so it can **copy a partially restored state** into the
  request-owned destination."
  → a request inheriting a partially-restored mamba state is
  conditioned on another session's context = the cross-session
  tool/path leakage we measured.
- **PR #37613 "reserved overflow ring for mamba host pool
  write-backup"** — OPEN. Related family: when the LRU mamba host
  pool (our ~3.6% slice) is full, write-backup rolls back the
  whole batch INCLUDING the KV side → orphaned KV companion
  states. Matches the structural slot-starvation of §3.2.
- (#33714 is NOT a corruption bug — a feature request: chunked
  long prompts never get host-backed-up. Ruled out.)

### 9.2 Our running tree has NEITHER fence (verified on the box)

`grep` over the running package
(`sglang-official/python/sglang/srt/`, editable install):
- `wait_for_hicache_load_complete` (#36743): **absent**
- `load_fence_stream` (#36738): **absent**
- the vulnerable deferred-COW path IS present
  (`model_runner.py:1625` `_maybe_execute_deferred_mamba_cow_and_clear`)

So the `qwen4-main-squashed` branch predates/diverges from both
fixes. The H1 race is live in our build — the A/B result
(HICACHE=0 ⇒ leakage class gone) is the behavioral proof.

### 9.3 Fix options (ranked)

1. **Keep HICACHE=0** (current, validated) — zero-risk, costs
   cold re-prefill after restarts.
2. **Cherry-pick #36738 onto the fork** (18 lines: adds
   `HiCacheController.load_fence_stream` + scheduler wiring) —
   small, merged upstream, low conflict risk.
3. **Apply the #36743 patch** (PR branch
   `jpezzulli/sglang:john/upstream-mamba-cow-load-order`) —
   unmerged upstream; needs the patch + its 2 regression tests.
4. Re-run the A/B with both fences + HICACHE=1: if the leakage
   class stays at zero under heavy load, HiCache can be re-enabled
   for the agent profile (restores the host-tier warm-start
   benefit).

## 10. FENCES APPLIED (2026-11-01) — options 2+3 done, A/B pending

Both upstream fixes are now written as fork patches and applied to the
running `sglang-official` tree (uncommitted, matching the existing
`git apply` series convention):

- `patches/0012-hicache-load-fence.patch` — PR #36738, adapted.
  `load_fence_stream` attribute + `start_loading` fence
  (`l2_transfer_engine.host_to_device_stream.wait_stream(...)`) +
  scheduler wiring + test mock. The fork prerequisite holds: the
  fork's `L2TransferEngine` already runs H2D copies on its own
  `host_to_device_stream` (`mem_cache/l2_transfer.py:55`), so the
  fence orders the right stream. `HybridCacheController` inherits
  `start_loading` (verified: its `__init__` calls `super().__init__`
  and never overrides `start_loading`), so one fence covers both.
- `patches/0013-mamba-cow-load-fence.patch` — PR #36743, adapted.
  `wait_for_hicache_load_complete()` on the pool + the call before the
  deferred COW in `_maybe_execute_deferred_mamba_cow_and_clear` +
  the 2 upstream regression tests. Fork note: `HybridReqToTokenPool`
  lives in `mem_cache/memory_pool.py` here (same as upstream), so the
  hunks applied unchanged. `LayerDoneCounter.wait_until` no-ops when
  `consumer_index < 0`, so the fence is safe when no load is active.

Verification (all on testcomp2, 2026-11-01):
- `git apply --check` clean for both; applied; `git status` grew 14 → 16
  files (scheduler.py + cache_controller.py + 2 test files new;
  memory_pool.py / model_runner.py were already dirty, hunks did not
  conflict).
- `py_compile` OK on all four modules; runtime import checks confirm
  the fence is in `start_loading`, the pool method exists, the
  scheduler wiring is present, and the COW call is in place.
- 0012's test file: `Ran 16 tests ... OK (skipped=1)`.
- 0013's 2 new tests: both `ok` (run individually).
- The file's other 9 tests ERROR with `torch.OutOfMemoryError` —
  pre-existing: the live server holds all 8 GPUs; those tests
  allocate real mamba pools. Not caused by the patches.

`#37613` (overflow ring) deliberately NOT applied: it targets the
storage-tier (L3) companion path, not the H2D restore race, and its
files (`pool_host/mamba.py`) don't exist in the fork's layout.

`scripts/apply_patches.sh` and `patches/README.md` updated with 0012/0013
(0012/0013 are no-ops at `HICACHE=0`, so the current daily boot is
unaffected).

**Next step (not yet done): the A/B — boot `HICACHE=1` with the fences
and run the per-hour measurement. Note the untested 6–8-parallel-heavy-
agent load case still applies.**

## 11. Heavy-load case PASSED (2026-11-01 14:03 UTC) — standing question retired

User confirmed: the 6–8-parallel-heavy-agent load HAS been run under the
HICACHE=0 boot, and NO phantom-folder permission requests reappeared.
Log corroboration (`hall_rate5.sh` + signature sweep, whole boot):

- Load evidence: Nov 01 hours 03–04 hit 1864 / 2025 reqs/h with
  1591/1660 big-reqs; `#running-req` saturation observed at 28–32
  (the `--max-running-requests 32` ceiling) — this is the intended
  heavy parallel-agent profile.
- The Nov 01 03:00–05:59 window contains ZERO unknown-tool lines
  (the `halluc=2/11/0` counts in those hours are the bad-param class).
- Whole boot: `c4b7c840` = 0, `visual_studio_solution_info` = 0,
  hex-session-id tool names = 0, permission-style patterns = 0.
- Totals: 51,816 reqs, 29 unknown-tool (all generic confabulations:
  exec/summary/Read/task_update/search/print/canary.exe/ask_user —
  all traced to in-session content), 519 bad-param (model-internal
  schema confusion, present on both boots). Overall 1.06%.

**Verdict:** the HICACHE=0 profile is validated under the full load
envelope, including the worst-case parallel-agent load. The
"ask about the 6–8-parallel load" reminder (AGENTS.md) is retired.
Remaining open item: the HICACHE=1 + fences A/B, deferred by the user
(no deliberate restart of the clean daily boot).

## 12. Proxy-log forensics: the `kirill` incident proven + bench (2026-11-01)

The user supplied the smoking-gun pair (copied to `logs-kirill/`):

- **`1793292095440`** (Oct 29 16:41 UTC, HICACHE=1 boot): the request
  contains **zero** `kirill` mentions (verified in the full 114 KB
  request); the response's reasoning contains **6** — foreign paths
  `C:\Users\kirill\Work\Orca\...` and a `claude` temp path — and the
  model then emitted a `bash` tool call (`cd /c/Users/kirill/...`)
  from that contaminated reasoning. The user's tree is
  `D:\shared\...`; `kirill` is another user's home. This is the
  cross-session KV-cache leak, proven at the proxy layer: the content
  was in NO input of that request.
- **`1793461528682` + 253 more reqs** (Oct 31 15:45–17:26 UTC,
  HICACHE=0 boot): one session with growing context (first/late
  requests share a 3000+ char prefix) — the SAME session continuing
  after the incident, carrying the leaked tool call in its own
  history. The Oct 31 responses mention kirill only self-referentially
  ("cd /c/Users/kirill/... — that wasn't even my working directory to
  begin with, the user rejected it"). Not a new leak; history replay.

Incident shape for the bench: one session, ~100 KB+ context, ~250
requests over ~100 min of heavy parallel load — the leak burst window.

**Bench:** `scripts/bench/agent_hallucination_bench.mjs` (commit
`0f758e2`) replays this shape: 8 lanes × 2 chains (main+subagent),
~30 KB shared prefix (radix-shared + host-tier offloaded), session
churn (forces prefix reload mid-flight), 100K+ token contexts via
synthetic tool results, unique per-session canaries (8-hex +
`/work/<hex>/`) -> `CROSS_SESSION_LEAK` on any cross-lane appearance,
`UNKNOWN_TOOL` on any undeclared tool name. Smoke-tested 2 lanes ×
2 min on the live HICACHE=0 boot: 33 reqs, 1.19 M in-tokens, 0 leaks,
0 unknown tools (4 errors = end-of-run deadline aborts).
Recommended A/B profile: 8 lanes × 60 min (old leak rate ~0.24/h —
15 min is only a smoke test).

## 13. The A/B result: fences did NOT prevent recurrence (2026-11-01)

The HICACHE=1 + fences boot ran the 60-min bench (19:17:22–20:17:21
UTC, 8 lanes, 2536 reqs, 165.5 M input tokens). The bench's own
detector reported **0 cross-session leaks, 0 unknown tools** — but
that is a false negative: it only detects *verbatim copies of its own
canaries*. The user's own agentic sessions (through the same proxy,
same saturated server) hallucinated during the same window:

**Timeline (UTC, all verified against the proxy archive — ground
truth):**

- 19:46:30 — user: "OK, the benchmark is working. Can you also check
  whether caching works?" (33 min into the bench). The next two
  assistant turns are normal (curls `/metrics` on 192.168.2.167).
- **19:48:09** — first contaminated response (req `1793562489544`):
  reasoning is coherent ("Let me curl /metrics") but the emitted
  `bash` tool call targets `http://192.168.10.100:8000/metrics`.
  The 261,724-byte request contains **0** occurrences of the IP.
- 19:49:06 — response adds `192.168.10.128`, `/data/launchpad/`,
  `bench_random.py` — none in the request.
- 19:50:54 — response adds `C:\Users\user\browser-laptop` — not in
  the request. User spots it ("what is that browser-laptop folder?").
- 19:49–19:52 — the user's other sessions emit undeclared tools:
  `exec`, `mcp__local__exec`, `mcp__local__filesystem_read_file`,
  `wsh` (4 events, same window).
- ~20:18 — user restarted the server to the HICACHE=0 daily profile.

**The decisive forensic finding:** a full scan of ALL 185,759 proxy
archive files (Oct 27–Nov 1, every request and response) shows the
first occurrence of `192.168.10.100` in the entire archive is the
victim's 19:48:09 *response* itself. `192.168.10.128`,
`/data/launchpad/bench_random.py`, `browser-laptop` likewise never
appear in any request anywhere. Unlike the `kirill` case (a verbatim
copy of another session's content), this content was **born in the
response** — it is not another session's text. It is the same
incoherent-KV-state class as the §7 root cause (mamba-state
incoherence): the corrupted state makes the model emit plausible
confabulations (a 192.168.10.x LAN IP, `:8000` the default vLLM
port, `/data/launchpad/` a plausible bench path) that are coherent
in intent but false in content.

**Server-side confirmation (watcher):** the watcher (baseline 0 on
the fresh boot) printed its first `!!! SERVER-SIDE UNKNOWN-TOOL` at
**19:50:03** (the server-side twin of the proxy's 19:49:29/19:49:43
`exec`/`mcp__local__exec` calls) and kept alerting through the bench
window — **25 `not defined in the tools list` hits by 20:15:00**,
minutes before the restart. Compare: the entire 3.5-day HICACHE=0
boot accumulated 29 such lines. The HICACHE=1 + fences boot produced
a full day's worth of the old boot's server-side leak signal in
under one hour.

**What this means:**

1. The 0012/0013 fences (verified live in the running tree) did NOT
   prevent the hallucination class under saturated load. The A/B is
   **negative**: the fences are not sufficient for daily HICACHE=1
   use. HICACHE=0 (`serve_best_no_kv_offload.sh`) remains the only
   safe profile.
2. The bench's canary detector is necessary but not sufficient — it
   cannot see the incoherence class. A stronger detector for the
   bench: the synthetic world is fully controlled, so any absolute
   path, IP, or folder name in a lane's output that is not
   `/work/<hex>/...` or a declared tool is a NOVEL-CONTENT hit.
   (Verified: the bench's own outputs contained none of the
   session's leaked content, and the bench's canaries never appeared
   in the user's session — no verbatim copy in either direction.)
3. The contamination window (19:48–19:52) coincides with the bench's
   saturated load — consistent with the race being load-dependent.
4. The user's session self-propagated the contamination after the
   first hit (contaminated turns entered its own history), so the
   session never recovered — same shape as the Oct 29 `kirill`
   incident's aftermath.

**Remaining:** the fences' upstream PRs (#36738/#36743) may be
necessary-but-not-sufficient, or the fork's `extra_buffer`
mamba-radix-cache strategy may have an additional race the fences
don't cover. The evidence package (this §13 + the proxy req/resp
pairs above) is ready to attach to an upstream report.

## 14. ROOT CAUSE FOUND + FIX: the QSA indexer state gap (2026-11-01)

The "additional race" suspected in §13 is now identified — and it is
not a race. It is a **missing state transfer**: the QSA sparse-
attention indexer's compressed keys are never backed up to the host
tier at all.

**The gap (verified in the live tree, `a34f194c1a`):** the model's
compressed index keys live in `QSATokenToKVPool.qsa_compressed_k_buffer_pool`
— a separate GPU buffer **outside** `full_kv_pool`. The HiCache host
stack for this model is assembled by `build_hybrid_mamba_stack`
(`hybrid_pool_assembler.py`) as `pools="KV + MAMBA"` — there is no
INDEXER entry (the `DSAIndexerPoolHost` sidecar exists but is wired
only for the DSA/MiniMax draft-pool paths, :1037/:1494/:1932). The
fork's `IndexKeyCache` (the Qwen3-Next-DSA variant) has
`cpu_copy`/`load_cpu_copy` hooks, but the QSA compressed pool has no
host mirror at all.

**Why this produces exactly the §13 class:** when a prefix is
evicted to RAM and later restored, the KV and mamba state come back
correctly (0012/0013 fence those), but the indexer's compressed keys
are whatever the GPU allocator handed out next — **another request's
keys**. The QSA indexer then selects the wrong attention blocks for
the restored prefix: the model attends to mismatched content and
confabulates coherent-in-intent, false-in-content specifics (a
192.168.10.x LAN IP, `:8000`, `/data/launchpad/`, `browser-laptop`).
That is why the content was "born in the response" (§13) — it is not
a copy of another session's text; it is the model's own generation
over incoherent attention. And it is why the fences (0012/0013) were
necessary but NOT sufficient: they fence the mamba state, which was
already being restored correctly.

**The fix — upstream PR #39893** (open, 2026-09-17, alphabetc1,
"fix: preserve QSA indexer state through HiCache", tested on the
exact model on B200; its removal test showed 93.8% mismatched
elements without the consumer fence). Ported to the fork's API as
**patch 0014** (`patches/0014-hicache-qsa-indexer-state.patch`,
`git apply`, uncommitted — the live tree has it applied and the
changed files compile + import in the deployment venv):

1. `mem_cache/pool_host/qsa.py` (new): `QSAIndexerPoolHost` —
   page-row host mirror of the compressed keys, packed (target
   layers + MTP draft) keyed by full-KV token slots; built on the
   fork's `DeepSeekV4PagedHostPool` (the fork's version lacks the
   PR's `page_aligned_only` kwarg — omitted; the QSA transfers are
   always whole-KV-page so the token-granular fallback is never
   taken).
2. `build_hybrid_mamba_stack`: optional `qsa_pool` arg → shrink the
   KV host slice proportionally, add the `PoolName.INDEXER` entry
   with its own (pre-extension) layer mapping.
3. `_MambaStrategy.build`: detect `QSATokenToKVPool`, pass it,
   register `SidecarPoolSpec(INDEXER, KV)` — the fork's
   `UnifiedRadixCache` sidecar machinery (indices ride the KV pool,
   all-or-nothing KV clamp) is generic and needed no changes.
   `pools_desc` becomes `KV + MAMBA + INDEXER`.
4. `get_qsa_compressed_k_buffer`: the consumer fence
   `_wait_for_layer(layer_id)` (the indexer must wait for the
   layer's H2D event before reading restored keys).

**Validation on the live box (2026-11-01, alongside the running
server, `scripts/test_qsa_indexer_hicache.py`):**

- D→H backup → poison → H→D restore to different pages → read
  through the production accessor: **bit-exact**.
- With the fence + a busy H2D stream: bit-exact.
- **Negative control (fence removed): race observed — stale data.**
  The test detects exactly the bug class, and the fence is what
  fixes it.
- `host.size == anchor.size` (the sidecar rides the KV slot space);
  size-per-token parity holds.

**Status:** the patch is applied to the live tree (inert on the
current HICACHE=0 boot — no host tier exists). The A/B gate is the
60-min bench with the NOVEL_CONTENT detector on HICACHE=1 + the
watcher; the boot must log `pools=KV + MAMBA + INDEXER`. Until that
passes, **HICACHE=0 remains the only safe profile**.

## 15. A/B VERDICT: PASS — patch 0014 fixes the class (2026-11-02)

The rerun of §13's exact A/B (same bench, same 8 lanes × 60 min, same
default 64 GiB/rank host tier — only variable: patch 0014 applied).

**Boot gate:** `pools=KV + MAMBA + INDEXER` on all 8 ranks; the
`qsa_indexer` host pool allocated (5.75 GB/rank × 13 layers). The
tier was genuinely exercised: device pool saturated (token usage
0.92–0.99 throughout), and metrics show
`sglang:cached_tokens_total{cache_source="host"} = 1.4484e+08` —
144.8M tokens restored from host during the run. A pass with an
unused tier would have been vacuous; this one is not.

**Bench result** (`bench-run.summary.json`, 2026-11-02 04:50 UTC):
2963 reqs, 192.5M in-tokens, 1.16M out, 0 errors,
**0 cross_session_leaks, 0 unknown_tool_calls, 74 novel_content_hits**.

| Signal | §13 (broken, no 0014) | §15 (with 0014) |
|---|---|---|
| Server-side unknown-tool lines | 25 by minute 58 | **0** |
| Bench canary cross-session leaks | (contaminated real users) | **0** |
| Bench unknown-tool calls | 0 | **0** |
| NOVEL_CONTENT hits | (detector too weak to fire) | 74 — all triaged false-positive, below |

**The 74 NOVEL_CONTENT hits are all false positives** — every one
forensically traced through the full proxy archive
(`B:\intersub\temp\llm-proxy\logs`, window = this boot only):

1. **OS-probe class (~38 lanes×hits, the bulk):** the models correctly
   detect that the bench's canned tool layer ignores arguments ("even a
   `read` of a nonexistent file and a `write` return the same canned
   notes.md payload") and probe it with canonical Linux paths —
   `/etc/hostname`, `/etc/os-release`, `/proc/self/...` — and
   deliberately-impossible paths (`/proc/1/nonexistent_xyz`,
   `/proc/self/root/definitely_missing_dir_8x2`). These are base-model
   knowledge: each string is **born in the probing lane's own
   reasoning** and never arrives in any request before being born
   there. The same string "born" in many lanes is the same prior
   firing many times, not a leak.
2. **Monitor lane (`60b816d1`):** its task text embeds the bench's own
   hit log, so quoting `/etc/hostname` / `parker` / `morgan` in its
   traffic is legitimate self-reference (verified: first occurrence is
   the HIT-log line injected by the bench itself).
3. **Two benign vanilla confabulations** (`/home/parker/projects/
   auth-service`, `/home/morgan/...`): generic invented usernames,
   born in the confabulating lane's own response, absent from every
   request in the archive, self-propagated only inside that lane's own
   history — no cross-lane arrival. These are ordinary model
   invention, not the §13 incident class (which was *real*
   user-specific facts: `192.168.10.100`, `/data/launchpad/`,
   `browser-laptop` — nothing a base prior could produce).

**Detector calibration:** the bench's NOVEL_CONTENT regex family
originally flagged any absolute path outside the synthetic world.
Commit `cbdf71c` whitelists generic single-segment OS paths
(`/etc`, `/proc`, `/sys`, `/tmp`, loopback IPs) while keeping the
incident class fully flagged: LAN IPs, drive paths, `/c/Users/...`,
`/data/`, `/home/<user>/`, `/mnt/`. Future runs should read clean.

**Verdict:** the QSA indexer state gap (§14) was the root cause;
patch 0014 closes it. **HICACHE=1 (`serve_best_kv_offload.sh`) is
cleared for the daily profile.** Caveats: (a) the A/B is one 60-min
synthetic run — the real-usage watcher (`scripts/bench/
watch_server_leaks.ps1`) stays armed for a few days; (b) patch 0014
is still uncommitted in the engine tree (kit convention) and the
upstream PR #39893 is still open — if it merges with different
mechanics, rebase then; (c) the 2026-11-01 failing boot predates
0014, so no counter-evidence exists against it.

# Qwen3.8-Flash-Next NVFP4 — local TP1 deployment (RTX PRO 6000 Blackwell, 96 GB)

**Status (2026-08-31, after the 8h optimization run): DONE — serving, heavily optimized, validated.**
Official sglang `qwen4-main-squashed` branch + local commits on `sm120-wy` (see git log in
`../sglang-official`). No Docker. **Beats jpezzulli/sglang-rtxpro6000's published figures by ~35-45%.**

> **Deployment target moved to `testcomp2`: 8× RTX 5090 (sm120, 32 GB/card), TP8.** The TP1
> figures in this doc are provenance from the 96 GB card. 5090/TP8 ops notes:

## Deployment on testcomp2 (8× RTX 5090, TP8)

- Paths: this repo `/mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang`,
  sglang checkout + `.venv` at `sglang-official/` (branch `qwen4-main-squashed`),
  model `/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4`, caches `cache/` in the repo root.
- sglang is built (editable install) but ships **without the patches** — after every
  `git pull` run once: `bash scripts/apply_patches.sh` (idempotent; no rebuild needed).
- **TP8 requires MoE expert parallelism** (`EP_SIZE=8` → `--ep-size 8`, default in both
  launchers): `moe_intermediate_size 640` under pure TP8 = 80/rank → NVFP4 scale swizzle wants
  padding (w2 K′=5 groups, not ×4) → "padding … gated activations" assert at load
  (`modelopt_quant.py`). EP8 keeps the full 640/expert (64 experts/rank) and the shared expert
  fuses into the EP MoE layer → full load passes. Correct alignment rule: per-rank
  intermediate must be ×64 → pure-TP valid only for TP 1/2/5 (TP4 = 160/rank is INVALID —
  earlier docs said otherwise; hit again 2026-09-10, K′=10). On 8×32 GB no pure-TP fits;
  fallback candidate: `TP=4 EP_SIZE=4` (4 cards only, untested).
- **FR-Spec at TP>1 needs patch 0007**: `eagle_worker_v2.init_lm_head` gathered the draft
  lm_head (`head.data[self.hot_token_id]`) with global ids against the vocab-parallel slice
  (31040 rows/rank at TP8) → device-side `vectorized_gather_kernel` OOB assert during draft
  init (crashed 2026-09-10 22:59 after a fully clean TP8+EP8 model load + KV pool alloc of
  2,060,224 tokens). 0007 all-gathers the full head along the vocab dim first.
- **`ninja` on PATH**: sglang's JIT kernel builder shells out to `ninja`; without the venv's
  bin dir on PATH, CUDA-graph warmup dies with `FileNotFoundError: 'ninja'` (hit 2026-09-10
  23:16). serve.sh now prepends `$REPO/.venv/bin`.
- **Graph-capture headroom on 32 GB cards**: MEMFRAC 0.90 preallocates the KV pool so tight
  that decode-graph capture OOM'd on every card (23:50 run: <100 MiB free, 128 MiB short).
  Defaults now 0.85 (best) / 0.88 (single); if capture still OOMs, drop another notch.
  KV pool cost: ~2.06M → ~1.8M tokens aggregate — irrelevant at 8-way/262144 ctx.
- **Host RAM**: sglang does NOT need ~1 TB. Committed RAM is tens of GB (JIT compile capped
  by MAX_JOBS=4 / FLASHINFER_NINJA_JOBS=4, PLE pinned ~1.3 GB, schedulers/tokenizers small);
  the large "used" figure is Linux page cache over the 206 shard files — reclaimable, only
  speeds up reloads. A 256 GB cap is fine; only cold/repeat weight loads get slower once
  the shard cache is evicted.
- **WORKING (2026-09-11 00:09)**: server up on :1025 with MEMFRAC 0.80; user validated
  245,109-token prompts at 1/2/4/8 parallel; both aliases serve (requests address the model
  as `glm-5.3-flash` or `pennyroyal`). Late Triton pool-kernel loads (alloc_extend,
  assign_req_to_token_pool) are benign — diagnostic warnings from `triton_load_watch`;
  serve_best.sh now fires a post-ready warmup request so they load inside the startup window.
- **Image requests OOM'd on GPU 0** (00:14 run, MEMFRAC 0.85): the transformers *fast* image
  processor runs torch ops on GPU 0 inside the tokenizer process — on a card with <100 MiB
  free it dies. Fix: `--image-processor-backend pil` (CPU-only preprocessing; default in
  serve.sh). With pil, MEMFRAC 0.85 likely works again.
- **Tool-call requests 400**: `"required": {}` (malformed tool schema from some agent
  frameworks) fails jsonschema validation → patch 0009 drops malformed `required` instead.
- **Server-killing device assert (2026-09-11 00:56, under investigation)**: one long agent
  request (~86k tokens, temp 0.6) aborted all 8 ranks ~25 decode tokens in with a
  single-element `IndexKernel` OOB gather (async-reported at `copy_done.synchronize()`).
  Ruled out: OOM, KV pressure, tool schemas, padded-vocab sampling, draft gathers.
  Top suspect: mamba `extra_buffer` state-restore chains from a deep radix tree.
  Repro: `scripts/run_crash_reproduction.sh` replays the last 13 real turns before the
  crash (bounded `.repro.json` copies, max_tokens 1500) + the pristine crash request,
  ×3 attempts, with `DBG_LAUNCH_BLOCKING=1` (exact kernel on next assert) and
  `DBG_CRASH_DUMP=1` (CUDA coredumps into `logs/crashdump/`). A 2-request replay ×3
  attempts was clean (~2k decode tokens) → content alone doesn't trigger it;
  the 13-turn chain recreates the live session's accumulated tree history.
- **CRASH REPRODUCED ×3 (16:53 attempt 2 / 17:31 attempt 7 / 17:44 attempt 2-seq 5, 13-turn chain)**:
  all inside the **EAGLE draft CUDA-graph replay** (`eagle_draft_cuda_graph_runner.py:279` →
  `eagle_worker_v2.py:625 draft`), not the target model / mamba restore. Chain seq 4
  (`1789088114760`, 80,128-cached prefill) crashed twice; the probe run moved to seq 5 →
  tree-state dependent, not request content. Concurrency ruled out (opencode request never
  reached the server during run 2). sglang's built-in in-graph probes (per-step `topk_index`
  bounds + NaN/Inf on draft logits) were ACTIVE and SILENT in run 3 → the per-step topk chain
  is clean. In-graph suspects narrowed to torch index/gather ops in the draft model forward
  whose FIRST element can be OOB: PLE short-conv `conv_state[state_indices/track_indices]`
  gathers+scatters (`qwen4_exp.py _short_conv`), `NGramPool.set_context` slot ids, the
  ping-pong track-slot source (`schedule_batch.set_mamba_track_indices_from_reqs` —
  `req_index_to_mamba_ping_pong_track_buffer_mapping` values; lazy mode stores **-1** for
  unallocated slots; track slots are what `cache_{un,}finished_req` hand the radix tree —
  matching the tree-state dependence; track-interval 64 → seq 4 crosses its first boundary
  ~64 decode steps after the big cache-hit restore), eagle draft **initial** `hot_token_id[topk_index]`
  (the only hot gather without a probe), and the PLE offloaded-embedding prefetch gather.
- **Patch 0010 (`patches/0010-sm120-track-probes.patch`) applied on testcomp2**: async
  `maybe_detect_oob` probes (no-ops unless `SGLANG_ENABLE_ASYNC_ASSERT=1`) at: eagle draft
  initial topk_index vs hot map; the ping-pong mapping table + gathered track slots
  (schedule_batch); `NGramPool.set_context` ids; `_short_conv` state_indices + decode
  track_indices. Probes are graph-capturable (`torch._assert_async`) and fire DURING replay
  with a named message. Live tree compiles; note: patch file's qwen4_exp.py hunk context
  assumes the working-tree state (0001b/0004/0005 patch files are stale vs HEAD — pre-existing
  drift, untouched).
- **Run 5 (18:19–18:20, with patch 0010b dump) — probe fired but it was OUR OWN FALSE ALARM**:
  `[track-dump] GEOM mamba_size=48 ngram_size=48 req_pool_rows=(9,2)` + first-ever
  `BAD rows (i,row)=[(0, [48, 47])] rpis=[8] next_idx=[1] bufs=[[48, 47]]`, then our E2 assert
  `index >= 48 ... mapping values` killed all ranks mid-decode of seq 3 (resp truncated to
  1,767 bytes). GROUND TRUTH established: MambaPool (`max_slots = size + 1`),
  ShortConvPool (`conv_state = (layers, size+1, ...)`), NGramPool (`context = (size+1, ...)`)
  ALL allocate **size+1 = 49 rows with row 0 = shared dummy**, and `MambaSlotAllocator.clear()`
  hands out ids 1..48 (`arange(1, size+1)`) — so **id 48 is legitimate** and probe bounds must
  be `size + 1`. Launch flags confirmed from crash pkl: no unified memory, `--max-mamba-cache-size 48`,
  `--max-running-requests 8` (req pool 9 rows), NEXTN topk=1 steps=3 draft=4,
  `--mamba-track-interval 64`, `--mamba-radix-cache-strategy extra_buffer`, page 64.
  The ORIGINAL IndexKernel assert message from runs 1–3 is unrecoverable (server logs overwritten;
  crash pkls hold only server_args/config/requests/launch_command).
  **Patch 0010c applied (compiles)**: E2 (mapping values + gathered track slots) and E3
  (NGramPool.set_context) bounds corrected to `size + 1`; dump gate now flags `<0 or >size`;
  BAD dump enriched with per-req `mamba_pool_idx` (working slot) + `rid`; NEW probe in
  `get_mamba_indices` (working mamba slots vs `size+1`, covers the GDN state path in-graph).
  E4/E1 bounds were already correct (`conv_state.shape[0]` / `hot_token_id.shape[0]`).
  NEXT: rerun → probes stay silent on legit 48s; either the original IndexKernel assert
  surfaces with its bound value (names the real tensor) or a corrected probe fires.
- **Run 6 (with 0010c) — ROOT CAUSE NAMED**: crash at attempt 5, seq 3; E1 fired on ALL ranks:
  `index >= 65536 (out of range): eagle draft initial topk_index vs hot map (spec_info origin)`.
  The draft's INITIAL `spec_info.topk_index` (carried from the previous round) contains a
  **full-vocab token id (≥ hot size 65536, < vocab ~152k)** — the stock probes were blind
  because they bound by `vocab_size` while the consumer gathers `hot_token_id[topk_index]`
  (hot map 64k from `--speculative-token-map`). Draft head IS gathered to 65536 rows
  (patch 0007: `head.data = head.data[hot_token_id]` after TP all-gather) → builds A/B
  (`_draft_extend_for_prefill` / `_draft_extend_for_decode`, topk over 65536-dim logits)
  mathematically cannot emit ≥ 65536. Draft graph runner zero + copies spec_info into
  static buffer (no staleness). Remaining writers: eagle_info filter/merge (subsets/cats)
  — so the leak is either a yet-unread path (e.g. verify-output/bonus-token flow, overlap
  future spec_info) or one of these under an unexpected state. Patch 0010d applied
  (compiles): gated `[topk-dump]` min/max dumps at consume, build=prefill, build=decode-extend,
  filter, merge=takeover, merge=cat → next run names the exact writer. topk=1 explains the
  single-element IndexKernel in runs 1–3 (the hot gather at eagle_worker_v2.py:674).
- **Crash-cycle automation (2026-09-11 19:11) — works end to end**: `scripts/crash_cycle.bat`
  (local) → `ssh testcomp2` → `scripts/crash_cycle_remote.sh` (deployed to the machine repo's
  `scripts/`): stops/kills every container of the vllm image (`ancestor=` filter only —
  grafana/prometheus/open-webui untouched; docker needs no sudo), runs
  `run_crash_reproduction.sh`, saves its exit code to `logs/last_repro_exit_code.txt`,
  restarts `/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4.sh` under nohup (that script is a
  foreground `docker run --rm -p 1025:1025`, log at `logs/vllm_start_<stamp>.log`), polls
  `:1025/v1/completions` with a "Hi" prompt (model auto-detected from /v1/models) until 200 OK
  (90 min timeout → exit 3), exits → ssh disconnects → the .bat notifies the opencode session
  (`ses_f72b307f3ffeQZQnrN91yNllIw`). Verified: syntax + image filter (1 match) + one full live
  cycle incl. vllm restart to 200 OK.
- **Runs 7/8 (19:19, ×2 cycle runs) — startup death, OUR instrumentation bug**: server died
  during draft graph capture: `[topk-dump] consume ...` at eagle_worker_v2.py:674 calls
  `.item()` → `cudaErrorStreamCaptureUnsupported` → `capture_end` fails → all ranks abort at
  attempt 1 (repro exit code 1 = "died during startup"). The consume dump is the only dump
  inside the captured `draft_forward` region; all other dump sites (build A/B after
  `execute()`, filter/merge, track-dump) run outside capture and are safe; probes
  (`maybe_detect_oob` = assert_async kernel) are capture-safe by construction (proven: run 6
  captured fine with E1 present). `get_mamba_indices` probe audited — kernel only, safe.
  **Patch 0010e applied + `patches/0010e-guard-topk-dump-capture.patch`**: consume dump now
  skipped when `get_is_capture_mode()`. Note for next crash: decode runs as graph REPLAY —
  python in `draft_forward` (incl. the consume dump) executes only at capture (zeros) or
  eager decode, NOT at replay; provenance for a replay-path E1 fire therefore comes from
  build=prefill / build=decode-extend / filter / merge dumps framing the writer, or, if all
  are sane while E1 fires, the corruption is inside the draft graph runner's buffer copy
  (eagle_draft_cuda_graph_runner.py:562/570).
- **Run 9 (19:46, exit 2, attempt 10 seq 2) — the ORIGINAL IndexKernel assert is back, all
  probes SILENT**: crash inside `eagle_draft_cuda_graph_runner.py:659 _replay_graph` (draft
  decode graph replay; faulthandler stack: :659 → eagle_worker_v2.py:626 draft → :1313
  forward_batch_generation → run_batch → event_loop_overlap). Stock `IndexKernel.cu:111`
  "index out of bounds" (single-element, runs 1–3 signature), NOT a named probe — so every
  probed input was valid at its probe point: all 230,920 `[topk-dump]` values in range
  (0 negative, 0 >65535 across 10 attempts), `set_kv_buffer (MHA)` stock probe
  (memory_pool.py:2469, bound size+page_size) silent, E1–E5 silent. The crashing op is an
  UNPROBED torch advanced-index op inside the captured draft graph. Pool state healthy at
  crash (token usage 0.05, mamba 0.08, cuda graph True, bs=1, ~150 tok/s, request ~80.3k
  tokens = the 289k-char chain request; NOT KV pressure). Coredumps (5× ~541 MB) rejected by
  cuda-gdb 13.1 ("file format not recognized") — route parked for good.
  Candidate unprobed ops in the captured draft path: embed gather (`embed_tokens(input_ids)`),
  per-step `out_cache_loc`/`positions` consumers, page-table two-level gathers (the triton
  `generate_draft_decode_kv_indices` is Triton, not IndexKernel — not the crasher).
  **Patch 0010f applied (`patches/0010f-eager-draft-debug-override.patch`) + wrapper knob
  `EAGER_DRAFT=1`**: env `SGLANG_DEBUG_EAGER_DRAFT=1` skips the draft decode graph and runs
  `draft_forward` eagerly — with DBG_LAUNCH_BLOCKING the identical assert then surfaces at
  the exact python op with a full stack. Next run: `EAGER_DRAFT=1` cycle run → if the crash
  reproduces eagerly, the stack names the op; if 30 attempts stay clean, the bug is
  replay-metadata staleness (capture-specific) — a different instrumentation pass.
- **Run 10 (20:24, exit 2, attempt 7 seq 5) — crash REPRODUCED eagerly (169k consume dumps
  confirm eager mode)**; last dumped topk valid (15377 < 65536). BUT the abort was the CUDA
  runtime's `abort()` on the device-assert trap (device asserts kill the process before any
  python exception can print — faulthandler "Fatal Python error: Aborted"), main thread at
  eagle_worker_v2.py:819 = the per-step `hot_token_id[topk_index]` gather launch. Since
  CUDA_LAUNCH_BLOCKING was ON ([dbg] echo confirmed) and the gather's index was just
  probed valid, the trap likely fired on the **overlap scheduler's concurrent stream**
  (the target-verify path — completely unprobed) while the draft thread sat at its sync.
  Verified clean: hot map values 0..248,076 < vocab 248,320 (embed-safe; real vocab is
  248,320, not 152k), draft embed = target's full embedding. Eager reproduction also
  exonerates capture/replay staleness — the bug is in the SHARED draft/verify path.
  **0010g applied (wrapper + repro_crash.sh, no server patch)**: `NO_OVERLAP=1` →
  `--disable-overlap-schedule`, so with CUDA_LAUNCH_BLOCKING the abort frame IS the
  trapping launch. Cycle driver now defaults `EAGER_DRAFT=1 NO_OVERLAP=1`. Next run's
  faulthandler main-thread frame names the op directly.
- **Run 11 (post-0010g, exit 2, attempt 13, seq 1) — IDENTICAL frame with overlap
  disabled** (`event_loop_normal` confirmed; `--disable-overlap-schedule` in the launch
  line): eagle_worker_v2.py:819 in draft_forward → :640 draft → :1315. Overlap exonerated
  too. Conclusion: `CUDA_LAUNCH_BLOCKING` cannot pin the trapper — **Triton and other
  driver-API launches (cudaGraphLaunch, cuLaunchKernel) bypass CUDA launch blocking**, so
  an async kernel (triton fused rope/kv-store, topk1, MoE, or the target verify graph
  replay) can trap while the CPU thread sits at the next blocked ATen launch (:819's
  gather). Countermove **0010h applied** (helper + 4 sites in eagle_worker_v2.py, gated on
  `SGLANG_DEBUG_EAGER_DRAFT=1`): `_dbg_sync_checkpoint` = log tag BEFORE
  `torch.cuda.synchronize()` — sites: after model forward / after topk / after hot gather
  (per step, :793/:829/:839) and before verify input build (:658). The last
  "[dbg-sync] checkpoint" line without its "passed" line names the trapping phase; a clean
  sync raises a full python traceback instead. Patch mirrored: `patches/0010h-...`.
  Next run's log: read the tail of [dbg-sync] lines + the first missing "passed".
- **Run 12 (21:31, exit 1) — startup death from 0010h's own sync during capture**: the
  draft graph capture at init runs draft_forward (eager via 0010f), the checkpoint's
  `torch.cuda.synchronize()` inside an active capture → "operation not permitted when
  stream is capturing" → "Capture cuda graph failed" → all ranks abort. Same class as
  runs 7/8. **0010i applied**: `_dbg_sync_checkpoint` now also returns early when
  `get_is_capture_mode()` (same predicate 0010e proved). Serving-phase checkpoints
  unaffected. Patch mirrored: `patches/0010i-...`.
- **Run 13 (22:01, exit 2, attempt 12, seq 1) — THE OOB VALUE OBSERVED**: all sync
  checkpoints passed through the whole draft; the crash came right after the
  **decode-extend** dumps printed `build=decode-extend bs=1 min=max=196625` —
  **196,625 > 65,536 (hot map size)** but < 248,320 (vocab) — a full-vocab value sitting
  in the chain `topk_index` slot that the next draft round gathers `hot_token_id` by.
  E1 in run 6 was real after all — the writer is the **decode-extend chain fill**
  (`ret_topk_index = argmax(draft_logits_output.next_token_logits[select_index])`,
  eagle_worker_v2.py ~:1126-1160) — but argmax over the 0007-sliced 65536-wide head
  cannot return 196,625, so the decode-extend logits were full-vocab (a head leak) OR the
  value came via the DSA IndexShare seed (`index_share_for_mtp_iteration` /
  `dsa_topk_indices` — indexer top-k are full-vocab indices). **0010j applied**: the
  decode-extend dump now also prints `logits_w` (the logits width) and the dsa_seed
  min/max — the next run discriminates the two theories. Also this run had NO assert
  message and NO python exception in the log — abort swallowed them (flush ordering);
  the dump values are the reliable signal.
- **Run 14 (22:14, exit 2, attempt 2, seq 2 — fast repro) — `logits_w=248320`: the
  decode-extend logits are FULL-VOCAB.** The decode path's per-step OOB probe uses
  `logits.shape[-1]` as the bound (= 248,320 if the draft logits are full-vocab) →
  the probe never fires while `hot_token_id` (65,536 rows) traps — explains the silent
  probes in runs 9-13 AND E1 firing in run 6 (E1's bound was hardcoded 65,536). Root
  cause shape: the draft model's head restriction (0007 slice at init) does NOT reach
  the actual GEMM — the draft's topk = argmax over FULL-vocab logits = full-vocab token
  ids written into the chain, gathered by `hot_token_id` (65,536 rows) → trap whenever
  a token id ≥ 65,536 lands in the chain (small ids pass silently as garbage tokens —
  explains intermittency + long-session crashes). **0010k applied**: dumps the draft
  `lm_head.weight.shape` + hot-map size at decode-extend and the decode-path logits
   width per round — discriminates "replacement never applied" (weight 248,320) vs
   "replacement applied but GEMM bypasses it" (weight 65,536 + logits_w 248,320).
- **Run 15 (22:26-22:31, exit 2, attempt 3, seq 2) — ROOT CAUSE NAILED**: draft
  `lm_head.weight.shape=(65536, 2560)` (the 0007 slice IS applied) BUT `logits_w=248320`
  in BOTH the decode and decode-extend paths. Mechanism: after 0007, every rank holds the
  FULL 65,536-row sliced head, but the logits processor still all-gathers per-rank
  logits (vocab-parallel contract: each rank should hold 1/8 of the hot vocab) →
  gathered = 8 × 65,536 = 524,288 rows → trimmed to vocab_size 248,320 → the draft's
  argmax returns `[block_offset + hot_rank]` (196,625 = 196,608 + 17 — hot rank 17 from
  block 3!) → these are NOT hot ranks → `hot_token_id[topk_index]` traps whenever the
  value ≥ 65,536; block-0 values (19, 13) pass silently as garbage tokens — explains
  intermittency, long-session crashes, silent probes (their bound = the inflated logits
  width), and E1 firing (hardcoded 65,536 bound). Also the FIRST crash ever seen
  (pre-0007, `vectorized_gather_kernel` OOB) was upstream sglang's own TP>1 +
  speculative-token-map head-indexing bug — 0007 fixed that but broke the gather
  contract. **0011 applied (the fix)**: after slicing, re-shard the hot head — each
  rank keeps rows `[rank*8192 : (rank+1)*8192]` — so the all-gather reconstructs
  exactly the 65,536 hot ranks. Patch mirrored: `patches/0011-hot-head-reshard.patch`.
  Expected post-fix signature: `logits_w=65536`, decode-extend min/max < 65,536,
  no crash. Debug instrumentation (0010-0010k) stays in place to verify the fix.
- **Run 16 (22:43-23:51+, exit 0) — FIX VALIDATED**: all 30 attempts passed WITHOUT a
  crash. Post-fix signature exactly as predicted: per-rank head shape `(8192, 2560)`,
  `logits_w=65536` in both paths, decode-extend values valid hot ranks (198 < 65,536),
  no asserts/aborts. The crash is RESOLVED by **0011-hot-head-reshard.patch**.
  State: patches 0007 + 0011 must stay (0007 slices the hot head, 0011 re-shards it);
  0010-0010k debug instrumentation is env-gated and inert in daily serving (only the
  cycle's debug phase sets those env vars) — candidate for removal after a soak period.
  Caveat: 30 attempts is good but not exhaustive evidence; if a crash ever recurs,
  re-check the [topk-dump]/[dbg-sync] signature first.
- **Post-fix housekeeping (2026-09-12)**: debug patches 0010-0010k stripped from BOTH
  machines (sglang checkout restored per file: eagle_worker_v2.py = HEAD + 0007 + 0011,
  memory_pool.py/schedule_batch.py = surgical probe removal keeping 0001b,
  qwen4_exp.py/eagle_info.py restored — debug changes were working-tree only, never
  committed). `patches/0011-hot-head-reshard.patch` regenerated CLEAN (applies on
  HEAD+0007; verified in a throwaway worktree). `apply_patches.sh` now applies ten
  patches (0011 added after 0007). Debug patch files removed from both patches/ dirs
  (preserved in this repo's git history). Coredumps deleted (50.7 GB -> 95 MB of .pkl
  crash records kept). README.md + patches/README.md updated to the ten-patch series.
  New: `docs/SETUP.md` — fresh-Ubuntu-26.04 rebuild guide (driver/CUDA/docker, both
  model deployments, patch flow, crash cycle, validation checklist).
- `./scripts/serve_best.sh` (TP8+EP8, 8-way, fp8 KV + fp8 stack, ctx 262144) or
  `./scripts/serve_single.sh` (786K ctx). Both run in the **foreground** — Ctrl+C stops the
  server and a sweep reaps leftover SGLang GPU processes; logs also in `logs/serve.log`.
  Run under tmux/screen to survive disconnects; no systemd unit, no auto-start.
- Pre-flight: launchers abort while any GPU holds >4 GB (the box also runs a vLLM TP8 server
  occupying all eight cards — stop it first; PIDs are printed; `FORCE=1` overrides).
- Headless (display Disabled on all GPUs) → `CUDAGRAPH_MAXBS=8` is safe here.
- 1 TB host RAM; the old systemd `MemoryMax` cage is gone with foreground mode — `MAX_JOBS=4`
  still caps the cicc JIT storm, which is the real protection.
- 32 GB/card knobs to walk if OOM at load or first graph capture: `MEMFRAC` 0.90 → 0.88/0.85
  (best profile) or 0.92 → lower (single-session); `MAMBA_CACHE` (default 6×MAXREQ — keep the
  6× ratio or spec graphs silently cap concurrency); `CUDAGRAPH_MAXBS` ≥ MAXREQ.
- 5090 deltas vs the 6000 card: no NVLink (PCIe P2P only) → expect a TP8 comm tax; ~1/3 the
  per-card HBM; 128 CPU cores (JIT caps `MAX_JOBS=4` remain mandatory — unbounded cicc × 128
  cores is a RAM storm, even with 1 TB).

## Result (stable `serve_best.sh` build, warm)
| | jpezzulli | ours (temp 0.6) | ours (greedy = lossless) |
|---|---|---|---|
| Decode C1 | 171 tok/s | **231.0 median / 234.6 best** | 202.9 / 205.0 |
| Decode C4 aggregate | 428 | **620 / 628.7** | 549 / 560 |
| Decode C8 aggregate (opt. 8-way variant) | — | 758 | |
| Prefill (2.5K) | ~10-12K | ~10.4K tok/s (17.8K cached) | |
| TTFT | — | ~139 ms | |
| 16K-context decode | — | ~265 tok/s (no degradation) | |
| 20-min soak + 60-min burn-in | — | 2.38M toks total, 0 errors, VRAM/RAM flat | |
| MTP accept length | 2.58 | 2.5-3.0 (relaxed 0.3) | 2.1-2.2 |
Context: 262144 default — the full native window, same KV pool (~406K fp8 tokens) and same speed as 32K. 151K-token needles pass at start/middle/end depths.
76K-needle stress: edges + most middle depths retrieve; occasional middle-depth misses are
model/QSA-inherent (reproduced with bf16 KV — not our fp8).
Correctness gates (all PASS, final build): greedy arithmetic/fact · 3× needles in 7.3K prompt ·
cached-prefix identical · 5-8× GSM-style @0.6 · code spot · French. VRAM peak 95.5 GB.

## Run — two profiles (same unit `qwen-sglang`, same endpoint; one command to switch)

*(historical, RTX PRO 6000 box: systemd unit; the testcomp2/5090 deployment runs in the foreground — Ctrl+C to stop, see the deployment section above)*
```bash
./serve_best.sh      # DEFAULT: interactive + agents. 4-way, fp8 stack on,
                     # ctx 262144 (native), KV pool ~572K tokens, C1 ~231 tok/s.
./serve_single.sh    # ONE HUGE SESSION: ctx 786432 (YaRN x3), KV pool ~827K tokens,
                     # C1 ~185 tok/s (fp8 dense copies traded for KV head-room).
curl -s http://127.0.0.1:1025/health   # 200 when ready (~5 min)
systemctl --user stop qwen-sglang       # stop
```
The 786K profile is validated with needle retrieval at 653K-token depth (start/middle/end
all pass); 653K prefill ~89 s cold, ~4 s on cached prefixes. ~827K tokens is the physical
ceiling of the card (81.5 GB weights on 96 GB). An 8-way variant of the default profile
(`MAXREQ=8 CUDAGRAPH_MAXBS=8 MAMBA_CACHE=48`) measured 758 tok/s aggregate if ever needed.
Endpoint **http://localhost:1025/v1**, models **`pennyroyal`** or **`glm-5.3-flash`** (aliases,
patch 0008; first is canonical. OpenAI-compatible; thinking on by
default → tokens in `delta.reasoning_content`). Or from the laptop: `omega --update` then
`omega --serve qwen3.8-flash-next`.

## What made it fast (2026-08-31 run, in order applied)
1. **sm120 Triton low-M GEMM** (commit 086a37f): cuBLAS-under-capture served the decode dense
   GEMMs at 20-75% of DRAM bandwidth; a split-free (1,16,128) Triton kernel reaches ~90%.
   Env `SGLANG_ENABLE_SM120_LOWM_BF16_GEMM` (default on for sm120).
2. **W8A16 fp8 weight-only** (284d7fe): per-row-scaled fp8e4m3 copies of every dense weight
   ≥4 MB, served by a (1,32,128) kernel at 89% of the halved floor. `SGLANG_SM120_LOWM_FP8_WEIGHT=1`,
   costs ~3.6 GB VRAM → `MEMFRAC=0.95`.
3. **fp8 HC mix + fp8 lm_head** (85e7da1): HC persistent kernel 14.5→10.8 µs/mix; lm_head GEMV
   (and every MTP draft step's logits) byte-halved. `SGLANG_SM120_LM_HEAD_FP8=1`.
4. **Relaxed MTP acceptance** (config): `SPEC_ACCEPT_SINGLE/ACC=0.3` force-accepts draft tokens
   the target gives ≥30% prob. Lossy at temp>0 (sharpens sampling; exact at temp 0).
   Ladder (C1 @0.6): 1.0 lossless=179 · 0.5=203 · **0.3=231**. All quality gates pass at 0.3.
5. **FR-Spec 64K hot-token map** (config): draft lm_head scores a 64K subset of the 248K vocab
   (`hot_tokens_64k.pt` = 32K base BPE + code-corpus top tokens + specials). Verify stays exact.
6. **8-way concurrency** (config): `MAXREQ=8 CUDAGRAPH_MAXBS=8 MAMBA_CACHE=48` — the mamba cache
   must be ~6× MAXREQ or spec graphs silently cap at bs4 (C8 was *slower* than C4 before this).

## Knobs (env → serve.sh)
`SPEC_ACCEPT_SINGLE/ACC` (0.3; 1.0 = lossless) · `SPEC_TOKEN_MAP` (path or `none`) ·
`SGLANG_SM120_LOWM_FP8_WEIGHT` / `SGLANG_SM120_LM_HEAD_FP8` (fp8 off ⇒ pure-bf16 kernels) ·
`MAXREQ/CUDAGRAPH_MAXBS/MAMBA_CACHE` (4/4/24; mamba ~6x MAXREQ) · `MEMFRAC` (0.95) · `CTX` (262144 = full native window) ·
`KVDTYPE=fp8_e4m3` · `LINEAR_BACKEND=flashinfer` · `GDN_MTP_CACHE_MODE=none` (WY/RecoverSSM).

## Engine patches (branch `sm120-wy` @ ../sglang-official)
`0002` fp8-QSA dequant · `0003` sm120 fp32 prefill state · `0001b` RecoverSSM/WY port (+1316) ·
`086a37f` sm120 Triton low-M GEMM · `284d7fe` fp8 weight-only · `85e7da1` fp8 HC + lm_head.
Re-apply on a fresh checkout: see patches/README.md + cherry-pick the three commits.

## Hard-won gotchas
Desktop crash = full-range graph capture → keep CUDAGRAPH_MAXBS small. RAM thrash = uncapped
cicc JIT → MAX_JOBS=4 + systemd MemoryMax=112G. `pkill -f sglang` self-matches the tool shell —
kill by PID / `systemctl --user stop qwen-sglang`. First bench after restart is JIT-polluted —
always warm first. Profiler CPU-annotation windows lie about GPU time (async) — attribute
kernels by correlation ID; micro-benches must rotate HBM-cold weights AND serialize by stream
order (independent kernels in one graph run concurrently). torch.compile is a dead end here
(custom fused ops raise NotImplementedError in forward_native).

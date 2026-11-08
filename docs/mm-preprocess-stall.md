# Multimodal preprocessing stall — the "requests vanish from the scheduler" bug

**Status: ROOT CAUSE FOUND, FIX MEASURED + DEPLOYED (2026-11-08 16:20), awaiting
restart.** The engine tree on testcomp2 now carries patch 0016 and the kit's
`serve.sh` carries the new flags; the running server (pid 1164753) is untouched
and activates the fix on the next `llmqwen` restart. Verification gate after
restart: boot line in `logs/serve.log` shows `--image-processor-backend
torchvision` + `SGLANG_MM_PROCESSOR_DEVICE=cpu`, and a 26-screenshot request
reaches the scheduler (ReqTimeStats) within ~10 s.
Investigated 2026-11-08 against `bug-scheduler.md` (the 13:36 incident report).
Live box: testcomp2. All evidence in `.slim/deepwork/ses_e443d5d79ffer7xOjA4sAKC36Y.md`
(claim register + forensic rounds 1–24).

## TL;DR

Requests never reach the scheduler — they stall in the **tokenizer manager's
multimodal preprocessing** (`--image-processor-backend pil`), which is **~500×
slower than the docs assume** on this box: **14 s per screenshot, ~6 min for a
26-image request**, on a pool of only **2 worker threads**. The proxy's 300 s
timeout fires first, the client resends, and the resend is fast because the mm
embedding cache is warm from the first attempt. The server is healthy
throughout; "vanish from the scheduler" is an observability artifact.

**Fix (measured, not yet applied): `--image-processor-backend torchvision` plus
a one-line patch making the fast processor run on CPU (`device="cpu"`).**
26 images: **1.22 s** (vs 368 s on the PIL path), CPU-only, so the GPU-0 OOM
that motivated the `pil` pin in 2026-09 cannot come back.

## 1. Why `bug-scheduler.md`'s reading is wrong

- `Abort request … not found in rid_to_state; likely already finished/removed`
  is the tokenizer manager's **benign "abort came too late" notice**
  (`tokenizer_manager.py:3167-3182`): it fires when the finish already beat the
  abort echo. The report reads it as "the tokenizer manager still holds the
  rid" — backwards.
- The missing `ReqTimeStats` lines for the four stuck entry_times are not the
  scheduler losing requests: the requests were **never submitted to the
  scheduler**. `#queue-req` counts the scheduler queue only; a request in mm
  preprocessing is invisible to it and to every log line.
- Verified against the live log: no aborts 13:41–13:47, no WARNING/ERROR
  13:30–13:45, scheduler rank-0 py-spy shows a normal loop (hicache
  bookkeeping → all-reduce), decode at 600–1000 tok/s throughout.

## 2. Root cause (live-captured during a recurring incident)

The incident was happening **while** investigating (15:35–15:47, same
fingerprint as 13:36): 5 requests in proxy state `sent` (no first byte) for
300–600 s.

- `py-spy dump` of the tokenizer process (pid 1164753): both
  `sglang-mm-processor_0/1` threads at **99.9 % CPU, ~100 % system time**
  (TIME+ 203 min / 198 min), frames inside transformers 5.12.1
  `image_processing_pil_qwen2_vl.py` → `_preprocess` (resize / rescale /
  normalize / `np.concatenate`).
- **Fault storm:** 36.2 M / 26.7 M minor faults in 8 s per thread (≈4.5 M/s) —
  the PIL path's `image.astype(np.float64) * scale` temporaries (2× the
  float32 size) churn mmap pages on a box with swap 7.8/8 GB used.
- The stuck requests carried **1–26 base64 screenshots** (1018×1622 →
  1024×1632, grid `[1,64,102]`, 19 584 patches each). Every one of the 19
  affected requests today is image-bearing; median prompt 140 k tokens.
- The mm pool is **exactly 2 workers** (`qwen_vl.py:
  auto_mm_processor_worker_num = 2`), with **no timeout** on preprocessing.
  Everything queues behind those two threads.
- The 5 CLOSE-WAIT sockets show the proxy's aborted connections are never
  read — the asyncio loop is starved behind the two GIL-hammering threads.

## 3. The decisive measurement (isolated, CPU-only, on the box)

`Qwen2VLImageProcessorPil` (what `--image-processor-backend pil` selects):

| path | 26 × 1018×1622 | per image |
|---|---|---|
| `backend="pil"` (production today) | **368.04 s** | 14.2 s |
| `backend="torchvision"` (fast) | **0.77 s** | 30 ms |
| fast, `device="cpu"` (the fix) | **1.22 s** | 47 ms |

- The README's "~tens of ms per image" for the pil backend is **wrong by
  ~500×** — it was never measured. The 14 s/image is real CPU work amplified
  by the box's swap pressure; the isolated run alone took 368 s and touched
  ~1 GB of float64 temporaries per 26-image batch.
- fast-vs-pil pixel parity: max abs diff **0.0157** (normalized units) —
  the known resize-interpolation difference; the checkpoint's own
  `preprocessor_config.json` declares `Qwen2VLImageProcessorFast` as the
  intended processor, so torchvision parity is the reference path.

## 4. Why `pil` was pinned, and why the fix is safe

`docs/STATUS.md` (2026-09-11): the fast image processor runs torch ops on
**GPU 0 inside the tokenizer process**, which OOMs when the scheduler has
filled the card. That mechanism is real and still present:
`base_processor.py:_fast_image_processor_device()` returns
`f"cuda:{server_args.base_gpu_id}"` (line 609 in the pinned tree) and
`process_mm_data` passes it as `kwargs["device"]` (line 696-698).

The pin fixed the OOM by accepting the 500× slowdown. The **correct axis is
the device, not the backend**: `backend="torchvision"` + `device="cpu"` —
measured 1.22 s, zero GPU allocation. Verified in the deployment venv
(transformers 5.12.1): `f(images=[img]*26, device="cpu")` →
`device=cpu, dtype=float32, shape=(169728, 1536)`.

## 5. Upstream state (the user's question)

**No upstream issue or PR matches this symptom** (image requests stalling in
mm preprocessing before the scheduler). Verified directly (2026-11-08):

- **sglang**: no issue/PR for "PIL backend slow / preprocessing hang / stuck
  before scheduler". The only `pil`-related items are cosmetic (e.g. PR
  #42127 grayscale→RGB in the PIL backend). The known-open caveats in
  `serve_best_kv_offload.sh` (#33714, #36743, #37613) are unrelated.
- **Upstream `main` today** (verified by fetching `base_processor.py`):
  `_fast_image_processor_device` **still hardcodes `cuda:{base_gpu_id}`** —
  upstream has *not* fixed the GPU-0 contention; it has papered over it with
  new machinery the pinned tree lacks: `retry_on_cpu_after_gpu_oom`
  (re-preprocess on CPU after a CUDA OOM), a per-call
  `_temporary_fast_processor_cuda_pool` (MemPool release), and
  `_preprocessing_competes_with_the_scheduler()` gating the worker count to
  **1** on the GPU path (CPU path gets 2). None of these makes the PIL
  backend faster; the 2-worker default for qwen (`auto_mm_processor_worker_num
  = 2`) is unchanged.
- **transformers**: no issue reports the PIL Qwen2VL backend costing seconds
  per image. The closest real signal is the Qwen3-VL repo's own
  `QwenLM/Qwen3-VL#1678` ("PIL.Image.resize is CPU-bound and single-threaded
  … dominates the latency") — same mechanism, different wrapper.
- Related-but-different: sglang **#28247** (open, Aug 2026) — API server hangs
  on large multimodal data because `process_mm_data_async` runs heavy work on
  the event loop; same subsystem family, different root cause (ours is the
  worker threads, not the loop).
- ⚠️ A research lane that cited sglang #38999 / #34513 / transformers #34500
  as fixes was **cancelled**: every number was verified wrong (GLM quant
  mapping / router RFC / diffllama). Treat uncited agent output as claims.

## 6. Remediation (recommended, pending approval — all require a restart)

1. **`scripts/serve.sh` + `serve_best_kv_offload.sh`:** `MM_IMG_BACKEND`
   default `pil` → `torchvision`.
2. **Patch 0016** (`0016-mm-processor-cpu-device.patch`): one-line change in
   `base_processor.py::_fast_image_processor_device` — return `"cpu"` instead
   of `f"cuda:{base_gpu_id}"` (guarded by an env/flag so the patch stays
   small and reviewable). This is the piece that keeps the 2026-09-11 GPU-0
   OOM dead.
3. Optional hardening: `--mm-processor-worker-num 4` (CPU preprocessing
   parallelism is real on this box — 256 cores) and/or a proxy-side
   preprocessing timeout surfacing 503s instead of silent 300 s stalls.
4. Docs: fix the README "~tens of ms" claim (this file replaces it).

**Validation gate:** after restart, `scripts/bench/`-style replay of a
26-screenshot request must reach the scheduler (ReqTimeStats appears) within
~10 s; the watcher (`bench/watch_server_leaks.ps1`) stays armed; the
`#queue-req`/dashboard `sent` state should no longer linger for image-heavy
requests.

## 7. Secondary observations (separate from this bug)

- The "repetition loops" note in `bug-scheduler.md` was out of scope here.
- The box is at swap 7.8/8 GB with ~128 GB MemAvailable — the hicache tier
  (440 GB) + PLE pinned mappings dominate. Not a leak (forensics r15), but
  the swap pressure is the amplifier that turns 6.9 s of image work into
  minutes of fault churn. Reducing mm-path temporaries (the fix) removes the
  trigger; the pressure itself is a standing capacity decision.

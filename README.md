# sglang-flashnext-sm120

**Qwen3.8-Flash-Next (180B MoE, NVFP4) at 231 tok/s single-stream on a single RTX PRO 6000 Blackwell (96 GB, sm120)** —
now deployed on **8× RTX 5090 (sm120, 32 GB/card) at TP8** on machine `testcomp2`.

Patches, launch scripts and benchmarks for serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` at TP1
on the official SGLang `qwen4-main-squashed` branch — no Docker, no fork. Builds on the sm120
groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000),
then goes ~35-45% past its published numbers.

## Deployment target: testcomp2 (8× RTX 5090, TP8)

The 5090 is sm120 like the RTX PRO 6000, so all nine patches apply unchanged — but the
scripts are re-pointed and re-tuned for this box (32 GB/card, TP8, 1 TB host RAM):

| | value |
|---|---|
| this repo on the machine | `/mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang` |
| sglang checkout + venv | `sglang-official/` inside the repo (branch `qwen4-main-squashed` @ `4ccff141` + patches) |
| model checkpoint | `/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| caches | `cache/` inside the repo (gitignored; `CACHE_BASE` to override) |

Run (after `git pull` on the machine):

```bash
bash scripts/apply_patches.sh   # once per pull: nine sm120 patches into sglang-official (no rebuild)
./scripts/serve_best.sh         # DEFAULT: TP8+EP8, 8-way, fp8 KV + fp8 stack, 262144 ctx, MEMFRAC 0.80, :1025
./scripts/serve_single.sh       # one huge session: 786432 ctx (YaRN ×3), max KV pool
```

Both launchers run sglang **in the foreground** — Ctrl+C stops it (a cleanup sweep reaps any
leftover SGLang GPU processes); logs also land in `logs/serve.log`. Use tmux/screen if you
want the server to survive a disconnect. There is no systemd unit and no auto-start.

**Why EP8:** pure TP8 cannot load this checkpoint — `moe_intermediate_size 640` sharded 8-way
= 80/rank requires NVFP4 w13/w2-scale swizzle padding, which the loader refuses for gated
experts (load-time assert). `--ep-size 8` keeps the full 640 per expert (64 experts/rank).
The NVFP4 swizzle needs the per-rank intermediate to be a multiple of 64, so pure-TP is only
valid for TP 1/2/5 — none of which fits 8×32 GB. Fallback candidate (untested, 4 cards only):
`TP=4 EP_SIZE=4 ./scripts/serve_best.sh`.

**Bring-up fixes (all patched in, see `patches/`):**
- `0007` — FR-Spec token map broke at TP>1: the draft lm_head gather indexed the
  vocab-parallel slice with global ids (device assert). The patch gathers the full head first.
  Required for TP8+EP8 + spec decoding.
- `0008` — comma-separated `--served-model-name "pennyroyal,glm-5.3-flash"`: both names listed
  on `/v1/models` and accepted as model ids.
- `0009` — agent frameworks that emit malformed tool schemas (`"required": {}` instead of an
  array) used to get a 400; the patch drops the invalid field before validation.
- `serve.sh` pins `--image-processor-backend pil`: the transformers *fast* image processor runs
  torch ops on GPU 0 inside the tokenizer process, which OOMs when the scheduler has filled the
  card. The pil backend preprocesses on CPU (~tens of ms per image, overlapped with GPU work).
- `serve.sh` prepends `.venv/bin` to `PATH` — sglang's JIT builder shells out to `ninja`, which
  only lives in the venv.
- `serve_best.sh` runs a post-ready warmup (one short + one ~40k-token request) so the lazy
  Triton kernels (pool alloc, sparse GQA prefill) device-load inside the startup window instead
  of on the user's first real request.
- Memory: MEMFRAC 0.90 → graph-capture OOM; 0.85 → text clean but the on-GPU image preproc
  OOM'd; **0.80 is the validated default** (KV pool ~1.62M tokens). With the pil backend, 0.85
  is likely viable again — try it if you want the bigger pool. Host RAM: only tens of GB are
  actually needed; the big "used" figure during load is page cache of the 206 shard files
  (reclaimable), so a 256 GB cap only slows (cold) weight loads.

The launchers refuse to start while GPUs are busy (the box also hosts a vLLM TP8 server that
occupies all eight cards — stop it first; offending PIDs are printed, `FORCE=1` overrides).
The box is headless (display Disabled on all 8 GPUs), so graph capture limits can stay at 8.
First start: model load + flashinfer autotune, ~20 min; `curl -s http://127.0.0.1:1025/health`
→ HTTP 200 when ready. 5090/TP8 performance numbers are TBD — TP1 figures below are the
provenance from the 96 GB card.

**Verified on testcomp2 (2026-09-11):** prompts up to 245,109 tokens at 1/2/4/8-way
concurrency, agent tool-call loops (43k-token prefixes, ~46% prefix-cache hit ratio), and image
requests all pass. The only expected log noise is `multimem all-gather disabled (CUDA driver
error: invalid device ordinal)` — NVLink multicast isn't available on 5090s, normal all-gather
is used instead.

## Results

| | jpezzulli | this repo (temp 0.6) | this repo (greedy, lossless) |
|---|---|---|---|
| Decode, 1 stream | 171 tok/s | **231 median / 243 best** | 203 |
| Decode, 4 streams | 428 | **620–657** | 549 |
| Decode, 8 streams (optional 8-way config) | — | 758 | |
| Prefill (2.5K) | 10–12K | ~10.4K tok/s | |
| TTFT | — | ~135 ms | |

Two launch profiles:

| profile | context window | KV pool | decode C1 |
|---|---|---|---|
| `serve_best.sh` — interactive + agents (4-way) | 262144 (native) | ~572K tokens | **231 tok/s** |
| `serve_single.sh` — one huge session | **786432** (YaRN ×3) | **~827K tokens** | 185 tok/s |

The long-context profile trades the fp8 dense-weight copies back for KV head-room and is
validated with needle retrieval at 653K-token depth (start / middle / end all pass).

Validated with greedy/needle/cached-prefix/GSM/code gates and 2.4M tokens of soak testing
(0 errors, flat VRAM/RAM).

## Contents

```
patches/            nine patches against sgl-project/sglang @ qwen4-main-squashed (see patches/README.md)
scripts/            serve.sh (knobbed launcher) · serve_best.sh · serve_single.sh (786K ctx)
                    apply_patches.sh (idempotent, into sglang-official) · do_build.sh
                    bench_sglang.py · make_hot_tokens.py
docs/               STATUS.md (ops guide) · PERF_CEILING.md (analysis + dead-ends)
results/            benchmark JSONs, baseline -> final
hot_tokens_64k.pt   FR-Spec draft-vocab map
```

## The optimizations

**Patches** (0001–0003 unblock sm120; 0004–0006 are the speed work, all env-gated; 0007–0009
are the TP8/EP8 bring-up fixes, all applied by `scripts/apply_patches.sh`):
1. `0001b` — RecoverSSM + WY output-only MTP verify on FlashInfer for sm120.
2. `0002` — FP8-KV tile dequant for the QSA sparse prefill (2× KV capacity).
3. `0003` — fp32 prefill state for the sm120 FlashInfer GDN kernel.
4. `0004` — Triton low-M GEMM: cuBLAS-under-graph-capture runs the decode projections at
   20–75% of DRAM bandwidth on sm120; this kernel reaches ~90%.
5. `0005` — W8A16 fp8 weight-only serving of the dense bf16 stack (85% of per-step traffic,
   untouched by the NVFP4 checkpoint). Runtime-only: the checkpoint is never modified.
6. `0006` — same fp8 treatment for the HyperConnection mix and the lm_head (also halves
   every MTP draft step's logits).
7. `0007` — FR-Spec token map: all-gather the vocab-parallel lm_head before the hot-token
   gather (TP>1 device assert).
8. `0008` — multi-alias `--served-model-name` (comma-separated, all listed on `/v1/models`).
9. `0009` — tolerate malformed tool schemas (`"required": {}` → dropped, not 400).

**Config levers** (in `scripts/serve.sh`, each documented inline with its measured ladder):
- Relaxed MTP acceptance `0.3` (C1 179 → 231; exact at temp 0, set `1.0` for lossless sampling).
- FR-Spec: draft head scores a 64K hot-token subset of the 248K vocab (verify stays exact).
- 8-way concurrency: `--max-mamba-cache-size` must be ~6× max-running-requests or the
  speculative CUDA graphs silently cap at bs=4 (8-way used to run *slower* than 4-way).

## Reproduce

On any machine (historical flow, single RTX PRO 6000):

```bash
# model checkpoint (~135 GB; the ~50 GB PLE n-gram table is served from host RAM)
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir Qwen3.8-Flash-Next-NVFP4
# note: hf_xet can stall on the largest shards; scripts/serve.sh's docs and
# docs/STATUS.md describe the curl fallback that resumes reliably.

# Pin the exact commit everything here was verified against (single commit, no history):
git init sglang-official && cd sglang-official
git remote add origin https://github.com/sgl-project/sglang
git fetch --depth 1 origin 4ccff141dbe992794f9da6c3aa23535b4f72000d
git checkout FETCH_HEAD
cd ..
# (qwen4-main-squashed @ 4ccff141dbe992794f9da6c3aa23535b4f72000d, verified 2026-09-11)

# This one checks out the CURRENT tip of the branch instead — patches may need
# adjustments if the upstream branch has moved since the pinned commit:
# git clone -b qwen4-main-squashed https://github.com/sgl-project/sglang sglang-official

cd sglang-official && bash ../scripts/do_build.sh && cd ..
bash scripts/apply_patches.sh       # 0001b+0002+0003+0007+0008+0009 via git apply; 0004-0006 via git am
# point scripts/serve.sh at your paths (TARGET_MODEL, REPO), then:
./scripts/serve_best.sh             # OpenAI API on :1025, ~5 min to ready
```

On testcomp2 the sglang checkout already lives in `sglang-official/` inside the repo and is
built (`scripts/do_build.sh` already points there) — only `scripts/apply_patches.sh` is needed
after each `git pull`.

Single-GPU-with-display safety knobs (learned the hard way): keep `--cuda-graph-max-bs`
small, cap JIT compilation with `MAX_JOBS=4`, run under a systemd `MemoryMax` cage.
Details and every explored dead-end: `docs/`.

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang), branch `qwen4-main-squashed`
  (pinned at `4ccff141dbe992794f9da6c3aa23535b4f72000d`)
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — the sm120
  RecoverSSM/WY and fp8-QSA work this repo ports and builds on

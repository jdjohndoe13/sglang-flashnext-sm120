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

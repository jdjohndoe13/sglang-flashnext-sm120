# Engine patches (against sgl-project/sglang @ `qwen4-main-squashed`)

Apply all fifteen with the idempotent helper (safe to re-run after every `git pull` of this repo):

```bash
bash scripts/apply_patches.sh
```

The install is editable (`uv pip install -e python`), so patches take effect without a rebuild.

| patch | what it does | applied as |
|---|---|---|
| `0001b-recoverssm-wy-sm120-PORTED.patch` | RecoverSSM + WY output-only MTP verify for the sm120 FlashInfer GDN kernel (ports jpezzulli 280825c3e2) | `git apply` (uncommitted) |
| `0002-fp8-qsa-tile-dequant.patch` | FP8-KV tile dequant for the QSA sparse prefill → 2× KV capacity | `git apply` (uncommitted) |
| `0003-sm120-fp32-prefill-state.patch` | fp32 prefill state for the sm120 FlashInfer GDN kernel | `git apply` (uncommitted) |
| `0004-sm120-lowm-triton-gemm.patch` | Triton low-M GEMM: decode projections at ~90% of DRAM bandwidth (cuBLAS-under-graph-capture only reaches 20–75% on sm120) | `git am` (commit) |
| `0005-sm120-fp8-weight-only.patch` | W8A16 fp8 weight-only serving of the dense bf16 stack (85% of per-step traffic). Runtime-only; checkpoint untouched | `git am` (commit) |
| `0006-sm120-fp8-hc-lmhead.patch` | Same fp8 treatment for the HyperConnection mix and lm_head (also halves every MTP draft step's logits) | `git am` (commit) |
| `0007-eagle-frspec-vocab-parallel-head-gather.patch` | FR-Spec token map at TP>1: all-gather the full lm_head before the hot-token slice (global ids vs vocab-parallel slice → device-side OOB gather assert) | `git apply` (uncommitted) |
| `0008-served-model-aliases.patch` | Comma-separated `--served-model-name "a,b"`: all names listed on /v1/models and accepted as model ids (chat/completions already accepted any name) | `git apply` (uncommitted) |
| `0009-tolerant-tool-schema-required.patch` | Drop malformed tool-schema `required` (e.g. `{}` instead of an array, emitted by some agent frameworks) before JSON-Schema validation instead of 400-ing | `git apply` (uncommitted) |
| `0011-hot-head-reshard.patch` | Re-shard the hot-token-sliced draft lm_head to 1/tp per rank. After 0007 every rank held the full 65,536-row slice, so the vocab-parallel logits all-gather concatenated tp copies (trimmed to vocab_size); the draft's argmax then returned `[block_offset + hot_rank]` values that are not hot ranks → `hot_token_id[topk_index]` device assert, killing all ranks | `git apply` (uncommitted) |
| `0012-hicache-load-fence.patch` | Fence HiCache load-back H2D behind the forward stream (upstream #36738, adapted). Without it, overlap-scheduling writes to reclaimed pages race the H2D restore → corrupted restored prefixes → the cross-session tool/path hallucinations of 2026-10 docs. Required before re-enabling `HICACHE=1` | `git apply` (uncommitted) |
| `0013-mamba-cow-load-fence.patch` | Deferred mamba COW waits for the final HiCache layer (upstream #36743, adapted). Restored radix slots are published while their layerwise H→D is in flight; whole-slot COW could copy partial mamba state. Companion to 0012 | `git apply` (uncommitted) |
| `0014-hicache-qsa-indexer-state.patch` | **Preserve QSA indexer state through HiCache (upstream #39893, ported to the fork API).** The QSA compressed index keys live in `qsa_compressed_k_buffer_pool` — outside `full_kv_pool` — so HiCache restored KV + mamba but re-used another request's indexer keys → the indexer selected wrong attention blocks → the 2026-11-01 hallucination class that 0012/0013 alone did NOT prevent (§13). Adds `QSAIndexerPoolHost` (host mirror, page-row packed incl. MTP draft), the INDEXER sidecar on the mamba stack, and the consumer-fence `_wait_for_layer` in `get_qsa_compressed_k_buffer`. REQUIRED before re-enabling `HICACHE=1`; with it, the expected boot log is `pools=KV + MAMBA + INDEXER` | `git apply` (uncommitted) |
| `0016-mm-processor-cpu-device.patch` | **Run the fast (torchvision) image processor on CPU when `SGLANG_MM_PROCESSOR_DEVICE=cpu` (set by serve.sh).** The fast processor's device is hardcoded to `cuda:{base_gpu_id}` — the tokenizer process shares GPU 0 with the scheduler, and the September GPU-0 OOM was fixed by pinning `--image-processor-backend pil`. That pin is itself the 2026-11-08 "requests vanish from the scheduler" incident: the PIL path measures **14 s/image** (368 s for 26 screenshots; float64 temporaries + ~4.5M page-faults/s) on a 2-worker pool with no timeout, so image requests stall *before* the scheduler and look lost. With this patch + `torchvision` backend: **1.2 s/26 images, CPU-only** — the GPU-0 OOM cannot come back. Env-gated: inert without the var. Full analysis: `docs/mm-preprocess-stall.md` | `git apply` (uncommitted) |
| `0015-mrope-fused-rope-fix.patch` | **Fused QK RoPE kernel must honor mrope `[3,T]` positions (upstream #34446 / e6355774, adapted).** The fused GemmaRMSNorm+NeoX-RoPE+gate kernel indexed positions as 1D, so multimodal requests (positions = mrope temporal/height/width rows) silently used only the temporal row for every rotary lane: height/width positional info was discarded for all vision tokens → systematic grounding/click errors and misreported image sizes (upstream #35345/#35772/#36762: center accuracy 26.7%→81.7%, y-error 4.6×→0.06×). Kernel gains an `MROPE` path with a per-lane axis map; `mrope.py` builds the map for all three layouts (glm/interleaved/contiguous), older triton/xpu kernels keep the glm-only map; `qwen3_5.py` passes the map for `[3,T]`. Ships the upstream CPU + CUDA tests (CI registration stripped). No effect on text-only requests | `git apply` (uncommitted) |

0001–0003 unblock sm120 (required); 0004–0006 are the speed work, all env-gated via
`SGLANG_ENABLE_SM120_LOWM_BF16_GEMM` / `SGLANG_SM120_LOWM_FP8_WEIGHT` / `SGLANG_SM120_LM_HEAD_FP8`.
0007–0011 are TP8/EP8 deployment fixes found while bringing up the 8× RTX 5090 box (all
required there; harmless at TP1). 0012–0013 are the HiCache load-fence pair (upstream
#36738/#36743 adapted); they are no-ops with `HICACHE=0` and REQUIRED before re-enabling
`HICACHE=1` — see hallucination-investigation.md §7–§9. All targets are sm120 (RTX PRO 6000
and RTX 5090), so all patches apply unchanged on both.

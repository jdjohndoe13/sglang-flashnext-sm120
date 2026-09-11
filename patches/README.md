# Engine patches (against sgl-project/sglang @ `qwen4-main-squashed`)

Apply all nine with the idempotent helper (safe to re-run after every `git pull` of this repo):

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

0001–0003 unblock sm120 (required); 0004–0006 are the speed work, all env-gated via
`SGLANG_ENABLE_SM120_LOWM_BF16_GEMM` / `SGLANG_SM120_LOWM_FP8_WEIGHT` / `SGLANG_SM120_LM_HEAD_FP8`.
0007–0009 are TP8/EP8 deployment fixes found while bringing up the 8× RTX 5090 box (both
required there; harmless at TP1). All targets are sm120 (RTX PRO 6000 and RTX 5090), so all
patches apply unchanged on both.

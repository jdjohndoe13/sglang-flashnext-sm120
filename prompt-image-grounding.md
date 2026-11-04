# prompt-image-grounding.md — the residual 1.44× grounding scale (handoff, 2026-11-03)

> **✅ RESOLVED 2026-11-03 — do not re-run the chase.** §4 exp. 1 was the
> decisive test and it PASSED: asked in the model's native convention
> (relative 0–1000 of the image), grounding is exact (≤0.1 px at both
> 1622×1018 and 1920×1080; `node scripts\bench\grounding_conv.mjs`). The
> absolute-px 1.44× is a translation artifact through a fabricated
> ~2340×1400 canvas (size-independent; blank-image claim 1000×667). The
> "bias" branch of exp. 1 (~1/1.44 answers) did NOT happen — perception is
> fine. Fix = client-side convention pin: always ask 0–1000, denormalize
> `x/1000 × W_real`. Full write-up: `docs/ui-misclick-research.md` §10.
> The §4/§5 text below is kept as the pre-resolution record.

**One-line brief:** the M-RoPE fix (patch 0015) is in and verified — vision
grounding no longer scatters/swaps — but the model still reports coordinates
in a canvas ~1.44× too big (self-consistently). This file is everything
needed to chase and fix that residual in a fresh session.

Related docs: `docs/ui-misclick-research.md` (the full investigation, §3
pre-patch table, §8 A/B result, §9 evidence), `AGENTS.md` "Open Thread — UI
misclick / grounding".

---

## 1. What is already FIXED (do not redo)

- **M-RoPE h/w loss (patch 0015)** — the fused QK-RoPE kernel read `[3,T]`
  positions as 1D; vision tokens lost height/width. Patch applied to the
  live tree (`sglang-official`, uncommitted), port of upstream #34446.
- **0015 launch bug** — the patch declared `MROPE: tl.constexpr` but never
  passed it; first boot crashed all 8 ranks
  (`TypeError: dynamic_func() missing 1 required positional argument: 'MROPE'`).
  Fixed 2026-11-03: `MROPE=mrope,` inserted in the kernel launch
  (`fused_qk_rmsnorm_rope_gate.py`, between `HAS_GATE` and `ENABLE_PDL`),
  patch file corrected + `git apply --check`-validated.
- **Current server is the patched boot** (21:02:58, all 8 ranks
  `pools=KV + MAMBA + INDEXER`, zero MROPE errors, `serve_best_kv_offload.sh`
  daily profile). Endpoint `http://192.168.2.208:19999/v1` (port-forward
  from this Windows box to testcomp2's SGLang on :1025).
- Transport is 1:1: `prompt_tokens: 1738` for the 1622×1018 test image
  (1632×1024 smart-resize grid → 1632 vision tokens + ~104 text). The
  server never resizes on this path and never rescales model output
  coordinates (no grounding helper exists in `srt/` — conversion is
  client-side).

## 2. The residual problem (the thing to fix)

Post-patch behavioral A/B (`node scripts\bench\grounding_ab.mjs`, 3 runs,
temperature 0, thinking off, 1622×1018 canvas, red circles at
(1000,400) (300,700) (1400,900)):

| circle (x-order) | truth | reported (run A / run B) | ratio |
|---|---|---|---|
| 1 | (300, 700) | (433, 999) / (433, 983) | (1.443, 1.427) |
| 2 | (1000, 400) | (1443, 570) / (1437, 563) | (1.443, 1.425) |
| 3 | (1400, 900) | (2021, 1285) / (2013, 1267) | (1.444, 1.428) |

- The model claims the image is **2337×1433 / 2346×1456** (stable across
  runs; true 1622×1018). Claimed-size ratios ≈ x 1.441, y 1.408 — the
  center ratios match the claimed canvas on BOTH axes (run A/B above).
- **Self-consistent:** the model's coordinate space = its claimed canvas.
  Normalizing a reported coord by the claimed size lands on the truth:
  `433 × 1622 / 2337 = 300.5 ≈ 300` ✓.
- **Stable but not a fixed constant:** pre-patch run 2 (under load) claimed
  1754×1113 — the pre-patch variance was mrope damage; post-patch it
  settled at ~2337×1433.
- **Pre-existing:** pre-patch run 1 (idle box) claimed the SAME 2337×1459.
  The bias predates and is independent of 0015 — do NOT revert 0015.
- Field symptom: misclicks toward the bottom-right growing with distance
  from the top-left (y=400 → +170 px, y=900 → +385 px).

Gotchas when reading the script output:
- The script prints per-marker ratios in **input order**; the model
  enumerates circles **x-sorted**. Always match by x-order (table above).
- The script's markers are FIXED coordinates — do NOT run it with a
  smaller `--size` (e.g. 800x600): the markers fall outside the canvas and
  the model's "identical points" answer is then correct, not a failure.
  For a second size use ≥1500×950, or patch the script to scale markers.

## 3. The key lead — the model's native convention is NOT absolute px

`docs/ui-misclick-research.md` §5 (verified citations):

| Model | Output convention | Conversion |
|---|---|---|
| Qwen2-VL | normalized [0, 1000) | `x/1000 × W` |
| Qwen2.5-VL | absolute px in smart-resized image | `x × W_orig / W_resized` |
| **Qwen3-VL / 3.5+ (this model)** | **relative, 0–1000 of the original image** | `x/1000 × W` |

The A/B prompt EXPLICITLY asks for absolute pixel integers, which fights
the model's native convention. The model may be internally in a different
space and "translating" through a wrong canvas estimate (its claimed
size). The upstream fix-report (#34446, Qwen3.8-27B) measured y-error
0.062× AFTER the fix — i.e. upstream's model grounds correctly when asked
in its native way, so the 1.44× is specific to this checkpoint/prompt
interaction, not an inherent model defect.

## 4. Experiments to run (in order, all cheap)

Run from this Windows box against `http://192.168.2.208:19999/v1`
(model `qwen-3.8-flash-next`, `chat_template_kwargs: {enable_thinking: false}`,
temperature 0). Reuse `grounding_ab.mjs`'s PNG builder + request shape.

1. **Decisive convention test — ask for the native 0–1000 grid.**
   Same image, same prompt but: "report each circle's center as relative
   coordinates in [0, 1000] of the image (x/width×1000, y/height×1000)".
   - If grounding is FINE (the 1.44× is a translation artifact of forcing
     absolute px), the answer ≈ **(185, 688) / (617, 393) / (863, 884)**
     (truth ÷ 1622×1018 × 1000). Resolution = client-side only: always
     use the native convention, denormalize `x/1000 × W`.
   - If the model's internal canvas really IS the 1.44× one, the answer
     is truth ÷ 2337×1433 × 1000 ≈ **(128, 488) / (428, 279) / (599, 628)**
     — i.e. ~1/1.44 of the correct values (note: dividing by the too-big
     canvas makes the 0–1000 answer SMALLER, the inverse of the
     absolute-px symptom). Bias is then perceptual — keep hunting
     server/model side.
2. **Size-dependence test.** The script's markers are fixed at
   (1000,400) (300,700) (1400,900), so only sizes ≥ ~1450×950 keep them
   inside the canvas. Run `node scripts\bench\grounding_ab.mjs
   http://192.168.2.208:19999/v1 1920x1080` (markers fit, same image
   scale as the win-b desktop class). If claimed size ≈ 1.44× for EVERY
   size → constant multiplier bug; if claimed size is a CONSTANT
   ~2337×1433 regardless of input → the model anchors to a fixed canvas
   (training-resolution prior). (1280x720 is NOT usable without patching
   the script to scale its markers — the third marker would be outside.)
3. **Ask-size-only, no circles.** Send a blank 1622×1018 PNG (markers
   removed) and ask only the image size. If it still says 2337×1433, the
   canvas estimate is baked into the vision path (grid/pos-embed), not a
   reasoning artifact.
4. **Processor math check (server side, read-only).** On testcomp2
   (`remote-exec ... --host testcomp2`, see §6):
   `/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4/preprocessor_config.json`
   — image factor is 32 per the doc (the `smart_resize` factor-28 in
   `srt/multimodal/processors/qwen_vl.py` is VIDEO-ONLY — do not use it
   for image math). Check `vision_config` (spatial_merge_size,
   num_position_embeddings) and how the image grid_thw is computed —
   check whether the model's per-patch positional span could be
   mis-scaled by ~√2 ≈ 1.414 (note: 1.441/1.408 ≈ 1.424 on both axes —
   suspiciously close to √2, and the per-axis ratios differ slightly,
   which smells like a grid-interpolation scale factor).
   Related family: Tzafon Northstar-CUA-Fast — scaling the positional
   embedding by 3× improved click accuracy 40%→80% (positional-decay
   mechanism, `ui-misclick-research.md` §6.8).
5. **Cross-check the live tree's mrope path for a second scale.**
   `get_cos_sin_with_position` / `forward_native` in
   `python/sglang/srt/layers/rotary_embedding/mrope.py` — verify the
   restored h/w rows feed the cos/sin table at the SAME scale the
   checkpoint was trained with (patch 0015 changed which row is read,
   not the scale — but the pre-patch run 1 identical bias means the
   question is what the ORIGINAL (buggy) kernel and the fixed kernel
   share: whatever feeds the vision tokens' temporal row).

## 5. Fix options (depending on experiment outcome)

- **Convention (likely):** pin the harness to the native 0–1000 grid
  (KB rule: `b:\ai\setup-remote-environment\.config\opencode\README.md`
  and the agents' computer-use instructions), denormalize client-side.
  Zero server change. Update `coords_check_loop` SKILL.md + any
  Screenshot+Click(loc) guidance to convert.
- **Self-normalization (works today, no diagnosis needed):** ask the model
  for the image size in the same request and map
  `x_click = x_reported × W_real / W_claimed`. Measured exact on the A/B
  data (300.5 vs 300). Costs one extra field in the prompt.
- **Loop (already works):** `coords_check_loop` converges from the bias
  (proved against the WORSE biased+scattered model: 2–3 iterations). The
  uniform bias is easier to walk back.
- **Server-side (only if exp. 4/5 finds a concrete scale bug):** fix the
  grid/pos-embed scale in the vision path; re-run the A/B.

## 6. Environment & access (this box)

- Server: testcomp2 (ssh `testcomp2`, 192.168.2.167), repo
  `/mnt/data/shared/models/sglang-flashnext-sm120`, log
  `logs/serve.log`, server on :1025, port-forwarded to
  `192.168.2.208:19999` on this Windows box.
- **All remote commands via the `remote-exec` binary** (see the
  `remote_exec` skill): write a `.sh` with the write tool, then
  `remote-exec <path> --host testcomp2 --timeout 60` — path UNQUOTED,
  read `output.txt` from the printed `LOCAL_DIR`.
- A/B: `node scripts\bench\grounding_ab.mjs` from
  `D:\shared\models\sglang-flashnext-sm120` (writes
  `grounding_ab_input.png` to cwd).
- Shell scanner rules (host): no inline `python -c` (use a .py file),
  no spaces in quoted args, no parentheses in positional args.
- Server restart (if you change server code): the script runs in the
  FOREGROUND — stop the current one, `bash scripts/serve_best_kv_offload.sh`
  (~20 min cold), boot gate `pools=KV + MAMBA + INDEXER` on all 8 ranks,
  then re-run the A/B. Patches 0014/0015 are uncommitted in the engine
  tree — re-apply via `bash scripts/apply_patches.sh` if the tree is
  refreshed.
- vLLM fallback: a docker (`vllm-DeepSeek-V4-Flash.sh`) may re-grab the
  GPUs + :1025 after the SGLang server dies — check before any
  re-run (`/v1/models` → `owned_by` must be `sglang`, not `vllm`).

## 7. Evidence index

- `docs/ui-misclick-research.md` §3 (pre-patch), §8 (post-patch A/B +
  verdict), §9 (run dirs).
- `re_20261103-210600-0471` — boot gate `pools=KV + MAMBA + INDEXER`.
- `re_20261103-202831-6221` — the MROPE launch fix applied to the live tree.
- `re_20261103-202404-8591` — the crash log (launch bug).
- `re_20261103-132242-9826` — pre-patch marker test (2337×1459, same bias).
- `re_20261103-114133-1455` — token A/B proving 1:1 transport.
- Post-patch A/B runs: local, 21:05–21:06 (outputs in this session's
  transcript; also summarized in ui-misclick-research.md §8).

## 8. Success criteria for this issue

- The A/B (or its 0–1000-convention variant) shows reported centers within
  a few % of truth on BOTH axes after applying the documented conversion,
  and the claimed image size matches the real (or claimed-grid) size —
  i.e. the model's answer space is fully characterized and invertible.
- The chosen fix (convention pin / self-normalization / server-side scale
  fix) is documented in `ui-misclick-research.md` and the agent-facing
  guidance (KB README + coords_check_loop SKILL.md) so every clicking
  agent applies it consistently.
- Keep the A/B as the regression gate: re-run it after any vision-path
  change (it is cheap, ~2 min, and the ratios are quantitative).

# UI misclick research — vision-grounding coordinate error on qwen-3.8-flash-next

**Date:** 2026-11-03 · **Status: root cause found (SGLang M-RoPE kernel bug), patch 0015 applied to the tree, A/B pending a restart**

## 1. Problem

Computer-use agents (OpenCode + windows-mcp on the win-a…win-f VMs) click wrong
coordinates on screenshots. Reported pattern: the model's estimated click lands
at ~78% of the true position ("gear at (710,62) is actually at (909,79)"),
clicks fall into dead space.

An earlier LLM analysis (the "excerpt") claimed: *the vLLM server resizes every
image above ~1,003,520 px (1280×28×28, the Qwen2.5-VL default) to ~0.78×, so the
model sees a downscaled screenshot and its coordinates land at 78% of the true
position; fix = lower the VM resolution to ≤1280×720 or raise the server's
`--mm-processor-kwargs '{"max_pixels": 2000000}'`.*

**This document records the verification of that claim, its disproof, and the
real root cause.**

## 2. The pipeline, link by link (all measured 2026-11-03)

| # | Link | Verdict | Evidence |
|---|---|---|---|
| 1 | "the vLLM server" | **False** — it's SGLang | `192.168.2.208:19999` (the agents' `baseURL`) is a port-forward from this Windows box to testcomp2's SGLang (`/v1/models` → `owned_by:"sglang"`; `get_server_info` matches the live 32-way profile: memfrac 0.80, max_total_tokens 1048576, ctx 262144) |
| 2 | `--mm-processor-kwargs` applied a 1.0M cap | **False** — the flag doesn't exist in this SGLang build; the equivalent `--mm-process-config` is `{}` (empty) | live `server_args.py` has no `mm_processor_kwargs`; `get_server_info` → `mm_process_config = {}`; SGLang's flag is `--mm-process-config` |
| 3 | `max_pixels ≈ 1,003,520` | **False** — checkpoint bound is **16,777,216** | `preprocessor_config.json`: `size.longest_edge = 16777216`, `shortest_edge = 65536`, patch 16, merge 2 (Qwen3-VL geometry; the excerpt's 28-px numbers are Qwen2.5-VL) |
| 4 | Server resizes 1622×1018 → ~1266×794 | **False — measured 1:1** | token A/B (below) |
| 5 | Tool→agent hop is 1:1 | **True for win-b** | byte-level probe of `http://192.168.250.131:7777/mcp` `Screenshot`: embedded PNG is exactly 1622×1018, no scale note. (The excerpt's claim that the tool's warning is "generic boilerplate" is wrong in general: on the local 7680×3240 virtual desktop the same tool DOES downscale — embedded 1920×810 with a real `Screenshot Coordinate Scale: 4.0` note.) |
| 6 | Client (OpenCode) resizes | **No** | no resize in the source at `b:\ai\opencode` (`ImageSizing` is generation-side only); `aisdk.ts` maps image parts through unchanged; no image-size limit in the provider config |

**Token A/B (the decisive server-side test).** Vision tokens = pixels/1024
(patch 16 × merge 2 → one merged token = 32×32 px). Sent to the live server
(delta = prompt_tokens minus the text-only baseline of the simple probe
question, ~27 tokens):

| image sent | prompt_tokens delta | implied px seen | verdict |
|---|---|---|---|
| 1622×1018 (1.65M px) | 1634 | 1,673,216 ≈ 1632×1024 (the smart-resized dims) | **1:1** — the excerpt's cap would give ~936 |
| 3840×2160 (8.29M px) | 8162 | 8,357,888 | **1:1** |

So the excerpt's entire mechanism is dead: nothing resizes the screenshots, and
its proposed fixes (`--mm-processor-kwargs '{"max_pixels":1700000}'` — rejected
by argparse as an unknown flag; or lowering VM resolution) would at best have
*lowered* the pixel budget from 16.7M to 1.7M.

## 3. Behavioral marker test (what the model actually perceives)

`scripts/bench/grounding_ab.mjs` (zero-dep node script; run from the Windows box
against the live server): draws 3 red circles at known coordinates on a
1622×1018 PNG, asks the model to report the centers and the image size it sees,
temperature 0, thinking off.

| run | state | claimed image size | per-circle ratios (x, y) |
|---|---|---|---|
| 1 (11:32) | pre-patch, idle | **2337×1459** (1.44× too big) | (1.19, 1.58) (2.38, 1.48) (1.31, 1.48) — y systematically ~1.48× |
| 2 (14:10) | pre-patch, box under load | **1754×1113** | (0.51, 1.84) (4.44, 0.58) (1.04, 1.07) — two circles effectively swapped |

The model **misreports the image canvas non-deterministically** and its
localization is badly biased/noisy. (If the excerpt's resize were real, the
model would claim ~1266×794 and land at ~0.78× — never observed.)

Two alternatives were checked and dismissed:
- *Width/height denormalization mixup* (1622/1018 = 1.593 ≈ the 1.48 y-ratio):
  the model's claim is non-deterministic and wrong on BOTH axes — not a
  single-axis scale error.
- *0–1000 convention mismatch* (see §5): the model returned pixel-scale values
  (>1000 on both axes), not a 0–1000 grid.

## 4. Root cause — fused QK RoPE kernel discards M-RoPE height/width

**Bug.** The fused QK GemmaRMSNorm+NeoX-RoPE+gate kernel
(`python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py`) loads
positions as 1-D:

```python
pos = tl.load(positions_ptr + token).to(tl.int64)   # line 107 (pre-fix)
```

Multimodal Qwen3.5-family checkpoints feed **M-RoPE `[3, T]` positions**
(temporal / height / width rows). The kernel reads only the temporal row — for
a single image the temporal row is *constant across all visual tokens*, so
every vision token gets the same h/w rotary angle. **Spatial position is
silently destroyed inside the model.** No error, no warning; text/OCR
unaffected; only grounding collapses.

**Upstream:** issues sgl-project/sglang#35345, #35772, #36762; fix PR **#34446**
"[rotary] Fix the fused Qwen3.5 RoPE kernel discarding mrope height and width",
merged 2026-08-30, sha `e6355774` (3 source files, +30/−5 kernel, +38/−32
mrope.py, +1/−0 qwen3_5.py). Measured upstream:
- #36762 (Qwen3.5-35B-A3B, greedy): center accuracy **26.7% → 81.7%**, offset
  ~95 px → 0.5 px.
- #35772 (Qwen3.8-27B, 20 inputs): **1/20 → 20/20** within half-row; y error
  4.60× → 0.062× (HF/vLLM reference: 0.048×/0.056×).
- Bisection: inputs byte-identical; image-token L2 diff ≈ 0.0401 immediately
  after vision merge, persisting through the first full-attention layer.

**Verified present in the live tree (a34f194c, the running server's code):**
- kernel line 107: 1-D load, docstring `positions: [T]` — unfixed;
- `qwen3_5.py:1204`: `if _is_cuda and self.attn_output_gate:` → always the fused
  path on CUDA, no `positions.ndim` guard;
- `qwen4_exp.py:1490`: `config.attn_output_gate = True` **hardcoded** → this
  model always hits the broken path;
- the three target files are clean in git (no patch touched them).

**Signature match:** the upstream reports describe exactly our symptoms —
systematic scale bias, degraded-but-not-random pointing, and the model
misreporting the image size it sees (the model has no positional read on its own
patch grid).

## 5. Coordinate conventions (lane-2 research, verified citations)

| Model | Output convention | Conversion to original pixels |
|---|---|---|
| Qwen2-VL | normalized [0, 1000) | `x/1000 × W` |
| Qwen2.5-VL | **absolute px in the smart-resized image** | `x × W_orig / W_resized` |
| **Qwen3-VL / 3.5+ (this model)** | **relative, 0–1000 of the original image** | `x/1000 × W` — no resize math needed |

- Qwen3-VL cookbook `2d_grounding.ipynb`: "changed from the absolute coordinates
  used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000".
- Qwen maintainer (Qwen3-VL #1486): "you just need to do `img_x = output_x /
  1000 * img_w`". Alibaba Cloud docs: "[0, 999]"; the mobile-agent cookbook uses
  /999 — the /1000 vs /999 ambiguity is ≤2 px at 1622 px, ignore.
- Official computer-use scaffold prompt: **"The screen's resolution is
  1000×1000."** (the image is NOT stretched; only the output is normalized).
- The SGLang server **never rescales model output coordinates** — no grounding
  helper exists anywhere in `srt/`; conversion is 100% client-side.
- The `smart_resize` in `srt/multimodal/processors/qwen_vl.py` (factor 28) is
  **video-only** — do not use it for image coordinate math (this checkpoint's
  image factor is 32).
- Convention hygiene for this stack: ask the model for the 0–1000 grid (its
  native convention), denormalize in the harness, and never mix conventions
  across turns (store coords in the model's own space).

## 6. Mitigation techniques (lane-1 research, with evidence)

Ranked for this stack; all require no retraining.

> **Constraint correction (2026-11-03, user-confirmed):** the target UI
> exposes NO accessibility tree, so items 1–2 below (UIA `Snapshot` +
> `Click(label)`, SoM via `use_annotation`) are NOT available there —
> element boxes cannot be generated without UIA (OmniParser would be the
> only SoM path, and it stays not-worth-it). The usable mitigation for
> UIA-opaque screens is the **marker-overlay verification loop** —
> implemented as the `coords_check_loop` skill (see §6a).

### 6a. The coords_check_loop skill (built 2026-11-03)

`~/.config/opencode/skills/coords_check_loop/` — before any
`Click(loc)` on a UIA-opaque screen, the agent runs the loop script:

1. on the VM: `CopyFromScreen` + GDI+ draws a hollow blue-dashed ring
   (white outline, crosshair arms) at the requested point — the ring
   center IS the click point; GetPixel self-checks the draw,
2. the marked desktop is sent to the model **via a direct stateless
   `POST /v1/chat/completions` call** (temperature 0, thinking off) and
   the model is asked **what is at the marker center** (description, not
   yes/no — avoids sycophancy),
3. the model answers one JSON line: `confirmed` / `adjust` (corrected
   coords) / `fail` (target not visible); on `adjust` it re-draws,
   max 3 iterations,
4. only `VERIFIED` coords may be clicked; every marked image is saved to
   `%TEMP%\coords_check\` for forensics.

Why the user's design works: the loop lets the model see the DISCREPANCY
between its intended target and where its coordinates actually land —
the systematic-bias error becomes a visible offset the model can walk
back ("the marker is to the right and below where I meant"). Converged
end-to-end on win-b with the UNPATCHED (biased) model: (1300,600) →
(283,27) → (105,27) → VERIFIED on the Visual Studio Save button, 3
iterations; the consolidated skill script re-verified in 2.

Why NOT a forked OpenCode session (the first design, abandoned same day):
prompts sent to a fork via `POST /api/session/:id/prompt` are delivered
to the **PARENT** session as user messages (observed 3×, images included)
— the exact pollution the fork was meant to prevent. The claim that forks
are exempt from the boot resume sweep is also wrong: `projector.ts`
inserts forks with `parent_id: null` (top-level), and the sweep
(`store.ts` `listSuspended`) requires `parent_id IS NULL`, so a crashed
fork IS resumable. The judge task is self-contained (image + target
description), so inherited context adds nothing and a direct call has no
session state at all — no leak path, nothing to resurrect.

Mechanics (all verified): images ride as `image_url` data-URIs in the
OpenAI request; VM base64 round-trips losslessly through the MCP
`PowerShell` tool (400 KB tested); the MCP endpoint needs `initialize`
before `tools/call`; the VM→host push route is dead (host firewall, no
elevation); the portable shell scanner splits quoted args at the first
space and rejects parentheses — pass the target as trailing positional
args after `--`.

1. **UIA element clicking — biggest win, already built.** windows-mcp
   `Snapshot` (UIA tree; `use_annotation=True` default = Set-of-Mark boxes) +
   `Click(label=N)` → clicks the element's UIA center — zero pixels estimated.
   Evidence: OSWorld failure analysis — "among the 550 failed examples, more
   than 75% have mouse click inaccuracies" (arXiv:2404.07972); OSWorld-Human
   puts visual grounding at 23% of errors (MLSys 2026). Keep
   `Screenshot`+`Click(loc)` only for UIA-opaque surfaces (canvas, custom-drawn,
   remote desktop). `WINDOWS_MCP_MERGE_WORD_NODES=1` for Chromium token blowup.
2. **Set-of-Mark / element-id clicking.** Numbered overlays, model answers an
   id: SoM paper (arXiv:2310.11441) — GPT-4V RefCOCOg ACC@0.5 25.7% → 86.4%
   (>3×) with no model change. Already provided by (1) via `use_annotation`.
3. **0–1000 convention pinning** (§5) — for the pixel-fallback path.
4. **Zoom/verify with `region=[l,t,r,b]`** (both `Snapshot` and `Screenshot`
   accept it). R-VLM (arXiv:2507.05673): training-free zoom-only +3.5%
   (UGround), +3.2% (web-agent); verification beats voting (MHVF 66.6% vs
   54.7% sample-and-vote). Rule: the model always answers in **full-screen
   space**; the harness does the crop mapping (copy Anthropic's convention).
   The Qwen team's own `image_zoom_in_tool` does exactly this (V* 85.9→93.7).
5. **Snapping shim** (UIA-opaque only): map the raw click to the nearest
   element center from the last `Snapshot`, max radius ~40–60 px; use box
   centers (`x+w/2`), filter off-screen nodes before snapping.
6. **Anthropic's downscale guidance** — for ANTHROPIC models only (start
   1280×720; "worth more than almost any other optimization"); Anthropic
   explicitly rejects 0–1000 normalized coords. Do not copy to Qwen.
7. **Don't bother:** multi-sample voting (reduces variance, not bias — the
   error here is systematic; GUI-RC +2–3% best case), OmniParser (UIA gives
   exact boxes for free), a co-located 7B grounder (UI-TARS/UGround/OS-Atlas —
   no VRAM headroom at memfrac 0.80).
8. **Related evidence:** Tzafon Northstar-CUA-Fast — scaling the positional
   embedding by 3× improved click accuracy 40%→80% (same mechanism family:
   positional decay); mlx-vlm #1256 — 8-bit quant drifts boxes ~2–4% (keep the
   vision path bf16); vLLM #29595 — a serving bug (not the model) caused "coords
   significantly off" reports.

## 7. Patch 0015 (built, applied, verified CPU-side)

`patches/0015-mrope-fused-rope-fix.patch` — port of #34446 to `a34f194c`
(the only contextual difference from upstream: the `get_exec().deterministic.
rl_on_policy_target` gate in `mrope.py __init__`).

- kernel: `MROPE` constexpr + per-lane axis load (`mrope_axis_map`),
  `positions: [T] or [3, T]` asserts;
- `mrope.py`: `_build_axis_map()` for all three layouts (glm round-robin /
  interleaved scatter / contiguous split); `_legacy_axis_map` exposes the map
  to the older triton/xpu kernels only for glm; Ernie overrides to `None`;
- `qwen3_5.py`: pass `self.rotary_emb.axis_map` when `positions.dim() == 2`;
- tests: upstream `test/registered/rotary/test_mrope_axis_map.py` (CPU) and
  `test/registered/kernels/ops/attention/test_fused_qk_rmsnorm_rope_gate.py`
  (CUDA), CI registration stripped.

Wired into `scripts/apply_patches.sh` (after 0014) and `patches/README.md`.
**Applied to the live tree 2026-11-03 14:08** (uncommitted, `git apply` — the
running server is untouched; the fix activates on the next start). CPU
verification on the box: axis map == `apply_interleaved_rope` exactly for
interleaved `[11,11,10]`/`[24,20,20]`, == section split for contiguous, glm
round-robin pinned, legacy map `None` for qwen, Ernie `None` — all pass.

## 8. The circle test — full description + A/B runbook

### What the test is

**Purpose:** a behavioral, end-to-end check of whether the model perceives
screenshot coordinates correctly — through the whole pipeline (encoder → ViT →
attention → generation), not just pixel transport.

**Method:** synthesize a 1622×1018 PNG (win-b's exact desktop size) filled
light-gray with three red circles of known centers:

```
ground truth centers:  (1000, 400)   (300, 700)   (1400, 900)
```

Send it to the live server with temperature 0, thinking off, and this exact
prompt (the script embeds it):

> The image contains three red circles. Report the pixel coordinates of each
> circle's center as integers (x, y) with (0,0) at the top-left of the image
> and x the horizontal axis. Also state the total width and height of the
> image in pixels as you see them. Answer as JSON:
> `{"image_size": [w, h], "centers": [[x1,y1],[x2,y2],[x3,y3]]}`

The script prints the model's answer, `prompt_tokens`, the claimed image size,
and a per-circle **ratio** = reported / ground-truth, per axis.

**Why these three observables:**

| Observable | Meaning |
|---|---|
| per-circle ratio (x, y) | 1.00 = correct; ~0.78 = the (disproven) downscale theory; systematic bias in one axis = positional corruption |
| claimed `image_size` | the model's self-report of the canvas it sees — the canary: the broken kernel gives vision tokens no positional read, so the model hallucinates the canvas (observed 2337×1459 and 1754×1113 for a 1622×1018 image) |
| `prompt_tokens` | transport check: ≈ **1738** for this prompt = the image passed 1:1 (1634 vision tokens = 1632×1024 grid / 1024); a 1.0M-cap resize would be ~1040; large drift from 1738 = something resized the image in flight |

**Interpretation of the pre-patch baseline (the 1.4× result):**

| run (pre-patch) | claimed size | ratios |
|---|---|---|
| idle box | 2337×1459 (1.44×) | x: 1.19 / 2.38 / 1.31 — y: 1.58 / 1.48 / 1.48 (systematic y scale ~1.48) |
| under load | 1754×1113 | (0.51, 1.84) (4.44, 0.58) (1.04, 1.07) — two circles effectively swapped |

Not random noise (a broken-but-positioned model would be noisy-and-centered);
a systematic, non-deterministic scale error — the M-RoPE signature.

### How to run it

Zero dependencies beyond node (the PNG is hand-encoded in the script). From
the Windows box, any time — including while the box is under load (it's one
request, ~1600 vision tokens):

```bash
node scripts\bench\grounding_ab.mjs
#   arg1 = base URL    (default http://192.168.2.208:19999/v1)
#   arg2 = image size  (default 1622x1018; accepts e.g. 1280x720 to test
#                       other desktops — pass the desktop's real size)
```

The input PNG is written to `grounding_ab_input.png` (cwd) for inspection.
Run it 2–3 times — the pre-patch failure was non-deterministic, so a single
pass is weak evidence; 3 clean passes is the verdict.

### A/B after the restart

```bash
# 1) restart (patch 0015 is already in the tree; no apply step needed)
bash scripts/serve_best_kv_offload.sh        # or the usual profile; ~20 min cold
# 2) after /health returns 200, from the Windows box:
node scripts\bench\grounding_ab.mjs          # x3
```

**Pass (all three):**
1. every circle's ratio within a few % of 1.00 on both axes (upstream
   reference: 81.7% of clicks within-target, 0.5 px offset),
2. claimed `image_size` ≈ 1622×1018 (≈1632×1024 is fine — the smart-resized
   grid),
3. `prompt_tokens` ≈ 1738 (unchanged — proves the transport is still 1:1).

**Fail → revert:** `git -C sglang-official apply -R
patches/0015-mrope-fused-rope-fix.patch` (re-running `apply_patches.sh`
re-applies it). Text-only serving is unaffected either way.

**After the fix:** switch the agents to `Snapshot` + `Click(label=N)` as the
primary path (removes pixel estimation entirely), keep the 0–1000 convention
for the fallback, and use `region=` zoom for small targets.

## 9. Evidence index (remote-exec run dirs on this box)

| run | what |
|---|---|
| `re_20261103-113234-6923` | first pipeline probe (wrong repo path — superseded) |
| `re_20261103-113248-8450` | preprocessor_config.json (16777216), no `--mm-processor-kwargs`, live cmdline |
| `re_20261103-113759-1811` | smart_resize sim; `.208` = SGLang not vLLM; `.208` is this box |
| `re_20261103-114048-2160` | `get_server_info` `.208` vs local: `mm_process_config={}` |
| `re_20261103-114133-1455` | **token A/B: 1622×1018 → 1634 tokens (1:1)** |
| `re_20261103-132242-9826` | **marker test #1: claims 2337×1459, y ~1.48×** |
| `re_20261103-133917-2452` | live-tree bug fingerprint (kernel 1-D load, no guard, gate hardcoded) |
| `re_20261103-135312-2521` | live-tree file sections for patch adaptation |
| `re_20261103-140828-9695` | patch 0015 applied to live tree (check+apply+py_compile OK) |
| `re_20261103-140902-1874` | CPU axis-map verification (all pass) |
| `mcp_shot.png` / `winb_shot.png` | byte-level tool probes (1920×810 @4.0 local / 1622×1018 win-b) |

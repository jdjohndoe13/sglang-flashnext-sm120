# Agent Notes — sglang-flashnext-sm120

Deployment kit for Qwen3.8-Flash-Next-NVFP4 on 8× RTX 5090 (sm_120) via SGLang.
Start with `README.md` (ops table + verified profiles) and `docs/STATUS.md` (ops guide).

## Open Thread — hallucination investigation (RESOLVED: root cause found, fix proven, A/B PASSED)

`hallucination-investigation.md` §14: root cause = the QSA indexer state
gap (`qsa_compressed_k_buffer_pool` lives outside `full_kv_pool`, never
mirrored to the host tier → restored prefixes re-used another request's
indexer keys → wrong attention blocks → confabulation). Fences
0012/0013 were necessary but not sufficient (§13). Fixed by **patch
0014** (`0014-hicache-qsa-indexer-state.patch`, port of upstream PR
#39893; applied to the live tree via `git apply`, in `apply_patches.sh`).
**§15: the A/B rerun PASSED** (2026-11-02, same 60-min/8-lane bench as
the failing §13 run): boot gate `pools=KV + MAMBA + INDEXER` on all 8
ranks, tier exercised (144.8M tokens restored from host), 2963 reqs,
0 cross-session leaks / 0 unknown-tool calls / 0 server-side unknown-tool
lines (vs 25 by min 58 on the broken run); all 74 NOVEL_CONTENT hits
forensically triaged as false positives (canonical OS-path probes +
benign invented-username confabulations, all born in the lane's own
reasoning). **`serve_best_kv_offload.sh` (HICACHE=1) is the daily
profile again.** Watcher (`scripts/bench/watch_server_leaks.ps1`)
stays armed for a few days; patch 0014 is still uncommitted in the
engine tree and upstream PR #39893 is still open — rebase if it merges
with different mechanics.

## Open Thread — UI misclick / grounding (RESOLVED 2026-11-03: 0015 verified + 1.44× = convention artifact, native 0–1000 grid is exact)

`docs/ui-misclick-research.md`: agents misclicking screenshots is NOT an
image-resize problem (pipeline proven 1:1 end-to-end — the "vLLM 1.0M
max_pixels" theory was wrong; the server is SGLang, its cap is 16.7M, and
`--mm-processor-kwargs` doesn't exist). Root cause = the fused QK-RoPE kernel
reading M-RoPE `[3,T]` positions as 1D → vision tokens lost height/width
positioning (upstream #35345/#35772, fixed by #34446). Verified in the live
tree; **patch 0015** (`0015-mrope-fused-rope-fix.patch`, in `apply_patches.sh`)
is applied to the tree (uncommitted) and CPU-verified — activates on the next
server restart. A/B: `node scripts\bench\grounding_ab.mjs` after restart
(pass = marker centers within a few % + claimed image size ≈ 1622×1018).
**A/B RAN 2026-11-03 21:05 (see §8 "A/B result"): the mrope fix WORKS** —
the pre-patch scatter/circle-swaps (x-ratios 1.19–2.38, non-deterministic
size claims) are gone; all 3 circles localize at an identical ratio (±0.1%)
in correct order, stable across runs. **But a uniform ~1.44× scale remains**
(model reports in a canvas 1.44× too big AND claims the image is that size —
self-consistent; normalize by the claimed size and it's exact). That bias
predates 0015 (pre-patch run 1 claimed the same 2337×1459) and is a separate
issue — **do NOT revert 0015**.
**RESOLVED 2026-11-03 (`docs/ui-misclick-research.md` §10): the residual
1.44× is a CONVENTION artifact, not a perception bug.** Asked in the
model's native convention (relative 0–1000 of the image), grounding is
EXACT (≤0.1 px error at 1622×1018 and 1920×1080 — `node
scripts\bench\grounding_conv.mjs`, PASS). Forced to absolute px, the model
translates through a fabricated ~2340×1400 canvas (size-independent anchor;
blank-image size-only claim is 1000×667 — the convention leaking out).
**Fix = client-side, zero server change: always ask for 0–1000 relative
coords, denormalize `x_click = x_rel/1000 × W_real`.** Guidance pinned in
KB README + coords_check_loop SKILL.md; regression gates:
`grounding_conv.mjs` (native, expect PASS) + `grounding_ab.mjs` (absolute,
expect the uniform bias — detects vision-path changes). Gotcha: the
model's enumeration order is NOT stable (x-sorted vs input order) — match
by nearest expectation, not by index.

**0015 had a launch bug (FOUND + FIXED 2026-11-03):** the first boot after
applying it crashed all 8 ranks at CUDA-graph warmup with
`TypeError: dynamic_func() missing 1 required positional argument: 'MROPE'`
(`serve.log`, 19:33→19:36) — the patch added `MROPE: tl.constexpr` to the
kernel signature but the launcher never passed `MROPE=mrope` ("CPU-verified"
never launched the kernel). Fixed by inserting `MROPE=mrope,` between
`HAS_GATE` and `ENABLE_PDL` in `fused_qk_rmsnorm_rope_gate.py` (live tree,
py_compile OK) and the same line added to the patch file (hunk
`@@ -178,24 +200,27 @@`; `git apply --check` passes on a pristine tree).
Any tree freshly patched with 0015 must include this line or it crashes the
same way on first boot.

Post-fix: agents should prefer `Snapshot` + `Click(label=N)` (UIA element
clicking, no pixel estimation) over `Screenshot` + `Click(loc)`. For
UIA-opaque targets (canvas/custom-drawn/remote-desktop, no accessible
elements) the `coords_check_loop` skill (user skills dir, mirrored to
`b:\ai\setup-remote-environment\.config\opencode\skills\`) verifies a
screenshot-measured point before clicking: the script draws a marker
(pixel-exact, GetPixel self-check) on a screenshot YOU provide as a file
— no MCP screen capture, no model endpoint, no session — and the calling
agent is the judge (v3: v1's direct model call hardcoded the endpoint;
v2's forked session leaked prompts to the parent, verified 3×, and forks
are top-level rows the resume sweep can resurrect). Only `confirmed`
coords get clicked. Fully tested 2026-11-03: converged (1300,600)→
(100,28) on the VS Save button in 2 iterations with the unpatched
(biased) model — the loop compensates for the coordinate-system bias.

## Cloned Dependency Source

Read-only dependency source repositories are available under
`.slim/clonedeps/repos/` for inspection. Do not edit these clones.

- `.slim/clonedeps/repos/sgl-project__sglang/` - `sgl-project/sglang` at
  `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`qwen4-main-squashed`, the exact
  branch the deployment venv runs); source of truth for HiCache/kv-cache routing
  (`mem_cache/registry.py`, `unified_radix_cache.py`,
  `hybrid_cache/hybrid_pool_assembler.py`, `pool_host/`, `server_args.py`).

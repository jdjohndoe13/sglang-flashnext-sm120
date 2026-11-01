# Agent Notes — sglang-flashnext-sm120

Deployment kit for Qwen3.8-Flash-Next-NVFP4 on 8× RTX 5090 (sm_120) via SGLang.
Start with `README.md` (ops table + verified profiles) and `docs/STATUS.md` (ops guide).

## Open Thread — hallucination investigation (REOPENED 2026-11-01, see §13)

`hallucination-investigation.md` §7: root cause found (HiCache host-tier
mamba-state incoherence); HICACHE=0 boot (`serve_best_no_kv_offload.sh`)
is the confirmed-good daily profile. **The HICACHE=1 + fences A/B was
RUN on 2026-11-01 (19:09–20:18 UTC, 60-min bench + real user sessions):
the fences (0012/0013) did NOT prevent the hallucination class.**
First contaminated response 19:48:09 (content born in the response,
absent from all 185,759 archived requests); 25 server-side
unknown-tool lines by 20:15 (the whole 3.5-day HICACHE=0 boot had 29);
the user's sessions emitted undeclared tools (`exec`,
`mcp__local__*`, `wsh`) during the same window. The bench's canary
detector reported 0/2536 — false negative; fixed with the
NOVEL_CONTENT detector (73052fa). **HICACHE=0 remains the ONLY safe
profile.** Evidence package (§13) ready for upstream report.

## Cloned Dependency Source

Read-only dependency source repositories are available under
`.slim/clonedeps/repos/` for inspection. Do not edit these clones.

- `.slim/clonedeps/repos/sgl-project__sglang/` - `sgl-project/sglang` at
  `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`qwen4-main-squashed`, the exact
  branch the deployment venv runs); source of truth for HiCache/kv-cache routing
  (`mem_cache/registry.py`, `unified_radix_cache.py`,
  `hybrid_cache/hybrid_pool_assembler.py`, `pool_host/`, `server_args.py`).

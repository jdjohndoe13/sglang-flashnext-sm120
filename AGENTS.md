# Agent Notes — sglang-flashnext-sm120

Deployment kit for Qwen3.8-Flash-Next-NVFP4 on 8× RTX 5090 (sm_120) via SGLang.
Start with `README.md` (ops table + verified profiles) and `docs/STATUS.md` (ops guide).

## Open Thread — hallucination investigation (CLOSED 2026-11-01)

`hallucination-investigation.md` §7: root cause found (HiCache host-tier
mamba-state incoherence); HICACHE=0 boot (`serve_best_no_kv_offload.sh`)
is the confirmed-good daily profile. **The 6–8-parallel-heavy-agent load
case was run (2026-11-01, peak 32 concurrent reqs / 2025 reqs/h) — no
phantom-folder permission requests reappeared.** Standing question is
retired; no leakage class has occurred under any load on the HICACHE=0
boot. Upstream fence patches 0012/0013 (PRs #36738/#36743) are applied
to the tree — the deferred HICACHE=1 A/B is the only remaining step
(§10). For the A/B use the bench:
`node scripts/bench/agent_hallucination_bench.mjs --lanes 8 --duration 60`
(§12).

## Cloned Dependency Source

Read-only dependency source repositories are available under
`.slim/clonedeps/repos/` for inspection. Do not edit these clones.

- `.slim/clonedeps/repos/sgl-project__sglang/` - `sgl-project/sglang` at
  `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`qwen4-main-squashed`, the exact
  branch the deployment venv runs); source of truth for HiCache/kv-cache routing
  (`mem_cache/registry.py`, `unified_radix_cache.py`,
  `hybrid_cache/hybrid_pool_assembler.py`, `pool_host/`, `server_args.py`).

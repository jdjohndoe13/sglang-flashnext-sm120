# Agent Notes — sglang-flashnext-sm120

Deployment kit for Qwen3.8-Flash-Next-NVFP4 on 8× RTX 5090 (sm_120) via SGLang.
Start with `README.md` (ops table + verified profiles) and `docs/STATUS.md` (ops guide).

## Open Thread — hallucination investigation (ROOT CAUSE FOUND + FIX PATCHED, A/B pending)

`hallucination-investigation.md` §7: root cause found (HiCache host-tier
mamba-state incoherence); HICACHE=0 boot (`serve_best_no_kv_offload.sh`)
is the confirmed-good daily profile. The HICACHE=1 + fences A/B (2026-11-01)
was NEGATIVE — the fences (0012/0013) did NOT prevent the class (§13).
**§14: the real gap is the QSA indexer state** — `qsa_compressed_k_buffer_pool`
lives outside `full_kv_pool` and was never mirrored to the host tier, so
restored prefixes re-used another request's indexer keys → wrong attention
blocks → confabulation. Ported upstream PR #39893 as **patch 0014**
(`0014-hicache-qsa-indexer-state.patch`; applied to the live tree, unit-
validated on-hardware: bit-exact round-trip + fence, negative control
reproduces the race). **Next step: the A/B rerun** — HICACHE=1 boot with
0014 (log must show `pools=KV + MAMBA + INDEXER`), 60-min bench
(`node scripts/bench/agent_hallucination_bench.mjs --lanes 8 --duration 60`)
+ watcher; until it passes, **HICACHE=0 remains the ONLY safe profile**.

## Cloned Dependency Source

Read-only dependency source repositories are available under
`.slim/clonedeps/repos/` for inspection. Do not edit these clones.

- `.slim/clonedeps/repos/sgl-project__sglang/` - `sgl-project/sglang` at
  `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`qwen4-main-squashed`, the exact
  branch the deployment venv runs); source of truth for HiCache/kv-cache routing
  (`mem_cache/registry.py`, `unified_radix_cache.py`,
  `hybrid_cache/hybrid_pool_assembler.py`, `pool_host/`, `server_args.py`).

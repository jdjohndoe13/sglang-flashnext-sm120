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

## Cloned Dependency Source

Read-only dependency source repositories are available under
`.slim/clonedeps/repos/` for inspection. Do not edit these clones.

- `.slim/clonedeps/repos/sgl-project__sglang/` - `sgl-project/sglang` at
  `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`qwen4-main-squashed`, the exact
  branch the deployment venv runs); source of truth for HiCache/kv-cache routing
  (`mem_cache/registry.py`, `unified_radix_cache.py`,
  `hybrid_cache/hybrid_pool_assembler.py`, `pool_host/`, `server_args.py`).

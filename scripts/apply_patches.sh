#!/usr/bin/env bash
# Apply the eight sm120 patches to ./sglang-official (branch qwen4-main-squashed).
# IDEMPOTENT: each patch is skipped if already applied (or conflicting) — safe to re-run
# after every `git pull` of this repo. Editable install => no rebuild needed afterwards.
#
#   bash scripts/apply_patches.sh
#
# ORDER MATTERS: the series was authored 0002 -> 0003 -> 0001b -> 0004 -> 0005 -> 0006 -> 0007
# (0001b's hunks against shared files, e.g. gdn_flashinfer.py / server_args.py, only match
# AFTER 0002/0003 — do not reorder). Same order as README.md "Reproduce".
#
#   0002  FP8-KV tile dequant for QSA sparse prefill       [git apply, uncommitted]
#   0003  fp32 prefill state for sm120 FlashInfer GDN     [git apply, uncommitted]
#   0001b RecoverSSM + WY output-only MTP verify (sm120)   [git apply, uncommitted]
#         (adds --gdn-mtp-cache-mode; skipping it makes serve die on that flag)
#   0004  Triton low-M GEMM (decode projections)          [git am -> commit]
#   0005  W8A16 fp8 weight-only serving of dense stack    [git am -> commit]
#   0006  fp8 HyperConnection mix + lm_head               [git am -> commit]
#   0007  FR-Spec: vocab-parallel-safe token-map head     [git apply, uncommitted]
#         (required for TP>1 + SPEC_TOKEN_MAP; without it the draft lm_head gather
#          indexes the vocab-parallel slice globally -> device-side assert)
#   0008  Multi-alias --served-model-name ("a,b" -> /v1/models lists both)  [git apply]
set -euo pipefail
# Locate the repo root (script normally lives at <repo>/scripts/). If run in an exotic way
# (stdin/`bash -s`), fall back to cwd when it looks like the repo root, else fail loudly.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [[ -d "./patches" && -d "./sglang-official" ]]; then
  ROOT="$PWD"
else
  echo "ERROR: cannot locate repo root — run as 'bash scripts/apply_patches.sh' from the repo, or set ROOT=<repo>." >&2
  exit 1
fi
REPO="${REPO:-$ROOT/sglang-official}"
cd "$REPO"
git rev-parse --is-inside-work-tree >/dev/null

applied=0; skipped=0
apply_one() {  # apply_one <patch-file> <mode: apply|am> [extra git apply args...]
  local p="$ROOT/patches/$1" mode="$2" out; shift 2
  [[ -f "$p" ]] || { echo "ERROR: patch file missing: $p" >&2; exit 1; }
  if out="$(git apply --check "$@" "$p" 2>&1)"; then
    if [[ "$mode" == "am" ]]; then
      git am "$p"
    else
      git apply "$@" "$p"
    fi
    echo "applied: $(basename "$p")"; applied=$((applied+1))
  else
    skipped=$((skipped+1))
    echo "skip: $(basename "$p") — git apply --check said:"
    printf '%s\n' "$out" | head -5 | sed 's/^/    /'
    echo "    (already applied, or the tree is not in the expected order/state —"
    echo "     see the ORDER MATTERS note in this script's header)"
  fi
}

apply_one 0002-fp8-qsa-tile-dequant.patch apply --exclude='test/*'
apply_one 0003-sm120-fp32-prefill-state.patch apply
apply_one 0001b-recoverssm-wy-sm120-PORTED.patch apply
apply_one 0004-sm120-lowm-triton-gemm.patch am
apply_one 0005-sm120-fp8-weight-only.patch am
apply_one 0006-sm120-fp8-hc-lmhead.patch am
apply_one 0007-eagle-frspec-vocab-parallel-head-gather.patch apply
apply_one 0008-served-model-aliases.patch apply

echo "done: $applied applied, $skipped skipped."
if [[ "$skipped" -gt 0 ]]; then
  echo "If a patch was expected to apply but was skipped, sglang-official may have uncommitted"
  echo "changes or a diverged checkout — inspect with: git -C '$REPO' status"
fi

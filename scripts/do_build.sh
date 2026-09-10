#!/usr/bin/env bash
# Build/install sglang (qwen4-main-squashed) into ./sglang-official/.venv on "testcomp2"
# (8x RTX 5090, sm120). sglang is already built on this machine — only needed for a fresh
# checkout or after a dependency bump. After this, apply the sm120 patches once:
#   bash scripts/apply_patches.sh
set -euo pipefail
cd /mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang/sglang-official
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 TORCH_CUDA_ARCH_LIST=12.0
export PATH=/home/user/.cargo/bin:/usr/local/cuda/bin:$PATH
export MAX_JOBS=24 CMAKE_BUILD_PARALLEL_LEVEL=24 CARGO_BUILD_JOBS=24
echo "=== install start $(date) ==="
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  -e python
echo "=== install exit=$? $(date) ==="
python -c "import sglang; print('sglang', sglang.__version__)" 2>&1 | tail -2

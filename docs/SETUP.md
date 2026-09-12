# SETUP.md — fresh-install guide (Ubuntu 26.04)

Rebuild the whole deployment on a freshly installed Ubuntu 26.04 box with 8× RTX 5090:
the **sglang Qwen3.8-Flash-Next server** (host venv, ten sm120 patches, port 1025),
plus the crash-investigation cycle.

Companion docs: `README.md` (what/why + benchmark results), `docs/STATUS.md` (ops +
crash investigation), `docs/PERF_CEILING.md` (analysis + dead-ends),
`patches/README.md` (patch series).

## 0. What this box runs

| service | engine | model | how | port |
|---|---|---|---|---|
| LLM server | sglang `qwen4-main-squashed` + 10 patches | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | host venv, foreground | 1025 |
| monitoring | grafana / prometheus / open-webui / gpu-temp-monitor | — | docker (optional) | various |

## 1. Host prerequisites

Verified versions from the working machine (`testcomp2`): NVIDIA driver **590.48.01**,
CUDA toolkit **13.1.115** (`/usr/local/cuda`), Docker **29.1.3** (optional — only for the
monitoring stack), gcc-13, python 3.12, rust/cargo, `uv`.

```bash
# NVIDIA driver (590+ for Blackwell sm120). On a headless GPU server prefer the
# NVIDIA repo driver, NOT the desktop metapackage. Keep displays disabled.
sudo apt update && sudo apt install -y build-essential curl git
# install the driver, e.g. via the NVIDIA CUDA network repo's driver runfile or:
#   sudo apt install nvidia-driver-590   (if packaged for your release)
nvidia-smi          # must list all 8 RTX 5090s, driver 590.x

# CUDA toolkit 13.1 (compiler only; the driver is already installed):
# download cuda_13.1.*_linux.run and run with --toolkit --silent (deselect the driver)
ls /usr/local/cuda/bin/nvcc      # -> /usr/local/cuda/bin/nvcc

# gcc-13 (Ubuntu 26.04 default gcc is newer; sglang's JIT needs 13):
sudo apt install -y gcc-13 g++-13

# rust/cargo (sglang build dependency):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv (used by scripts/do_build.sh):
curl -LsSf https://astral.sh/uv/install.sh | sh

# python 3.12 (sglang venv):
sudo apt install -y python3.12 python3.12-venv

# OPTIONAL — docker + NVIDIA container toolkit (only for the monitoring stack):
# curl -fsSL https://get.docker.com | sh
# sudo apt install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 2. Directory layout (data disk)

One big nvme mount for everything (`/mnt/data` on the working machine, 3.2 TB; ~500 GB
needed: model ~135 GB + caches + build artifacts).

```bash
sudo mkdir -p /mnt/data /mnt/huggingface
# either mount the data disk at /mnt/data and symlink /mnt/huggingface into it:
sudo ln -s /mnt/data/huggingface /mnt/huggingface
sudo chown -R $USER /mnt/data
mkdir -p /mnt/data/shared/models /mnt/data/huggingface
```

`testcomp2` layout for reference:

| path | what |
|---|---|
| `/mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang` | this repo (scripts/patches/docs + `sglang-official/` checkout + `cache/` + `logs/`) |
| `/mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4` | model checkpoint (~135 GB) |

## 3. Get this repo

```bash
cd /mnt/data/shared/models
# clone your copy of this repo (github/local backup), e.g.:
git clone <your-repo-url> qwen3.8-flash-next-nvfp4-sglang
cd qwen3.8-flash-next-nvfp4-sglang
```

The repo carries: `patches/` (ten sm120 patches), `scripts/` (`serve.sh`,
`serve_best.sh`, `serve_single.sh`, `apply_patches.sh`, `do_build.sh`,
`bench_sglang.py`, `make_hot_tokens.py`, the crash-cycle scripts),
`docs/` (STATUS/PERF_CEILING/SETUP), `hot_tokens_64k.pt` (the FR-Spec draft-vocab map),
`README.md`.

`hot_tokens_64k.pt` comes with the repo. To regenerate from scratch instead:
`python scripts/make_hot_tokens.py <model_dir> <corpus_glob>... -o hot_tokens_64k.pt`
(all first 32K BPE ids + frequent tokens of a local corpus + specials, padded to 65,536).

## 4. Model checkpoint

```bash
# model (~135 GB; the ~50 GB PLE n-gram table is served from host RAM):
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir /mnt/huggingface/RadixArk/Qwen3.8-Flash-Next-NVFP4
# NOTE: hf_xet can stall on the largest shards — the curl fallback that resumes
# reliably is described in docs/STATUS.md.
```

## 5. sglang Qwen3.8-Flash-Next server (host venv, ten patches)

```bash
cd /mnt/data/shared/models/qwen3.8-flash-next-nvfp4-sglang

# 1) pin the sglang checkout everything was verified against
#    (qwen4-main-squashed @ 4ccff141dbe992794f9da6c3aa23535b4f72000d):
git init sglang-official && cd sglang-official
git remote add origin https://github.com/sgl-project/sglang
git fetch --depth 1 origin 4ccff141dbe992794f9da6c3aa23535b4f72000d
git checkout FETCH_HEAD
# (alternative: current branch tip — patches may need adjusting if upstream moved:
#  git clone -b qwen4-main-squashed https://github.com/sgl-project/sglang sglang-official)
cd ..

# 2) build the editable venv (~30-60 min; MAX_JOBS=24 in do_build.sh):
bash scripts/do_build.sh
#    = uv pip install -e python with the cu130 wheel index, gcc-13, CUDA_HOME=/usr/local/cuda
#    -> creates sglang-official/.venv (sglang 0.5.20.dev11, sglang-kernel cu130)

# 3) apply the ten sm120 patches (idempotent; re-run after every git pull):
bash scripts/apply_patches.sh
#    0001b+0002+0003+0007+0008+0009+0011 via git apply; 0004-0006 via git am (commits).

# 4) serve (TP8+EP8 is the only 8x32GB configuration; see README for why):
./scripts/serve_best.sh      # interactive/agent profile: 262144 ctx, fp8 stack, :1025
./scripts/serve_single.sh    # one huge session: 786432 ctx (YaRN x3)
curl -s http://127.0.0.1:1025/health    # -> 200 when ready (~20 min first start:
#           model load + flashinfer autotune + graph capture)
```

Defaults that matter on this hardware (all knobbed in `scripts/serve.sh`):
`TP=8 EP_SIZE=8` (pure TP8 cannot load the NVFP4 checkpoint — see README),
`MEMFRAC=0.80` (0.90 OOMs graph capture on 32 GB cards), `MAXREQ=8` with
`--max-mamba-cache-size` ≥ 6× MAXREQ, `LINEAR_BACKEND=flashinfer` (needs the patches),
`--speculative-token-map hot_tokens_64k.pt` (patches 0007+0011 make TP>1 + the token map
safe), `--image-processor-backend pil` (the fast image processor OOMs the scheduler card),
`.venv/bin` prepended to PATH (sglang's JIT builder needs the venv's ninja).

The launchers refuse to start while GPUs hold >4 GB (anything else occupying the cards
blocks startup — stop it first; `FORCE=1` overrides). Use tmux/screen to survive
disconnects.

## 6. Crash-investigation cycle (optional, triggered from Windows)

Reproduces the (now-fixed) crash on demand, restores the server afterwards:

```
scripts\crash_cycle.bat            (Windows: edit HOST / REMOTE_SCRIPT / SESSION inside)
  -> scripts/crash_cycle_remote.sh (machine: stop containers -> repro -> restart vLLM -> health poll)
  -> scripts/run_crash_reproduction.sh + scripts/repro_crash.sh (the stress loop)
  -> opencode session gets pinged when the cycle completes
```

- Repro exit codes: `0` = N clean attempts, `1` = died during startup, `2` = crash
  reproduced, `3` = vLLM restart timed out.
- The debug phase sets `EAGER_DRAFT=1 NO_OVERLAP=1 DBG_LAUNCH_BLOCKING=1 DBG_CRASH_DUMP=1`
  (eager draft + no overlap + launch blocking + crash dumps); results in `logs/`
  (`repro.log`, `repro_server.log`, `crashdump/`).
- Requires an `opencode` CLI on the Windows box with the target session id
  (`SESSION=` in crash_cycle.bat).

## 7. Validation checklist

```bash
curl -s http://127.0.0.1:1025/health                       # 200
curl -s http://127.0.0.1:1025/v1/models                    # lists qwen-3.8-flash-next
curl -s http://127.0.0.1:1025/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-3.8-flash-next","messages":[{"role":"user","content":"Hi"}]}'   # completes
python scripts/bench_sglang.py                             # throughput sanity (see script args)
```

Also verify the 0011 fix signature when spec decoding is on: with
`SGLANG_DEBUG_TRACK_DUMP=1` the decode-extend dumps must show `logits_w=65536` and
min/max < 65,536; the per-rank draft head must be `(8192, 2560)`. Long agent sessions
were the original crash trigger — soak one before trusting the install.

## 8. Troubleshooting pointers

- First-start OOM / graph capture: lower `MEMFRAC` (0.80 validated), keep
  `CUDAGRAPH_MAXBS` = MAXREQ, headless box assumed.
- fp8 KV + triton GDN/QSA crash: `KVDTYPE=auto` (fp8_e4m3 needs the flashinfer backend).
- FlashInfer GDN sm120 contradictions (fp32 vs bf16 state): all resolved by the patch
  series — keep it applied; see `docs/STATUS.md` and `docs/PERF_CEILING.md`.
- Startup death mentioning captures or `stream is capturing`: debug env vars leaked into
  a normal serve — the repro wrapper sets them, plain `serve.sh` must not.
- `multimem all-gather disabled (CUDA driver error: invalid device ordinal)` in the log:
  normal on 5090s (no NVLink multicast), not an error.

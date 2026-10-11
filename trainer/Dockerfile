# The GPU image for gs-trainer (docs/plans/2026-10-08-gsplat-cloud.md, "Image").
#
# Modal builds it with `Image.from_dockerfile` (gs_trainer/modal_app.py) and caches it; it is rebuilt only when
# this file, pyproject.toml or uv.lock change. Our own code is not baked in: Modal adds `gs_trainer` and
# `freespace.py` to every run, so editing the trainer needs no rebuild. RunPod or Vast could pull the same
# image from a registry.
#
# The devel base is needed for nvcc: gsplat's CUDA kernels are compiled here, once, without a GPU. cuDNN
# comes with the torch wheel (nvidia-cudnn-cu12), so the plain devel tag is enough.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# git: gsplat is a git dependency. libgl1/libegl1/libgomp1/libglib2.0-0/libusb-1.0-0: Open3D and OpenCV load
# them on import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential libgl1 libegl1 libgomp1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv, pinned to the version that wrote uv.lock.
RUN curl -LsSf https://astral.sh/uv/0.12.23/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# Python 3.12 managed by uv, the environment at a fixed path outside the project so `PATH` can point at it.
# Modal runs its functions with the `python` it finds on PATH, i.e. this environment's.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Only the lock and the project file, so the slow layer (torch cu128, Open3D, ~6 GB) is cached until they
# change. On Linux the lock resolves torch and torchvision to the cu128 wheels.
WORKDIR /opt/gs-trainer
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project --no-dev --python 3.12 \
    && uv pip install --python /opt/venv/bin/python pip
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# gsplat is installed as pure Python (BUILD_NO_CUDA=1, see pyproject.toml); its kernels are JIT-compiled by
# torch.utils.cpp_extension into $TORCH_EXTENSIONS_DIR/gsplat_cuda on first use (gsplat/cuda/build.py). That
# compile is NOT done here: on Modal's default builder (few cores) it ran for over 1.5 h. `modal_app.py`
# runs it as a `run_function` image step with 32 cores instead. The arch list must stay set at run time
# too: torch rewrites build.ninja on every load, and different arch flags would make ninja recompile
# everything. 8.9 = L40S/4090, 9.0 = H100 (no A100: a third more compile for a GPU we don't use).
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.9;9.0" \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    MAX_JOBS=32

# LPIPS (gs_trainer.evaluate) needs torchvision's AlexNet weights; fetch them now rather than on every run.
ENV TORCH_HOME=/opt/torch_home
RUN python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)" \
    && ls /opt/torch_home/hub/checkpoints/

# Depth Anything 3 (gs_trainer.da3, the P4 depth prior). Last, so changing it never recompiles gsplat.
# Its own metadata wants numpy<2, xformers, opencv-python, fastapi, e3nn, open3d, ... which inference never
# imports, so DA3 goes in with --no-deps, plus only what `depth_anything_3.api` and the DA3-BASE model
# classes import (checked by walking the import graph at 3d835ec and running DA3-BASE in a venv built
# like this one). `-c pins` holds everything already installed at its version: nothing from uv.lock moves.
# evo also without deps: DA3 imports only `evo.core.trajectory` (numpy, scipy), not pandas/seaborn/rosbags.
# xformers stays out (DINOv2's SwiGLU falls back to plain torch).
RUN uv pip freeze --python /opt/venv/bin/python | grep -v ' @ ' > /tmp/pins.txt \
    && uv pip install --python /opt/venv/bin/python -c /tmp/pins.txt \
        einops==0.8.2 huggingface-hub==2.1.1 safetensors==0.8.0 omegaconf==2.3.1 addict==2.4.0 \
        moviepy==1.0.3 matplotlib==3.11.2 pycolmap==4.2.1 trimesh==5.1.1 \
    && uv pip install --python /opt/venv/bin/python --no-deps evo==1.37.1 \
        "git+https://github.com/ByteDance-Seed/Depth-Anything-3@3d835ec" \
    && rm /tmp/pins.txt

# The Apache-2.0 DA3-BASE weights (~0.5 GB) in a fixed cache, so `depth_prior` downloads nothing. Fetched
# with huggingface_hub, not by loading the model: `depth_anything_3.api` imports gsplat, and importing gsplat
# on a builder with nvcc but no compiled kernels starts the (slow) JIT compile right here.
ENV HF_HOME=/opt/hf
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-BASE')" \
    && ls /opt/hf/hub/models--depth-anything--DA3-BASE/snapshots/*/model.safetensors

WORKDIR /root

# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS python-deps

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ninja-build \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.12.5
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ARG MAX_JOBS=10

ENV VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:/opt/build-tools/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_LINK_MODE=copy \
    MAX_JOBS=${MAX_JOBS}

WORKDIR /app/model_repo

RUN python3 -m venv /opt/build-tools && \
    python3 -m venv ${VIRTUAL_ENV}

# Dependency installation is cached independently from model and runner source changes.
COPY --link pyproject.toml uv.lock ./

ARG INSTALL_REQUIREMENTS=1
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then \
      /opt/build-tools/bin/pip install "uv==${UV_VERSION}" && \
      /opt/build-tools/bin/uv sync \
        --active \
        --frozen \
        --no-dev \
        --no-install-project; \
    fi

# The PyPI wheel defers gsplat compilation until first use. Rebuild the locked
# version from source so the runtime image needs neither nvcc nor build tools.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then \
      gsplat_version="$(python -c 'from importlib.metadata import version; print(version("gsplat"))')" && \
      pip install \
        --force-reinstall \
        --no-build-isolation \
        --no-deps \
        --no-binary=gsplat \
        "gsplat==${gsplat_version}" && \
      python -c "import gsplat.csrc; print(gsplat.csrc.__file__)"; \
    fi

# Verify the compiled fat binary, including the forward-compatible PTX target.
RUN if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then \
      extension_path="$(python -c 'import gsplat.csrc; print(gsplat.csrc.__file__)')" && \
      elf_targets="$(cuobjdump --list-elf "${extension_path}")" && \
      ptx_targets="$(cuobjdump --list-ptx "${extension_path}")" && \
      printf '%s\n%s\n' "${elf_targets}" "${ptx_targets}" && \
      for architecture in 75 80 86 89; do \
        echo "${elf_targets}" | grep -q "sm_${architecture}"; \
      done && \
      echo "${ptx_targets}" | grep -q "sm_89"; \
    fi

# PyTorch's locked cu128 wheels provide CUDA, cuDNN, and NCCL user-space
# libraries. The CUDA base image supplies the NVIDIA container integration
# without duplicating those libraries in the final image.
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      python3 && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"

ENV VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:${PATH} \
    PYTHONPATH=/app/model_repo:/app/model_repo/Depth-Anything-V2:/app \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNNER_PORT=58090 \
    RUNNER_NAME=panodreamer-perspective \
    RUNNER_TYPE=generator \
    RUNNER_VERSION=0.1.0 \
    RUNNER_CONTRACT_VERSION=1 \
    RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.adapters.perspective:run_job \
    PANODREAMER_AUTO_DOWNLOAD_WEIGHTS=1

WORKDIR /app/model_repo

COPY --link --from=python-deps /opt/runner /opt/runner

# Source is deliberately last so normal code changes reuse the dependency and
# CUDA extension layers.
COPY --link . .

RUN test -f Depth-Anything-V2/depth_anything_v2/dpt.py && \
    test -f 3d-moments/core/inpainter.py && \
    python -B -c "import torch; import gsplat.csrc; from depth_anything_v2.dpt import DepthAnythingV2; from runner_wrapper.adapters.perspective import run_job as perspective_job; from runner_wrapper.adapters.panorama import run_job as panorama_job; print(torch.__version__, gsplat.csrc.__file__, perspective_job.__name__, panorama_job.__name__)"

EXPOSE 58090

CMD ["python", "-m", "runner_wrapper.server"]

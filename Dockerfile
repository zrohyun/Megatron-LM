# syntax=docker/dockerfile:1
#
# Megatron-LM + FlashAttention-3 + TransformerEngine + APEX + DeepEP
# CUDA 12.9 (NGC 25.06) base. ASCII-only comments to avoid encoding issues.

FROM nvcr.io/nvidia/pytorch:25.06-py3 AS base
#FROM nvcr.io/nvidia/pytorch:25.04-py3 AS base

ENV SHELL=/bin/bash \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ------- System packages -------
RUN rm -rf /opt/megatron-lm || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo openssh-server htop bwm-ng gdb sysstat sshpass pstack git zsh tmux curl wget ca-certificates \
        build-essential cmake pkg-config python3-dev libssl-dev ranger cron tree zsh python3-venv psmisc \
        libfabric-dev libaio-dev libnuma-dev gettext && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq && \
    rm -rf /var/lib/apt/lists/*

# ------- Python baseline & extras -------
    RUN python -m pip install --upgrade pip && \
    unset PIP_CONSTRAINT && pip install \
    debugpy dm-tree torch_tb_profiler einops wandb \
    megatron-energon==7.2.* \
    sentencepiece tokenizers transformers==4.51.* torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black isort flake8 pylint coverage mypy \
    setuptools \
    tensorboard pybind11

# ------- grouped_gemm (optional) -------
RUN TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

RUN unset PIP_CONSTRAINT && pip install flashinfer-python

# NCCL 최신화가 필요하면 이 부분을 주석 해제하고 빌드할 것
#COPY ./nccl/nccl-local-repo-ubuntu2404-2.28.3-cuda12.9_1.0-1_amd64.deb /root/
#RUN dpkg -i /root/nccl-local-repo-ubuntu2404-2.28.3-cuda12.9_1.0-1_amd64.deb
#RUN cp /var/nccl-local-repo-ubuntu2404-2.28.3-cuda12.9/nccl-local-545D4944-keyring.gpg /usr/share/keyrings/
#RUN apt-get update && apt install libnccl2 libnccl-dev && rm -rf /var/lib/apt/lists/*

# CUDNN 최신화
# https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.13.0.50_cuda12-archive.tar.xz
COPY ./cudnn-linux-x86_64-9.13.0.50_cuda12-archive/include/ /usr/include/
COPY ./cudnn-linux-x86_64-9.13.0.50_cuda12-archive/include/ /usr/include/x86_64-linux-gnu/
COPY ./cudnn-linux-x86_64-9.13.0.50_cuda12-archive/lib/ /usr/local/cuda/targets/x86_64-linux/lib/

# ------- Project: Megatron-LM (editable) -------
ENV MEGATRON_HOME=/opt/megatron-lm
WORKDIR $MEGATRON_HOME
# Expect pyproject.toml and setup.py in build context
COPY ./Megatron-LM $MEGATRON_HOME
RUN unset PIP_CONSTRAINT && pip install --no-build-isolation -e ".[dev]"

# ------- FlashAttention-3 (install first) -------
WORKDIR /tmp
ARG FLASH_ATTENTION_COMMIT=3ba6f82
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout ${FLASH_ATTENTION_COMMIT} && git submodule update --init && \
    cd hopper && python setup.py install && \
    PYTHON_SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p "$PYTHON_SITE/flash_attn_3" && \
    cp flash_attn_interface.py "$PYTHON_SITE/flash_attn_3/flash_attn_interface.py"

# ------- TransformerEngine 2.8 (commit id a9f2655; rebuild after FA-3) -------
RUN pip uninstall -y transformer-engine-cu12 transformer-engine-torch transformer-engine || true && \ 
    pip install -U pybind11 && unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="80;90;100" NVTE_FRAMEWORK=pytorch \
      pip install --no-build-isolation --no-cache-dir \
      transformer-engine[pytorch]==2.8.0
      #git+https://github.com/dalgarak/TransformerEngine.git@fa3_mla_cp_te2.7
      #git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.8


# ------- APEX (all useful extensions enabled) -------
# NCCL 관련, 기본 빼고는 안정상으로 일단 다 뺌.
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_XENTROPY=1 APEX_NCCL_P2P=1 \
    APEX_DISTRIBUTED_ADAM=1 APEX_NCCL_ALLOCATOR=1 \
    APEX_FAST_LAYER_NORM=1 APEX_FAST_MULTIHEAD_ATTN=1 \
    APEX_FUSED_CONV_BIAS_RELU=0 \
    pip install -v --no-build-isolation .

# ------- DeepEP + NVSHMEM -------
# IBGDA dependency symlink
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so || true

# Best-effort python wrapper (optional)
RUN pip install --no-cache-dir nvidia-nvshmem-cu12 || true

# deepEP의 경우 a84a248 commit은 2025.04 월 경이다. 이후 수정이 많이 일어났으니 checkout 위치를 달리해줘야 함.
WORKDIR /home/dpsk_a2a
RUN git clone -b v2.5.1 https://github.com/NVIDIA/gdrcopy.git && \
    git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout v1.2.1 && cd -

# Download and prepare nvshmem source (may require auth in some environments)
RUN wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz -O nvshmem_src.tar.xz && \ 
    tar -xvf nvshmem_src.tar.xz && mv libnvshmem-linux-x86_64-3.3.9_cuda12-archive deepep-nvshmem

# Define core paths explicitly (avoid referencing undefined vars)
ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=/usr/local/mpi/include
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:/usr/local/mpi/lib:/usr/local/x86_64-linux-gnu:/usr/local/cuda/lib64
ENV GDRCOPY_HOME=/home/dpsk_a2a/gdrcopy

# Build nvshmem (Hopper, sm_90) with IBGDA
WORKDIR /home/dpsk_a2a/deepep-nvshmem

# Build DeepEP with temporary CPLUS_INCLUDE_DIR extension
# 주의: A100은 지원 안함. Hopper 이상 부터 지원하게 설정. compute_90, compute_100
WORKDIR /home/dpsk_a2a/DeepEP
RUN bash -lc 'SAVED_CPLUS_INCLUDE_DIR="${CPLUS_INCLUDE_DIR:-}"; \ 
    export CPLUS_INCLUDE_DIR="${CPLUS_INCLUDE_DIR:-}:/usr/include:/usr/include/x86_64-linux-gnu/"; \
    NVSHMEM_DIR=$NVSHMEM_DIR TORCH_CUDA_ARCH_LIST="9.0 10.0" python setup.py develop && \
    NVSHMEM_DIR=$NVSHMEM_DIR TORCH_CUDA_ARCH_LIST="9.0 10.0" python setup.py install; \
    export CPLUS_INCLUDE_DIR="$SAVED_CPLUS_INCLUDE_DIR"'

# To Huggingface Model Compatible
# triton의 경우, Attribute Error: type object 'CompiledKernel' has no attribute 'launch_enter_hook' 오류 때문.
# pytorch-triton과 버전이 서로 달라서 그렇다. pytorch-triton==3.3.0-* 임.
RUN pip install --no-cache-dir nvidia-modelopt[hf] nvidia-modelopt-core --extra-index-url https://pypi.nvidia.com || true
# 여전히 Huggingface Transformers 쪽 버전 차이 경고가 있음. transformers==4.51.3 이 필요할 수 있음
RUN pip install --no-cache-dir triton==3.2.0 || true

RUN addgroup --gid 500 irteam
RUN useradd -m -d /home/irteam -s /bin/bash -g 500 -u 500 irteam

# ------- Final workspace -------
# 중요! 고성능 스토리지를 쓰기 위해서는 사용자 권한을 500:500으로 줘야한다고 함
USER 500:500
WORKDIR /home

# Notes:
# - For FP8 blockwise (--fp8-recipe "blockwise"), CUDA 12.9 is required; this base satisfies it.
# - FlashAttention-3 fused backward requires: qk_nope_head_dim + qk_rope_head_dim == v_head_dim.
#   Adjust model dims accordingly (e.g., 96 + 32 = 128) to enable fused/flash backward.
# - If NCCL Watchdog Timeout occurs, try: --distributed-timeout-minutes 60.
 
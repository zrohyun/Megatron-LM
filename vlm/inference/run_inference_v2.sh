#!/bin/bash
# Rice-GPT VLM Inference v2  
# Based on v1 inference code with ForwardStep pattern
# Usage: ./run_inference_v2.sh [OPTIONS]

set -e

# ========== 기본 설정 ==========
NUM_GPUS=1
NNODES=1
SCRIPT="vlm/inference/inference_v2.py"

TRAINING_PATH="/mnt/vlm-data/personal/zro/workspace/Megatron-LM-gh-zro"
MEGATRON_PATH="${TRAINING_PATH}/megatron"

TM=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_PATH="${TRAINING_PATH}/.logs"
mkdir -vp "${LOG_PATH}"

export MEGATRON_PATH="${MEGATRON_PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 토크나이저 및 체크포인트 경로
TOKENIZER_PATH="${TRAINING_PATH}/tokenizers/wbl_mm_tokenizer_v5"
CHECKPOINT_BASE_DIR="/mnt/checkpoint/ncai/multimodal/outputs"
# CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/vlm_stage_1/checkpoints/V7_rice_gpt_stage_1_7b_a1b_gbs_16_no_moe_loss_llava_558k_tk_v5_vit_rotary_emb_win_size__vlm_20251217__wbl_llm_sft_1206_lr_3e6_seq_16384"
# CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/vlm_stage_2/checkpoints/v9_rice_gpt_stage_2_7b_a1b_gbs_128_lr_5e6_img_size_30_s_factor_1_0_no_capacity___v6"
CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/vlm_stage_3/checkpoints/v5_rice_gpt_stage_3_7b_a1b_gbs_128_lr_3e6_warmup_0_02_finevision_4M_tk6_no_capacity__v10__v6"

IMAGE_PATH="${TRAINING_PATH}/test_images/dt.jpg"
USER_PROMPT="Describe the image."
NUM_TOKENS=128
SEQ_LENGTH=4096
DECODER_SEQ_LENGTH=4096
TEMPERATURE=0.1
TOP_K=1

echo "=========================================="
echo "Rice-GPT VLM Inference v2"
echo "=========================================="
echo "TRAINING_PATH: ${TRAINING_PATH}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "IMAGE_PATH: ${IMAGE_PATH}"
echo "USER_PROMPT: ${USER_PROMPT}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# ==========  ==========
cd "${TRAINING_PATH}"

MODEL_ARGS=(
  --model-name "rice-gpt-7b-a1b"
  --disable-bias-linear
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings 32768
  --num-layers 24
  --position-embedding-type none
  --hidden-size 1536
  --ffn-hidden-size 5760
  --num-attention-heads 12
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --swiglu
  --untie-embeddings-and-output-weights
  --no-masked-softmax-fusion
  --normalization RMSNorm
)

MOE_ARGS=(
  --num-experts 64
  --moe-layer-freq "([0]*1+[1]*23)"
  --moe-ffn-hidden-size 960
  --moe-shared-expert-intermediate-size 960
  --moe-router-load-balancing-type global_aux_loss
  --moe-router-topk 5
  --moe-router-topk-scaling-factor 1.0
  --moe-router-score-function sigmoid
  --moe-token-dispatcher-type alltoall
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-grouped-gemm
  --moe-aux-loss-coeff 2e-2
  --moe-z-loss-coeff 1e-3
  --overlap-grad-reduce
)

MLA_ARGS=(
  --multi-latent-attention
  --q-lora-rank 768
  --kv-lora-rank 512
  --qk-head-dim 128
  --qk-pos-emb-head-dim 64
  --v-head-dim 128
  --rotary-scaling-factor 1.0
  --rope-type rope
  --apply-layernorm-1p
  --rotary-base 10000
  --rotary-base-global 1000000
  --qk-layernorm
)

TOKENIZER_ARGS=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_PATH}"
)

PRECISION_ARGS=(
  --bf16
  --use-cpu-initialization
)

LOAD_ARGS=(
  --no-load-optim
  --no-load-rng
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --distributed-backend nccl
  --attention-backend flash
)

GENERATION_ARGS=(
  --decoder-seq-length ${DECODER_SEQ_LENGTH}
  --num-tokens-to-generate ${NUM_TOKENS}
  --temperature ${TEMPERATURE}
  --top-k ${TOP_K}
)

INPUT_ARGS=(
  --image-path "${IMAGE_PATH}"
  --user-prompt "${USER_PROMPT}"
)

CHECKPOINT_ARGS=(
  --load "${CHECKPOINT_PATH}"
  # --ckpt-step 2000
)

INFERENCE_ARGS=(
  --no-initialization
  --inference-max-batch-size 4
  --inference-max-seq-length ${SEQ_LENGTH}
  --inference-max-requests 1
  --eval-iters 1
)

EXTRA_ARGS=(
  --train-iters 1
  --save-interval 10000
  --split "100,0,0"
)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72


PYTHONPATH="${TRAINING_PATH}:${PYTHONPATH}" \
python ${SCRIPT} \
  "${MODEL_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MLA_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${GENERATION_ARGS[@]}" \
  "${INPUT_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${INFERENCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "${LOG_PATH}/inference_${TM}.log"

echo "=========================================="
echo "Inference completed!"
echo "=========================================="
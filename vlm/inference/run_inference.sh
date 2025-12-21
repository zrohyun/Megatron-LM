#!/bin/bash

set -e

# ========== 기본 설정 ==========
NUM_GPUS=1
NNODES=1
SCRIPT="vlm/inference/inference.py"

TRAINING_PATH="${TRAINING_PATH:-/workspace/Megatron-LM}"
MEGATRON_PATH="${MEGATRON_PATH:-${TRAINING_PATH%/}Megatron-LM/megatron}"


# ========== 실행 ==========
PYTHONPATH="$TRAINING_PATH:$MEGATRON_PATH:$PYTHONPATH" \
  python3 -m torch.distributed.run \
  --standalone \
  --nnodes=${NNODES} \
  --nproc_per_node=${NUM_GPUS} \
  ${SCRIPT} \
  \
  --disable-bias-linear \
  --seq-length 4096 \
  --max-position-embeddings 32768 \
  --num-layers 24 \
  --position-embedding-type none \
  --hidden-size 1536 \
  --ffn-hidden-size 5760 \
  --num-attention-heads 12 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --no-masked-softmax-fusion \
  \
  --num-experts 64 \
  --moe-layer-freq "([0]*1+[1]*23)" \
  --moe-ffn-hidden-size 960 \
  --moe-shared-expert-intermediate-size 960 \
  --moe-router-topk 5 \
  --moe-router-topk-scaling-factor 1.0 \
  --moe-router-score-function sigmoid \
  --moe-token-dispatcher-type alltoall \
  --moe-router-dtype fp32 \
  --moe-permute-fusion \
  \
  --multi-latent-attention \
  --q-lora-rank 768 \
  --kv-lora-rank 512 \
  --qk-head-dim 128 \
  --qk-pos-emb-head-dim 64 \
  --v-head-dim 128 \
  --rotary-scaling-factor 1.0 \
  --normalization RMSNorm \
  --rope-type rope \
  --apply-layernorm-1p \
  --rotary-base 10000 \
  --rotary-base-global 1000000 \
  --qk-layernorm \
  \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model ${TRAINING_PATH}/tokenizers/wbl_mm_tokenizer_v3 \
  \
  --bf16 \
  \
  --no-load-optim \
  --no-load-rng \
  \
  --expert-model-parallel-size 1 \
  \
  --inference-max-requests 1 \
  --num-tokens-to-generate 100 \
  --ckpt-format torch_dist \
  --decoder-seq-length 16384 \
  --inference-max-seq-length 16384 \
  \
  --load /workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_2_7b_a1b_251210
  # --prompts \
  # "Hey, are you conscious? Can you talk to me?"
 
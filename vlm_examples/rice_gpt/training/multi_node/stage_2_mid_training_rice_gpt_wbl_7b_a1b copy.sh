#!/bin/bash

TRAINING_PATH="${TRAINING_PATH:-/workspace/Megatron-LM}"
MEGATRON_PATH="${MEGATRON_PATH:-${TRAINING_PATH}/megatron}"

# The script needs to be run on at least 1 nodes.

SINGLE_NODE=0

# --- Multi-node configuration ---
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=10.10.90.6       # IP    h100-03
export MASTER_PORT=23456
export NNODES=4
export GPUS_PER_NODE=8
export NODE_RANK=${NODE_RANK:-0}                 # ★ 마스터는 반드시 0

# 소켓/랜데부는 이더넷
export NCCL_SOCKET_IFNAME=ibs2 # eno1
export GLOO_SOCKET_IFNAME=ibs2 # eno1 

# RDMA(IB) 데이터 경로 (보유 HCA에 맞게)
export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA='^mlx5_.*'
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=60

# Change for multinode config
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=1
PP=1
SEQ_LEN=16384
MBS=1
GBS=256
# NSTEP=2500
NSAMPLES=4800000  # 4803515

DATA_PATH="${TRAINING_PATH}/vlm/stage_2_dataset.yaml"
TOKENIZER_PATH="${TRAINING_PATH}/tokenizers/wbl_mm_tokenizer_v5"
CHECKPOINT_PATH="${TRAINING_PATH}/checkpoints/stage_1/checkpoints/v12_rice_gpt_stage_1_7b_a1b_gbs_256_lr_2e_4_warmup_0_1_decay_0_01_no_moe_loss_tk_v5__vlm_20251231_wbl_llm_sft_1229_without_tulu_iter_0013500"

DEFAULT_OUTPUT_DIR="${TRAINING_PATH}/checkpoints/stage_2"
JOB_NAME="v14_rice_gpt_stage_2_7b_a1b_gbs_${GBS}_vv_and_stage1_data_lr_1e5_vlrmult_0_1_warmup_0_1_decay_0_01_tk5_router_freeze__v12"

SAVE_CKPT_PATH="${DEFAULT_OUTPUT_DIR}/checkpoints/${JOB_NAME}"
TENSORBOARD_PATH="${DEFAULT_OUTPUT_DIR}/tensorboards/${JOB_NAME}"
WANDB_PATH="${DEFAULT_OUTPUT_DIR}/wandb/${JOB_NAME}"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
# mkdir -p "$WANDB_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"

cp "$DATA_PATH" "$SAVE_CKPT_PATH"

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26000"}

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

MODEL_ARGS=(
    --model-name rice-gpt-7b-a1b
    --disable-bias-linear
    --seq-length "${SEQ_LEN}"
    --decoder-seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --num-layers 24
    --position-embedding-type none
    --hidden-size 1536
    --ffn-hidden-size 5760
    --num-attention-heads 12
    --init-method-std 0.0134
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion

    --sliding-window-interleave-k 6
    --sliding-window-size 512
)

# --moe-ffn-hidden-size는 세팅하지 않는다. ffn-hidden-size를 따라감.
#--moe-use-upcycling
MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq '([0]*1+[1]*23)'
    --moe-ffn-hidden-size 960
    --moe-shared-expert-intermediate-size 960 # 2048
    --moe-router-load-balancing-type global_aux_loss
    --moe-router-topk 5
    --moe-grouped-gemm
    --moe-aux-loss-coeff 0.   # 2e-2 #1e-4
    --moe-z-loss-coeff 0.     # 1e-3
    --moe-router-topk-scaling-factor 1.0
    --moe-router-score-function 'sigmoid'
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-dtype fp32
    --overlap-param-gather
    --overlap-grad-reduce
)

# YaRN을 사용하지 않으므로 mscale을 넣지 않는다.
#    --mscale 1.0
#    --mscale-all-dim 1.0
MLA_ARGS=(
    --multi-latent-attention
    --q-lora-rank 768 # 768
    --kv-lora-rank 512 # 768
    --qk-head-dim 128 
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1.0
    --normalization RMSNorm
    --rope-type rope
    --apply-layernorm-1p
    --rotary-base 10000
    --rotary-base-global 1000000
    --qk-layernorm
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template rice-gpt
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules all
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --vision-lr-mult 0.1
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    # --train-iters "${NSTEP}"
    # --lr-decay-iters "${NSTEP}"
    --train-samples "${NSAMPLES}"
    --lr-decay-samples "${NSAMPLES}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    # --initial-loss-scale 65536
    --bf16
    # --load "$CHECKPOINT_PATH"
    --pretrained-checkpoint "$CHECKPOINT_PATH" 
    --save "$SAVE_CKPT_PATH"
    --save-interval 2000
    --ckpt-format torch_dist
    # --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
    --exit-on-missing-checkpoint

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 4

    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32

    # --async-save
    --no-load-rng
    --no-load-optim

    --eval-iters 0

    --manual-gc
    --manual-gc-interval 10
    
    --no-check-for-nan-in-loss-and-grad
    # --cross-entropy-loss-fusion
    # --cross-entropy-fusion-impl 'te'

    --sft
    --finetune

    --freeze-router
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size 8
    --context-parallel-size 1
    # --sequence-parallel
    --use-distributed-optimizer
    --cp-comm-type 'p2p'
    --distributed-backend nccl
)

# FP8_ARGS=(
#     --fp8-format 'e4m3'
#     --fp8-recipe 'blockwise'
#     --fp8-amax-history-len 1024 
#     --fp8-amax-compute-algo 'max'
#     --num-layers-at-start-in-bf16 1
#     --num-layers-at-end-in-bf16 1
# )

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --tensorboard-queue-size 1000
    --use-pytorch-profiler  # 이걸 키면 느려지므로 실제 학습에서는 끄는게 좋다고함 (etri 에서는 속도 차이가 크게 없다고 하긴함)
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-throughput
    --log-params-norm
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
        --wandb-save-dir "${WANDB_PATH}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}.log"

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72

export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FLASH_ATTN=1

PYTHONPATH="$TRAINING_PATH:$MEGATRON_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$TRAINING_PATH/vlm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${MLA_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"

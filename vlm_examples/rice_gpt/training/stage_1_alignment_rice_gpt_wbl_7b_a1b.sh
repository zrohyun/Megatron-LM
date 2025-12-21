#!/bin/bash

TRAINING_PATH="${TRAINING_PATH:-/mnt/output/ncai/multi_modal/Megatron-LM}"
MEGATRON_PATH="${MEGATRON_PATH:-${TRAINING_PATH%/}megatron}"

TP=1
PP=1
SEQ_LEN=4096
MBS=8
GBS=64
NSTEP=8720

DATA_PATH="${TRAINING_PATH}/vlm/pretrain_dataset.yaml"
TOKENIZER_PATH="/mnt/output/ncai/multi_modal/Megatron-LM/tokenizers/wbl_tokenizer_v2_mm"
CHECKPOINT_PATH="/mnt/output/ncai/multi_modal/Megatron-LM/vlm_merged/iter_0000001"

#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
# List of IP addresses for the nodes in the training cluster
declare -a list_ip=(
    "localhost"
)

# Get the primary IP of the current node
CURRENT_IP=$(hostname -I | awk '{print $1}')

if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

# Dynamically determine NNODES, MASTER_ADDR
NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    echo "--- Single-node mode ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
else
    # Find the rank of the current node
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done

    # Exit if the current IP is not in the list
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        echo "Please run this script on a node with an IP in list_ip."
        exit 1
    fi

    echo "--- Running on ${NNODES} nodes ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi
# --- End of Multi-node configuration ---


SAVE_CKPT_PATH="${TRAINING_PATH}/checkpoints/$(basename "$0" .sh)"
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
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
)

# --moe-ffn-hidden-size는 세팅하지 않는다. ffn-hidden-size를 따라감.
#--moe-use-upcycling
MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq '([0]*1+[1]*23)'
    --moe-ffn-hidden-size 960
    --moe-shared-expert-intermediate-size 960 # 2048
    # --moe-router-padding-for-fp8
    --moe-router-load-balancing-type global_aux_loss
    --moe-router-topk 5
    --moe-grouped-gemm
    --moe-aux-loss-coeff 2e-2 #1e-4
    --moe-z-loss-coeff 1e-3
    --moe-router-topk-scaling-factor 2.5
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
    --training-phase pretrain    # pretrain
    --trainable-modules vision_projection
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-4
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "${NSTEP}"
    --lr-decay-iters "${NSTEP}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 2000
    --ckpt-format torch_dist
    # --dataloader-save "${SAVE_CKPT_PATH}/dataloader"  # EP > 1 이면 dataloader-save 가 안되는듯????

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 4

    # --async-save
    --no-load-rng
    --no-load-optim

    --eval-iters 0
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --expert-model-parallel-size 8
    --context-parallel-size 1
    # --sequence-parallel   # Sequence parallelism requires tensor_model_parallel_size > 1
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
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72

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
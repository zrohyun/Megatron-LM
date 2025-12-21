#!/bin/bash

# Runs Mixtral 8x7B model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456
export NNODES=1
export GPUS_PER_NODE=8
export NODE_RANK=0                 # ★ 마스터는 반드시 0

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
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 인자 순서, load, save checkpoint, tokenizer, data path
CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
)

# MLA이므로 position-embedding-type을 none으로 둔다.
#  --no-position-embedding은 deprecated. 쓰지 않는다.
#  --max-position-embeddings 4096 이외의 값을 사용하면 경고가 나온다.
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 17
    --position-embedding-type none
    --hidden-size 2048
    --ffn-hidden-size 3584
    --num-attention-heads 16
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
    --num-experts 128
    --moe-layer-freq '([0]*1+[1]*16)'
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512 # 2048
    --moe-shared-expert-overlap
    --moe-router-padding-for-fp8
    --moe-router-load-balancing-type global_aux_loss
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-2 #1e-4
    --moe-z-loss-coeff 1e-3
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
    --q-lora-rank 1024 # 768
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

# MOE 제거 
DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --per-split-data-args-path per_split_data_args_dict_v3.json
)

# global-batch-size * sequence length = Effective batch size임
# 본 모델 형식에서 튜닝함. global-batch-size는 WORLD_SIZE * micro_batch_size의 배수여야 함. 즉, 8개 GPU면 3*8 = 24의 배수.
TRAINING_ARGS=(
    --micro-batch-size 2 #1
    --global-batch-size 1024 #512
    --lr 2e-4
    --min-lr 2.0e-4
    --train-iters 20 # 2000000
    --lr-decay-style constant
    --weight-decay 0.1
    --lr-warmup-iters 0 # 7500
    --clip-grad 1.0
    --bf16
    --grad-reduce-in-bf16
    --use-flash-attn
    --attention-backend 'flash'
    --seed 20251002
    --manual-gc \
    --manual-gc-interval 10 \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl 'te'
)

# 아래 parallel-size의 곱이 world_size와 같아야 한다.
# --cp-comm-type 'a2a+p2p' 를 하는 경우, --hierarchical-context-parallel-sizes [숫자] 설정을 필요.
# MoE-specific TP size를 세팅하기 위해서는 --expert-tensor-parallel-size [숫자]로 세팅할 것
# tensor-model-parallel-size가 2 이상일 경우 --tp-comm-overlap을 사용.
# tensor-model-parallel-size가 2 미만일 경우 --sequence-parallel은 disable 됨에 주의.
# context-parallel과 expert-parallel 수는 동일하게 맞춰도 된다 (배수로 적용되지 X)
# --recompute-activations
#    --tp-comm-overlap
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --cp-comm-type 'p2p'
)

#    --fp8-recipe 'delayed'
#    blockwise scaling에는 CUDA 12.9 세팅이 필요.
#    --fp8-recipe 'blockwise'
#    blockwise가 좀 더 빠르긴 한데, loss 흔들림이 소형 모델 (5B) 테스트에서 발견되었음.
FP8_ARGS=(
    --fp8-format 'e4m3'
    --fp8-recipe 'blockwise'
    --fp8-amax-history-len 1024 
    --fp8-amax-compute-algo 'max'
    --num-layers-at-start-in-bf16 1
    --num-layers-at-end-in-bf16 1
)

# 속도차이?   --use-pytorch-profiler 차이는 크게 없음을 확인 
# save-interval은 0.5 day / 1 day로 잡아도 되고, retain-interval은 그것의 4~7배로 잡는 것을 추천 (4일~1주일 분량)
#    --log-progress \
#    --log-params-norm \
# FIXME: upcycling 활성화하려면 아래의 load parameter를 사용.
#    --load $LOAD_PATH \
#    --load $CHECKPOINT_PATH \
# 25.08.16 upcycling은 checkpoint loading 하는 파트 마저 수정 필요. 아직 잘 작동하지 않아서 수정할게 많다.
LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 20000
    --save-interval 20  # 2000
    --save-retain-interval 10000
    --ckpt-format torch_dist
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --tensorboard-log-interval 1
    --tensorboard-queue-size 1000
    --use-pytorch-profiler
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-throughput
    --log-params-norm
    --no-load-optim
    --no-load-rng
    --async-save
    --timing-log-level 2
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"WBL_MOE_SMALL"}
        --wandb-exp-name ${WANDB_NAME:-"WBL_MOE_SMALL_TEST1"}
    )
fi

# WORLD_SIZE=48 python -u report_theoretical_memory.py \
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt_for_wbl.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
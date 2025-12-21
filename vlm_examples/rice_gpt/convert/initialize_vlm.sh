# Runs Mixtral 8x7B model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456
export NNODES=1
export GPUS_PER_NODE=1
export NODE_RANK=0                 # ★ 마스터는 반드시 0

# Change for multinode config
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
)


SCRIPT_ARGS=(
    --language-model-path "/workspace/multimodalmodel_team/outputs/checkpoints/sft_1206_lr_3e6_seq_16384"
    --vision-model-path "/workspace/Megatron-LM/checkpoints/mcore_models/llava-ov-1.5/vision-model-mcore/release"
    --vision-patch-path "/workspace/Megatron-LM/checkpoints/mcore_models/llava-ov-1.5/patch-mcore/release"
    --save-ckpt-path "/workspace/Megatron-LM/checkpoints/merged/vlm_20251217__wbl_llm_sft_1206_lr_3e6_seq_16384"
    --model-name "rice-gpt-7b-a1b"
    --allow-missing-vision-projection-checkpoint
    --trainable-modules "all"
)


MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 16384
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

MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq '([0]*1+[1]*23)'
    --moe-ffn-hidden-size 960
    --moe-shared-expert-intermediate-size 960 # 2048
    # --moe-shared-expert-overlap
    # --moe-router-padding-for-fp8
    --moe-router-load-balancing-type global_aux_loss
    --moe-router-topk 5
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
    --tokenizer-model ./tokenizers/wbl_mm_tokenizer_v5
    --per-split-data-args-path per_split_data_args_dict_v3.json
)

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
    --seed 2275
    --manual-gc \
    --manual-gc-interval 10 \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl 'te'

    --decoder-seq-length 16384

    --no-load-optim
    --no-load-rng
    --no-save-rng
    --no-save-optim

    --ckpt-format torch_dist
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --cp-comm-type 'p2p'
)

FP8_ARGS=(
    --fp8-format 'e4m3'
    --fp8-recipe 'blockwise'
    --fp8-amax-history-len 1024 
    --fp8-amax-compute-algo 'max'
    --num-layers-at-start-in-bf16 1
    --num-layers-at-end-in-bf16 1
)


torchrun ${DISTRIBUTED_ARGS[@]} -m vlm_examples.rice_gpt.convert.initialize_vlm \
    ${SCRIPT_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]}

#!/bin/bash
# vLLM Server with Data Parallel (DP=8, TP=1)
# Usage: bash start_serving_dp8_tp1.sh

set -e

PORT=${PORT:-8000}
GPU_MEM=${GPU_MEMORY_UTILIZATION:-0.4}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}  # Increased for multi-image support (2-3 images, safe for OOM)
TIMEOUT=${TIMEOUT:-300}
DATA_PARALLEL_SIZE=8
MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_4/checkpoints/v1_rice_gpt_stage_4_7b_a1b_gbs_128_lr_3e6_warmup_0_02_18M_tk5_no_capacity__v9__v6/iter_0144531-HF"}
# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_x/checkpoints/v1_rice_gpt_stage_x_7b_a1b_gbs_128_lr_1e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v1__v9__v6/iter_0027578-HF"}
# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_4/checkpoints/v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/iter_0064000-HF"}

# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/ncai/multimodal/outputs/vlm_stage_4/checkpoints/v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/iter_0032000-HF"}
# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/ncai/multimodal/outputs/vlm_stage_4/checkpoints/v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/iter_0032000-HF"}
echo "================================================"
echo "Starting vLLM Server with Data Parallel"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "Data Parallel Size: $DATA_PARALLEL_SIZE"
echo "Tensor Parallel Size: 1"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Request Timeout: ${TIMEOUT}s"
echo "================================================"
echo ""

# Apply monkey patches for tokenizer compatibility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Environment variables for stability
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Disabled for stability (enforce-eager incompatible)
export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE_LEVEL=0 
export TORCH_COMPILE_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=6,7
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --dtype bfloat16 \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEM \
    --port $PORT \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-model-len $MAX_MODEL_LEN \
    --disable-log-stats \
    --limit-mm-per-prompt '{"image": 10}' \
    --served-model-name "wbl-vlm-7b" \
    --max-log-len 100 \
# --enforce-eager
# --disable-custom-all-reduce \

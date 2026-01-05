#!/bin/bash
# vLLM OpenAI API Compatible Server
# Usage: bash start_serving.sh [tensor_parallel_size]

set -e

TP_SIZE=${1:-1}
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEMORY_UTILIZATION:-0.9}
# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/ncai/multimodal/outputs/vlm_stage_4/checkpoints/v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/iter_0010000-HF"}
# MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_4/checkpoints/v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/iter_0064000-HF"}
MODEL_PATH=${MODEL_PATH:-"/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_x/checkpoints/v1_rice_gpt_stage_x_7b_a1b_gbs_128_lr_1e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v1__v9__v6/iter_0027578-HF"}

echo "================================================"
echo "Starting vLLM Server"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM"
echo "================================================"
echo ""

# Apply monkey patches for tokenizer compatibility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# NOTE: 변환 코드 변경 후에 v1 engine 환경변수 강제 해줘야함.
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM \
    --port $PORT \
    --disable-log-stats \
    --disable-custom-all-reduce \
    --limit-mm-per-prompt '{"image": 10}' \
    --served-model-name "wbl-vlm-7b"
    # --enforce-eager \

# Example usage after server starts:
#
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "wbl-vlm-7b",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {"type": "text", "text": "What is in this image?"},
#           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
#         ]
#       }
#     ],
#     "max_tokens": 128
#   }'

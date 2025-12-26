#!/bin/bash

TRAINING_PATH="${TRAINING_PATH:-/workspace/Megatron-LM}"

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

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26000"}

NUM_PROCESSES=$((NNODES * 8))

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72


MODEL_PATH="/workspace/multimodalmodel_team/data/checkpoint/WBLVLMoE-A1B-Stage4-HF/iter_0144531-HF"


PYTHONPATH="$TRAINING_PATH:$PYTHONPATH" \
    accelerate launch \
        --config_file ${TRAINING_PATH}/vlm_examples/rice_gpt/training/config/config_fsdp.json \
        --num_machines $NNODES \
        --num_processes $NUM_PROCESSES \
        --machine_rank $NODE_RANK \
        "$TRAINING_PATH/vlm/train_dpo.py" \
            --data_path "/workspace/varco-mllm/dpo_data/wbl_arrow" \
            --model_name_or_path $MODEL_PATH \
            --learning_rate 5.0e-7 \
            --loss_type sigmoid bco_pair sft \
            --loss_weights 0.8 0.2 1.0 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --max_length 16384 \
            --gradient_accumulation_steps 1 \
            --eval_strategy no \
            --eval_steps 0 \
            --output_dir checkpoints/vlm-dpo-test-v1 \
            --report_to tensorboard  \
            --trust_remote_code true \
            --no_remove_unused_columns \
            --attn_implementation flash_attention_2 \
            --save_strategy steps \
            --save_safetensors true \
            --save_steps 100 \
            --save_only_model true \
            --gradient_checkpointing true \
            --warmup_ratio 0.1 \
            --dataloader_num_workers 16 \
            --logging_steps 1 \
            --bf16 true \
            --dtype bfloat16 \
            --per_device_eval_batch_size 1
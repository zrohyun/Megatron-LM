set -e

# ========== 기본 설정 ==========
NUM_GPUS=1
NNODES=1
SCRIPT="vlm/bridge/export_megatron_to_hf.py"

HF_MODEL="vlm/bridge/WBLVLMoE-A1B-HF-Dummy"

TRAINING_PATH="/mnt/vlm-data/personal/zro/workspace/Megatron-LM-gh-zro"
# MEGATRON_PATH="${TRAINING_PATH}/megatron"


MCORE_BASE_PATH="/mnt/checkpoint/ncai/multimodal/outputs"
MCORE_PATH="${MCORE_PATH:-${MCORE_BASE_PATH}/vlm_stage_3/checkpoints/v5_rice_gpt_stage_3_7b_a1b_gbs_128_lr_3e6_warmup_0_02_finevision_4M_tk6_no_capacity__v10__v6}"

export MEGATRON_PATH="${MEGATRON_PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# PYTHONPATH="${MEGATRON_BRIDGE_PATH}:${PYTHONPATH}" \
python -m torch.distributed.run \
 --standalone \
 --nnodes=${NNODES} \
 --nproc_per_node=${NUM_GPUS} \
  ${SCRIPT} \
  --hf-model ${HF_MODEL} \
  --megatron-path ${MCORE_PATH} \
  --hf-path ${MCORE_PATH}-HF
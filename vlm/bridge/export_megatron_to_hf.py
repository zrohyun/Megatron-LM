import torch
import shutil
import argparse
import sys
from pathlib import Path
import pdb

# 커스텀 HF 모델 등록을 위해 modeling 파일 import
# WBLVLMoE-A1B-HF-Dummy 경로를 sys.path에 추가
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "WBLVLMoE-A1B-HF-Dummy"))

# 이 import가 실행되면 AutoModelForCausalLM.register가 호출됨
from modeling_wbl_vl_moe import WBLVLMoEForCausalLM, WBLVLMoEConfig
import transformers
# transformers 모듈에 직접 등록
transformers.WBLVLMoEForCausalLM = WBLVLMoEForCausalLM
transformers.WBLVLMoEConfig = WBLVLMoEConfig

from megatron.core import parallel_state
from megatron.core.enums import ModelType

from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.bridge import AutoBridge

# from pretrain_gpt_for_wbl import model_provider_with_args
from vlm.bridge.rice_gpt_provider import rice_gpt_model_provider_with_args

from vlm.bridge.wbl_vl_moe_bridge import WBLBridge # register bridge


## 특정 체크포인트에 따라 아래 args_config로 보완해줘야 하는 경우들이 있음
SEQ_LEN = 4096

# Megatron args 정의
args_config = {
    # === 모델 구조 ===
    'model-name': 'rice-gpt-7b-a1b',
    'seq-length': SEQ_LEN,
    'decoder-seq-length': SEQ_LEN,
    'max-position-embeddings': 32768,
    'num-layers': 24,
    'position-embedding-type': 'none',  # !!!! None 으로 받으면 안됨.. str only 
    'hidden-size': 1536,
    'ffn-hidden-size': 5760,
    'num-attention-heads': 12,
    'init-method-std': 0.0134,    
    'untie-embeddings-and-output-weights': True,
    'no-masked-softmax-fusion': True,
    'attention-dropout': 0.0,
    'hidden-dropout': 0.0,
    'swiglu': True,    
    'normalization': 'RMSNorm', 

    # === MoE ===
    'num-experts': 64,
    'moe-ffn-hidden-size': 960,
    'moe-shared-expert-intermediate-size': 960,
    'moe-router-load-balancing-type': 'global_aux_loss',
    'moe-router-topk': 5,
    'moe-grouped-gemm': True,
    'moe-router-topk-scaling-factor': 2.5,
    'moe-router-score-function': 'sigmoid',
    'moe-token-dispatcher-type': 'alltoall',
    'moe-permute-fusion': True,
    'moe-router-dtype': 'fp32',
    'bf16': True,
    
    # === 병렬화 ===
    'tensor-model-parallel-size': 1,
    'pipeline-model-parallel-size': 1,
    
    # === Tokenizer ===
    'tokenizer-type': 'HuggingFaceTokenizer', # 커스텀 경로를 쓴다면 보통 이것
    'tokenizer-model': "/workspace/vlm/tokenizers/wbl_mm_tokenizer_v3",
    # 'vocab-size': 151936,
    
    # === 학습/추론 기본 ===
    'micro-batch-size': 1,    
    'global-batch-size': 1, # 필수 인자 경우가 많음

    # === MLA 설정 === 
    'multi-latent-attention': True,
    'q-lora-rank': 768,
    'kv-lora-rank': 512,
    'qk-head-dim': 128,
    'qk-pos-emb-head-dim': 64,
    'v-head-dim': 128,
    'rotary-scaling-factor': 1.0,
    'rope-type': 'rope',
    'apply-layernorm-1p': True,
    'rotary-base': 10000,
    'rotary-base-global': 1000000,
    'qk-layernorm': True,
    
    # === Checkpoint ===
    'no-load-rng': True,
    'no-load-optim': True,
    'load-optim': False,
    'load-rng': False,
    
    # === Inference ===
    'inference-max-batch-size': 4,
    'inference-max-seq-length': SEQ_LEN,
    'inference-max-requests': 1,
    
    # === 기타 설정 ===    
    'use-cpu-initialization': True, # 메모리 절약
    'data-path': '',
    'save': '',
    # 'load': CHECKPOINT_PATH,
    'distributed-backend': 'nccl',    
    'recompute_vision': False,
    "allow-missing-vision-projection-checkpoint": False,
    "trainable_modules": ["vision_projection=="],
    # 'recompute-granularity': 'full',
    # 'recompute-method': 'uniform',
    # 'recompute-num-layers': 4,    

    # === dummy ===
    'save-interval': '10000',    # "몇 스텝마다 저장할래?" (실제 저장은 안 함)
    'train-iters': '1',          # "몇 번 학습할래?" (1로 설정해두고 eval 모드 사용)
    'eval-iters': '1',           # "검증은 몇 번 할래?"
    'eval-interval': '1000',     # "검증 주기는?"
    'split': '949,50,1',         # "데이터셋은 어떻게 나눌래?" (데이터 없어도 형식상 필요)

    # === 테스트용
    'input-path': "/workspace/Megatron-LM/test-images",
    'prompt-type': "captioning",
    'output-path': "/workspace/Megatron-LM/test-outputs",
    'num-partitions': 0,
    'partition-id': 0,
    'gt-path': '',

}

def load_megatron_model(megatron_path):
    _, mlm_args = load_model_config(megatron_path)
    # mlm_args = vars(mlm_args)
    # for k, v in args_config.items():
    #     if k not in mlm_args:
    #         mlm_args[k.replace("-", "_")] = v
    # mlm_args  = argparse.Namespace(**mlm_args)
    mlm_args.use_cpu_initialization = args.use_cpu_initialization
    
    mlm_args.sliding_window_size = 512
    mlm_args.sliding_window_interleave_k = 6

    # TODO: parallel conversion
    mlm_args.context_parallel_size = 1
    mlm_args.expert_model_parallel_size = 1
    mlm_args.expert_tensor_parallel_size = 1
    mlm_args.pipeline_model_parallel_size = 1
    mlm_args.tensor_model_parallel_size = 1
    mlm_args.sequence_parallel = False
    mlm_args.context_parallel_size = 1
    mlm_args.transformer_pipeline_model_parallel_size = 1

    mlm_args.recompute_granularity = None

    # with torch.device("meta"):
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = rice_gpt_model_provider_with_args(mlm_args, pre_process=pre_process, post_process=post_process)
    model.model_type = ModelType.encoder_or_decoder
    _load_model_weights_from_checkpoint(megatron_path, [model])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--megatron-path", type=str)
    parser.add_argument("--hf-path", type=str)
    parser.add_argument('--no-use-cpu-initialization', action='store_false', dest='use_cpu_initialization')
    args = parser.parse_args()

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    backend = "gloo" if args.use_cpu_initialization else "nccl"
    with temporary_distributed_context(backend):
        megatron_model = load_megatron_model(args.megatron_path)
        bridge.save_hf_pretrained([megatron_model], args.hf_path)
        shutil.copy(f"{args.hf_model}/modeling_wbl_vl_moe.py", args.hf_path)
        shutil.copy(f"{args.hf_model}/preprocessor_config.json", args.hf_path)
        shutil.copy(f"{args.hf_model}/video_preprocessor_config.json", args.hf_path)
    print(f"✅ Successfully exported model to: {args.hf_path}")

    export_path = Path(args.hf_path)
    if export_path.exists():
        print("📁 Export structure:")
        for item in export_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")

    print("🔍 You can now load this model with:")
    print("   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.hf_path}')")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
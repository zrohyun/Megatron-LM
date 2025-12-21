# %% cell 1: prepare and model load
import os

os.environ["CUDA_VISIBLE_DEVICES"] = (
    "2"  # 사용할 GPU 번호 * import torch  하기 전에 해야 함
)
# cell 1: 환경 설정 & import
import os
import sys
import torch
import traceback
import copy
from datetime import datetime

# Megatron-LM 경로 추가 (실제 경로로 수정)
MEGATRON_PATH = "/workspace/Megatron-LM"  # 실제 Megatron-LM repo 경로
sys.path.insert(0, MEGATRON_PATH)

# Megatron import
from megatron.core import mpu
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron  # notebook용
from vlm.train.pretrain.pretrain_rice_gpt import model_provider  # 실제 모델 provider
from megatron.training import get_args, initialize_megatron, get_model

from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.inference.text_generation.generation import (
    generate_tokens_probs_and_return_on_first_stage,
)

from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

print("Megatron imports completed")

# Jupyter의 기본 인자(-f kernel.json)를 제거하여 충돌 방지
sys.argv = [""]

# RANK 0으로 강제 설정 (single GPU 테스트용)
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # 보통 Megatron에서 필요함

SEQ_LEN = 4096
TP = 1
PP = 1
TOKENIZER_PATH = "/workspace/Megatron-LM/tokenizers/wbl_mm_tokenizer_v3"

# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_3_7b_a1b_251216"
CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_1_7b_a1b_251205"
# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_2_7b_a1b_251213"  #rice_gpt_stage_3_7b_a1b_251216 iter 0078000
# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/outputs/checkpoints/stage_1_alignment_rice_gpt_wbl_7b_a1b"
# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/outputs/checkpoints/vlm_merged_wbl_llm_sft_lr_3e6_seqlen_16384_iter_0008300"
# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_1_7b_a1b_251217_vit_re"
# CHECKPOINT_PATH = "/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_2_7b_a1b_251217_v7"

# Megatron args 정의
args_config = {
    # === 모델 구조 ===
    "model-name": "rice-gpt-7b-a1b",
    "num-layers": 24,
    "hidden-size": 1536,
    "ffn-hidden-size": 5760,
    "num-attention-heads": 12,
    "seq-length": SEQ_LEN,
    "decoder-seq-length": SEQ_LEN,
    "max-position-embeddings": 32768,
    "position-embedding-type": "none",  # VLM 등에서 Vision Encoder가 처리할 경우 'none' 사용
    "normalization": "RMSNorm",
    "swiglu": True,
    "untie-embeddings-and-output-weights": True,
    "disable-bias-linear": True,
    "no-masked-softmax-fusion": True,
    # Dropout은 추론 시 0.0이 정석 (eval 모드에서도 확실히 하기 위해)
    "attention-dropout": 0.0,
    "hidden-dropout": 0.0,
    # === [필수] MLA (Multi-Latent Attention) 설정 ===
    "multi-latent-attention": True,
    "q-lora-rank": 768,
    "kv-lora-rank": 512,
    "qk-head-dim": 128,
    "qk-pos-emb-head-dim": 64,
    "v-head-dim": 128,
    "rotary-scaling-factor": 1.0,
    "rope-type": "rope",
    "apply-layernorm-1p": True,
    "rotary-base": 10000,
    "rotary-base-global": 1000000,
    "qk-layernorm": True,
    # === [필수] MoE (Mixture of Experts) 설정 ===
    "num-experts": 64,
    "moe-layer-freq": "([0]*1+[1]*23)",
    "moe-ffn-hidden-size": 960,
    "moe-shared-expert-intermediate-size": 960,
    "moe-router-load-balancing-type": "global_aux_loss",
    "moe-router-topk": 5,
    "moe-grouped-gemm": True,
    "moe-router-topk-scaling-factor": 2.5,
    "moe-router-score-function": "sigmoid",
    "moe-token-dispatcher-type": "alltoall",
    "moe-permute-fusion": True,
    "moe-router-dtype": "fp32",
    # === [필수] 병렬화 및 정밀도 ===
    "tensor-model-parallel-size": TP,
    "pipeline-model-parallel-size": PP,
    "bf16": True,
    "use-cpu-initialization": True,  # 메모리 부족 방지
    "distributed-backend": "nccl",
    # === [필수] Tokenizer 및 Checkpoint 로드 ===
    "tokenizer-type": "HuggingFaceTokenizer",
    "tokenizer-model": TOKENIZER_PATH,
    "load": CHECKPOINT_PATH,
    # ★ 추론 핵심: Optimizer와 RNG 상태는 로드하지 않음 (속도 향상 & 에러 방지)
    "no-load-optim": True,
    "no-load-rng": True,
    # 'no-initailization': True,
    "no-initialization": True,
    # 'lazy-mpu-init': True,
    # === [필수] Inference 설정 ===
    "inference-max-batch-size": 4,
    "inference-max-seq-length": SEQ_LEN,
    "inference-max-requests": 1,
    # === [Megatron 초기화 통과용 Dummy] ===
    # 아래 값들은 추론에 쓰이지 않지만, 없으면 initialize_megatron에서 에러가 날 수 있음
    "micro-batch-size": 1,
    "global-batch-size": 1,
    "train-iters": 1,
    "eval-iters": 1,
    "save-interval": 10000,
    "split": "100,0,0",  # 데이터셋 split 비율 더미
}

sys.argv = ["inference_script.py"]
for key, value in args_config.items():
    if value is False or value is None:
        continue

    # Boolean Flag 처리 (값이 True면 플래그만 추가)
    if value is True:
        sys.argv.append(f"--{key}")
    else:
        # 값을 가진 인자 처리
        sys.argv.append(f"--{key}")
        sys.argv.append(str(value))


def add_vlm_inference_args(parser):
    group = parser.add_argument_group(title="VLM Inference")

    group.add_argument(
        "--prompt-type",
        type=str,
        default="captioning",
        choices=["captioning", "qa"],
        help="Test type",
    )
    group.add_argument(
        "--model-name", type=str, default="rice_vlm", help="Model Name for VLM"
    )
    group.add_argument(
        "--num-partitions", type=int, default=0, help="Path to input image"
    )
    group.add_argument("--partition-id", type=int, default=0, help="Partition index")
    group.add_argument("--drop-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--gt-path", type=str, default=None, help="Optional ground truth file"
    )
    return parser


try:
    # args_defaults를 쓰지 않고, 위에서 만든 sys.argv를 통해 파싱하게 합니다.
    initialize_megatron(
        extra_args_provider=add_vlm_inference_args, ignore_unknown_args=True
    )
    print("✅ Initialization Success!")
    # 확인
    args = get_args()
    print(f"Micro Batch Size: {args.micro_batch_size}")

except Exception:
    print("❌ Initialization Failed! 상세 에러 로그:")
    traceback.print_exc()

model = get_model(model_provider, wrap_with_ddp=False)
print(f" 📂 get model structure from model provider")

print("now loading... :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# ==== 실행 ====
# load_dir = '/workspace/multimodalmodel_team_data/checkpoint/rice_gpt_stage_2_7b_a1b_251213'
# args.load = load_dir
# load_checkpoint_smart(model, load_dir)
load_checkpoint(model, None, None, strict=False)

if isinstance(model, list):
    model = model[0]

# 모델을 추론용으로 스위치
model.eval()

print("Loaded time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

try:
    image_processor = AutoProcessor.from_pretrained(
        TOKENIZER_PATH, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    traceback.print_exc()
    raise e


# inference에 사용할 forward overriding / input feeding을 담당
class VLMForwardStep(ForwardStep):
    def __init__(
        self,
        images,
        image_grid_thw,
        model,
        max_batch_size,
        max_sequence_length,
        inference_context,
    ):
        # image_grid_thw 인자 추가
        super().__init__(model, max_batch_size)
        self._images = images
        self._image_grid_thw = image_grid_thw  # 저장해둠
        self.max_sequence_length = max_sequence_length
        self.inference_context = inference_context

    def _forward(self, tokens, position_ids, attention_mask=None):
        return self.model(
            images=self._images,
            image_grid_thw=self._image_grid_thw,
            input_ids=tokens,  # tokens는 input_ids로 전달
            position_ids=position_ids,
            attention_mask=attention_mask,  # None이어도 명시적으로 전달
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )

    def __call__(self, tokens, position_ids, attention_mask=None):
        logits = super().__call__(tokens, position_ids, attention_mask)

        num_tokens = tokens.size(1)
        if num_tokens > 1:
            if "image_tokens_count" in self.inference_context.key_value_memory_dict:
                self.inference_context.sequence_len_offset += (
                    self.inference_context.key_value_memory_dict["image_tokens_count"]
                )
        return logits


### 추론 함수
def generate_vlm_response(image_path, conversation, max_new_tokens=128):
    """
    Args:
        image_path (str or None): 이미지 경로. 없으면 None.
        conversation (list): [{'role': 'user', 'content': [...]}] 형태의 대화 리스트.
        max_new_tokens (int): 생성할 최대 토큰 수.
    Returns:
        str: 생성된 답변 텍스트.
    """

    # 1. 이미지 로드 및 프롬프트 구성
    pixel_values = None
    image_grid_thw = None

    # 대화 내용을 복사하여 수정 (원본 보존)
    messages = copy.deepcopy(conversation)

    if image_path:
        try:
            raw_image = Image.open(image_path).convert("RGB")
            # user 메시지(보통 첫번째)에 이미지 추가
            # (이미지 프로세서 템플릿에 따라 위치가 다를 수 있으나 일반적인 방식 적용)
            for msg in messages:
                if msg["role"] == "user":
                    # 이미지가 이미 있는지 확인 후 없으면 추가
                    has_image = any(
                        item.get("type") == "image" for item in msg["content"]
                    )
                    if not has_image:
                        msg["content"].insert(0, {"type": "image", "image": raw_image})
                    break
        except Exception as e:
            print(f"Error loading image: {e}")
            return ""

    # 2. 입력 데이터 처리 (Tokenize & Transform)
    # image_processor와 tokenizer는 전역 변수 사용 가정
    inputs = image_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 3. Input IDs 준비
    raw_input_ids = inputs["input_ids"]
    if not isinstance(raw_input_ids, torch.Tensor):
        raw_input_ids = torch.tensor(raw_input_ids, dtype=torch.long)

    if raw_input_ids.dim() == 1:
        raw_input_ids = raw_input_ids.unsqueeze(0)  # [1, seq_len]

    raw_input_ids = raw_input_ids.cuda()

    batch_size = raw_input_ids.size(0)
    if "attention_mask" in inputs:
        prompt_length = inputs["attention_mask"].sum(dim=1).max().item()
    else:
        prompt_length = raw_input_ids.size(1)

    total_sequence_length = prompt_length + max_new_tokens

    # 이미지 텐서 준비 (존재할 경우)
    if "pixel_values" in inputs:
        # 모델이 bf16이면 이미지도 bf16이어야 함
        pixel_values = inputs["pixel_values"].cuda().to(torch.bfloat16)

        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"].cuda()

    # 4. Inference Context 설정
    # 매 호출마다 새로운 컨텍스트를 생성하는 것이 안전함
    inference_context = StaticInferenceContext(
        max_batch_size=batch_size, max_sequence_length=total_sequence_length
    )

    # 5. Token Buffer 생성
    tokens = torch.zeros(
        (batch_size, total_sequence_length),
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    # 프롬프트 복사
    tokens[:, :prompt_length] = raw_input_ids

    # Lengths 텐서
    lengths = torch.tensor(
        [prompt_length] * batch_size,
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )

    # 6. Forward Step 정의
    # 람다 내부에서 매번 조건문을 도는 것보다 미리 정의된 변수를 캡처하는 것이 효율적
    # VLMForwardStep 클래스가 전역 혹은 임포트되어 있어야 함
    def custom_forward_step(model, context):
        return VLMForwardStep(
            images=pixel_values,
            image_grid_thw=image_grid_thw,
            model=model,
            max_batch_size=batch_size,
            max_sequence_length=total_sequence_length,
            inference_context=context,
        )

    print(f"Generating... (Prompt len: {prompt_length}, Max new: {max_new_tokens})")
    print(f"pixel_values dtype: {pixel_values.dtype}")
    print(f"input_ids: {tokens.dtype}")

    # 7. 생성 실행
    with torch.no_grad():
        output_tokens, generated_lengths, _, _ = (
            generate_tokens_probs_and_return_on_first_stage(
                model=model,
                inference_context=inference_context,
                forward_step=custom_forward_step,
                tokens=tokens,
                lengths=lengths,
                return_output_log_probs=False,
                top_k=1,
                temperature=0.1,
                use_eod_token_for_early_termination=True,
            )
        )

    # 8. 결과 디코딩 (프롬프트 제외하고 답변만 추출)
    # generated_lengths는 실제 생성된 전체 길이를 담고 있음
    actual_gen_len = generated_lengths[0].item()

    # [프롬프트 길이 : 실제 생성된 끝] 까지 슬라이싱
    generated_ids = output_tokens[0, prompt_length:actual_gen_len]
    decoded_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    if len(decoded_text.strip()) < 5:
        print("==== debug ====")
        print(
            "input: \n"
            + tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)
        )
        print(
            "output: \n"
            + tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
        )

    return decoded_text.strip()


# %% cell 2: inference
if __name__ == "__main__":
    # image_path = None
    image_paths = [
        "/workspace/Megatron-LM/test_images/mario.jpg",
        "/workspace/Megatron-LM/test_images/dt.jpg",
        "/workspace/Megatron-LM/test_images/test_coco.jpg",
    ]
    messages = [
        "Describe the image.",
        "Describe the image.",
        "Describe the objects in this image and their specific colors.",
    ]
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        for msg in messages
    ]
    for image_path, conv in zip(image_paths, conversations):
        print(f"{image_path}, {conv}")
        response = generate_vlm_response(image_path, conv)
        print("-" * 20)
        print("Model Response:", response)
        print("-" * 20)

# %%

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from qwen_vl_utils import process_vision_info
import pdb
import os



####### TESTING GENERATION
# 환경변수로 model_path를 받음, 없으면 기본값 사용
model_path = os.environ.get("MODEL_PATH", "/data/ISTD_VOL01/multimodalmodel_team/data/checkpoint/WBLVLMoE-A1B-Stage2-HF")
test_image = os.environ.get("TEST_IMAGE", "test_images/mario.jpg")
print(f"Using model: {model_path}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)


processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    # 
    #     "role": "system",
    #     "content": [
    #         {"type": "text", "text": "You are a helpful assistant."}
    #     ]
    # },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": test_image,
            },
            {
                "type": "image",
                "image": test_image,
            },
            # {"type": "text", "text": "Charlotte Perriand (24 October 1903 - 27 October 1999) was"},
            # {"type": "text", "text": "Hey, are you conscious? Can you talk to me?"},
            {"type": "text", "text": "Describe the image"},
            # {"type": "text", "text": "Hey, are you conscious? Can you talk to me?"},
            {"type": "text", "text": "Describe the image"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
system = "<|role_start|>system<|role_end|>\nYou are a helpful assistant\n"

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    # text=[system + text],
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
)

# print(inputs.keys())
inputs = inputs.to("cuda")
# print(processor.batch_decode(inputs.input_ids[0]))

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = model.generate(**inputs, max_new_tokens=256)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 결과 출력 (webhook에서 파싱할 수 있도록 마커 사용)
print("===MODEL_OUTPUT_START===")
for i, text in enumerate(output_text):
    print(f"[Response {i+1}]: {text}")
print("===MODEL_OUTPUT_END===")
print(f"Model: {model_path}")
print(f"Test completed successfully!")

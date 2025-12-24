import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from qwen_vl_utils import process_vision_info
import pdb
import os

# lm_path = "/data/ISTD_VOL01/lm_team/personal/jeongho/WBL-7B-A1B-Instruct-HF"
# lm_model = AutoModelForCausalLM.from_pretrained(
#     lm_path,
#     dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2"
# )
# lm_tokenizer = AutoTokenizer.from_pretrained(lm_path, trust_remote_code=True)

####### TESTING GENERATION
# model_path = '/data/ISTD_VOL01/multimodalmodel_team/data/checkpoint/rice_gpt_stage_2_7b_a1b_251210/iter_0052421-HF'
model_path = '/data/ISTD_VOL01/multimodalmodel_team/data/checkpoint/rice_gpt_stage_2_7b_a1b_251213/iter_0052421-HF'
# model_path = "/data/ISTD_VOL01/multimodalmodel_team/data/outputs/checkpoints/stage_1_alignment_rice_gpt_wbl_7b_a1b/iter_0001650-HF-debug"
# model_path = "vlm_merged_wbl_llm_sft_lr_3e6_seqlen_16384_iter_0008300-HF"
# model_path = '/data/ISTD_VOL01/multimodalmodel_team/data/outputs/checkpoints/vlm_merged_wbl_llm_sft_lr_3e6_seqlen_16384_iter_0008300-HF'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)


processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# processor.tokenizer = AutoTokenizer.from_pretrained("/workspace/tokenizers/wbl_tokenizer_v3_mm", trust_remote_code=True)
# processor.tokenizer = lm_tokenizer


messages = [
    # {
    #     "role": "system",
    #     "content": [
    #         {"type": "text", "text": "You are a helpful assistant."}
    #     ]
    # },
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "vlm/bridge/mario.jpg",
            # },
            # {"type": "text", "text": "Charlotte Perriand (24 October 1903 - 27 October 1999) was"},
            {"type": "text", "text": "Hey, are you conscious? Can you talk to me?"},
            # {"type": "text", "text": "Describe the image"},
        ],
    }
]

# prompt = "Charlotte Perriand (24 October 1903 - 27 October 1999) was"
# inputs = processor.tokenizer(prompt, padding=True, return_tensors="pt")["input_ids"].to(model.device)

# generate_ids = lm_model.generate(inputs, max_new_tokens=100)

# print(processor.batch_decode(generate_ids[0]))

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
system = "<|role_start|>system<|role_end|>\nYou are a helpful assistant\n"
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[system + text],
    # text=[text],
    images=image_inputs,
    videos=video_inputs,
    # padding=True,
    return_tensors="pt",
)
print(inputs.keys())
inputs = inputs.to("cuda")
input_ids = inputs["input_ids"]

print(processor.batch_decode(input_ids[0]))

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=64)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


decoder = model
# --- 2. Define Generation Parameters ---
MAX_NEW_TOKENS = 100
TEMPERATURE = 1
TOP_P = 0.9 
# Use the decoder's config or the tokenizer's properties for these:
EOS_TOKEN_ID = processor.tokenizer.eos_token_id 
PAD_TOKEN_ID = processor.tokenizer.pad_token_id or EOS_TOKEN_ID

# --- 4. The Custom Autoregressive Generation Loop ---
for _ in range(MAX_NEW_TOKENS):
    # A. Forward Pass
    # Use the decoder to get the logits for the next token.
    # The decoder's forward pass gives the model's prediction for the NEXT token.
    # We only care about the logits corresponding to the LAST token in the sequence.
    
    # NOTE: Pass only the input_ids to the extracted decoder. 
    # Use 'labels=None' or similar if the decoder requires a specific signature.
    with torch.no_grad():
        # Get output from the raw decoder (e.g., LlamaForCausalLM)
        # The structure might vary slightly, but we need the logits tensor.
        outputs = decoder(input_ids=input_ids)
    
        # # 2. Extract the hidden state from the base output
        # last_hidden_state = outputs.last_hidden_state
        
        # # 3. Manually pass the hidden state through the LM Head
        # # This projects the vectors (hidden_size) to the vocabulary size (vocab_size)
        # logits = model.lm_head(last_hidden_state)
        logits = outputs.logits

    # B. Get Next Token Logits
    # Slice the logits to only consider the prediction for the very last token
    next_token_logits = logits[:, -1, :] 

    # C. Apply Sampling (using TOP_P and TEMPERATURE)
    # 1. Apply Temperature (softening/sharpening distribution)
    next_token_logits = next_token_logits / TEMPERATURE

    # # Set logits of the tokens to be removed to a very low value (-inf)
    # # to ensure they are never selected.
    # indices_to_remove = sorted_indices[sorted_indices_to_remove]
    # next_token_logits = next_token_logits.scatter_(-1, indices_to_remove, -float('Inf'))

    # D. Select Next Token (Multinomial Sampling)
    probabilities = torch.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)

    # E. Append Token
    # Append the new token to the running sequence (input_ids)
    input_ids = torch.cat([input_ids, next_token], dim=-1)


# --- 5. Decode and Print Result ---
generated_text = processor.tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("\n## Custom Generated Text ##")
print(generated_text)
# Copyright (c) 2025, Rice-GPT VLM Inference v2
# Based on v1 inference code with ForwardStep pattern

import os
import sys
import copy
import torch
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Megatron-LM 경로 추가 (rice_gpt_inference.py와 동일)
MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/mnt/vlm-data/personal/zro/workspace/Megatron-LM-MMM")
sys.path.insert(0, MEGATRON_PATH)

# Megatron imports
from megatron.core import mpu
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_model, print_rank_0

from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.inference.text_generation.generation import (
    generate_tokens_probs_and_return_on_first_stage,
)

from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

# VLM model provider
from vlm.train.pretrain.pretrain_rice_gpt import model_provider

print("Megatron imports completed")


def add_vlm_inference_args(parser):
    """VLM inference specific arguments."""
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
        "--num-partitions", type=int, default=0, help="Number of partitions"
    )
    group.add_argument("--partition-id", type=int, default=0, help="Partition index")
    group.add_argument("--drop-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--gt-path", type=str, default=None, help="Optional ground truth file"
    )
    
    # Inference specific args
    group.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to input image for inference"
    )
    group.add_argument(
        "--user-prompt",
        type=str,
        default="Describe the image.",
        help="User prompt for VLM inference"
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling"
    )
    
    return parser


class VLMForwardStep(ForwardStep):
    """Rice-GPT VLM Forward Step with StaticInferenceContext."""

    def __init__(
        self,
        images: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        model,
        max_batch_size: int,
        max_sequence_length: int,
        inference_context: StaticInferenceContext,
    ):
        super().__init__(model, max_batch_size)
        self._images = images
        self._image_grid_thw = image_grid_thw
        self.max_sequence_length = max_sequence_length
        self.inference_context = inference_context

    def _forward(self, tokens, position_ids, attention_mask=None):
        return self.model(
            images=self._images,
            image_grid_thw=self._image_grid_thw,
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
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


def generate_vlm_response(
    model,
    image_processor,
    tokenizer,
    image_path: Optional[str],
    conversation: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    mode: str = "image+text",
    temperature: float = 0.1,
    top_k: int = 1,
) -> str:
    """
    Rice-GPT VLM inference function.
    
    Args:
        model: The VLM model
        image_processor: HuggingFace image processor
        tokenizer: HuggingFace tokenizer
        image_path (str or None): Image path. None for text-only mode.
        conversation (list): [{'role': 'user', 'content': [...]}] format conversation.
        max_new_tokens (int): Maximum tokens to generate.
        mode (str): "image+text" or "text-only"
        temperature (float): Sampling temperature
        top_k (int): Top-k sampling
        
    Returns:
        str: Generated response text.
    """
    args = get_args()
    
    # 1. Load image and prepare prompt
    pixel_values = None
    image_grid_thw = None

    messages = copy.deepcopy(conversation)

    if image_path and mode == "image+text":
        try:
            raw_image = Image.open(image_path).convert("RGB")
            for msg in messages:
                if msg["role"] == "user":
                    has_image = any(
                        item.get("type") == "image" for item in msg["content"]
                    )
                    if not has_image:
                        msg["content"].insert(0, {"type": "image", "image": raw_image})
                    break
        except Exception as e:
            print_rank_0(f"Error loading image: {e}")
            return ""

    # 2. Process input (Tokenize & Transform)
    inputs = image_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 3. Prepare Input IDs
    raw_input_ids = inputs["input_ids"]
    if not isinstance(raw_input_ids, torch.Tensor):
        raw_input_ids = torch.tensor(raw_input_ids, dtype=torch.long)

    if raw_input_ids.dim() == 1:
        raw_input_ids = raw_input_ids.unsqueeze(0)

    raw_input_ids = raw_input_ids.cuda()

    batch_size = raw_input_ids.size(0)
    if "attention_mask" in inputs:
        prompt_length = inputs["attention_mask"].sum(dim=1).max().item()
    else:
        prompt_length = raw_input_ids.size(1)

    total_sequence_length = prompt_length + max_new_tokens

    # Prepare image tensors if available
    if "pixel_values" in inputs and mode == "image+text":
        pixel_values = inputs["pixel_values"].cuda().to(torch.bfloat16)

        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"].cuda()

    # 4. Setup Inference Context
    inference_context = StaticInferenceContext(
        max_batch_size=batch_size, max_sequence_length=total_sequence_length
    )

    # 5. Create Token Buffer
    tokens = torch.zeros(
        (batch_size, total_sequence_length),
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    tokens[:, :prompt_length] = raw_input_ids

    lengths = torch.tensor(
        [prompt_length] * batch_size,
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )

    # 6. Define Forward Step
    def custom_forward_step(model, context):
        return VLMForwardStep(
            images=pixel_values,
            image_grid_thw=image_grid_thw,
            model=model,
            max_batch_size=batch_size,
            max_sequence_length=total_sequence_length,
            inference_context=context,
        )

    print_rank_0(f"Generating ({mode})... (Prompt len: {prompt_length}, Max new: {max_new_tokens})")
    if pixel_values is not None:
        print_rank_0(f"pixel_values dtype: {pixel_values.dtype}, shape: {pixel_values.shape}")
    else:
        print_rank_0("pixel_values: None (text-only mode)")
    print_rank_0(f"input_ids dtype: {tokens.dtype}")

    # 7. Run generation
    with torch.no_grad():
        output_tokens, generated_lengths, _, _ = (
            generate_tokens_probs_and_return_on_first_stage(
                model=model,
                inference_context=inference_context,
                forward_step=custom_forward_step,
                tokens=tokens,
                lengths=lengths,
                return_output_log_probs=False,
                top_k=top_k,
                temperature=temperature,
                use_eod_token_for_early_termination=True,
            )
        )

    # 8. Decode result (extract only answer, excluding prompt)
    actual_gen_len = generated_lengths[0].item()
    generated_ids = output_tokens[0, prompt_length:actual_gen_len]
    decoded_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    if len(decoded_text.strip()) < 5:
        print_rank_0("==== debug ====")
        print_rank_0(
            "input: \n"
            + tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)
        )
        print_rank_0(
            "output: \n"
            + tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
        )

    return decoded_text.strip()


def load_model_checkpoint(model):
    """Load model from Megatron checkpoint."""
    args = get_args()
    
    start_ts = datetime.now()
    print_rank_0(f"[{start_ts}] Loading checkpoint from {args.load}")
    load_checkpoint(model, None, None, strict=False)
    end_ts = datetime.now()
    print_rank_0(f"[{end_ts}] Finished loading checkpoint (elapsed: {end_ts - start_ts})")
    
    if isinstance(model, list):
        model = model[0]
    
    return model


# @torch.inference_mode()
def main():
    """Main entry point for Rice-GPT VLM inference."""
    
    # === DEBUGPY Support (rank 0 only) ===
    if os.getenv("DEBUGPY", "0") == "1":
        import debugpy
        port = int(os.getenv("DEBUGPY_PORT", "5678"))
        rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
        if rank == 0:
            try:
                debugpy.listen(("127.0.0.1", port))
                print(f"[debugpy] waiting on 127.0.0.1:{port} (rank={rank}) ...", flush=True)
                debugpy.wait_for_client()
                print("[debugpy] attached.", flush=True)
            except Exception as e:
                print(f"[debugpy] listen failed: {e}", flush=True)
    # =====================================

    # Initialize Megatron
    initialize_megatron(
        extra_args_provider=add_vlm_inference_args,
        ignore_unknown_args=True,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "train_iters": 1,
            "eval_iters": 1,
            "save_interval": 10000,
            "split": "100,0,0",
        },
    )

    args = get_args()
    
    # Get model
    print_rank_0("Loading model structure from model provider...")
    model = get_model(model_provider, wrap_with_ddp=False)
    
    # Load checkpoint
    model = load_model_checkpoint(model)
    model.eval()
    
    # Load processor and tokenizer
    print_rank_0(f"Loading processor from {args.tokenizer_model}...")
    try:
        image_processor = AutoProcessor.from_pretrained(
            args.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model, trust_remote_code=True
        )
    except Exception as e:
        print_rank_0(f"Failed to load processor/tokenizer: {e}")
        traceback.print_exc()
        raise e

    # Prepare inference parameters
    image_path = args.image_path
    user_prompt = args.user_prompt
    max_new_tokens = getattr(args, 'num_tokens_to_generate', 128)
    temperature = getattr(args, 'temperature', 0.1)
    top_k = getattr(args, 'top_k', 1)
    
    if image_path is None:
        # Default test image (rice_gpt_inference.py와 동일)
        image_path = os.path.join(MEGATRON_PATH, "test_images", "dt.jpg")
    
    # Build conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        }
    ]

    # === Image+Text Inference ===
    print_rank_0("\n" + "=" * 50)
    print_rank_0("Image+Text Inference")
    print_rank_0("=" * 50)
    print_rank_0(f"Image: {image_path}")
    print_rank_0(f"Prompt: {user_prompt}")
    print_rank_0("-" * 20)
    
    response_image = generate_vlm_response(
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_path=image_path,
        conversation=conversation,
        max_new_tokens=max_new_tokens,
        mode="image+text",
        temperature=temperature,
        top_k=top_k,
    )
    
    print_rank_0("-" * 20)
    print_rank_0(f"Model Response (image+text): {response_image}")
    print_rank_0("-" * 20)

    # === Text-only Inference (optional comparison) ===
    print_rank_0("\n" + "=" * 50)
    print_rank_0("Text-Only Inference")
    print_rank_0("=" * 50)
    
    text_only_message = "What is Artificial Intelligence? Just answer the question."
    text_only_conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_only_message,
                }
            ],
        }
    ]
    
    print_rank_0(f"Prompt: {text_only_message}")
    print_rank_0("-" * 20)
    
    response_text = generate_vlm_response(
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_path=None,
        conversation=text_only_conversation,
        max_new_tokens=max_new_tokens,
        mode="text-only",
        temperature=temperature,
        top_k=top_k,
    )
    
    print_rank_0("-" * 20)
    print_rank_0(f"Model Response (text-only): {response_text}")
    print_rank_0("-" * 20)

    # Print GPU memory stats
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        stats = torch.cuda.memory_stats()
        print_rank_0(
            f"\nGPU Memory Stats:\n"
            f"  - Allocated: {stats['allocated_bytes.all.peak'] / (1024**3):.1f}GB\n"
            f"  - Reserved: {stats['reserved_bytes.all.peak'] / (1024**3):.1f}GB"
        )

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

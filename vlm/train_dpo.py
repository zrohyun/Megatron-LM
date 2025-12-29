# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import pathlib
import logging
from itertools import takewhile
from typing import Dict, Optional, Sequence, List, Any
import ast

import yaml
import time
import random
import yaml
import math
import re
import torch
import torch.distributed as dist

from PIL import Image, ImageFile

import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, TrainerCallback, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from PIL import Image, ImageFile
from datasets import load_from_disk
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from accelerate import Accelerator
from accelerate.logging import get_logger
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import pad
from accelerate.utils import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed


accelerator = Accelerator()


IGNORE_INDEX = -100  # ID for labels that should be ignored.
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = field(default=384)
    image_split_resolution: int = field(default=384)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    
    
def rank0_print(*args):
    if accelerator.is_main_process:
        print(*args)


def resize_if_too_small(image: Image.Image, min_size: int = 28) -> Image.Image:
    width, height = image.size

    if width >= min_size and height >= min_size:
        return image

    if width < height:
        scale = min_size / width
    else:
        scale = min_size / height

    new_w = max(int(round(width * scale)), min_size)
    new_h = max(int(round(height * scale)), min_size)

    return image.resize((new_w, new_h), Image.BICUBIC)


def _resize_image_max_pixel(image, size_factor=28, min_pixels=None, max_pixels=None):
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, **kwargs):
        elapsed = int(time.time() - self.start_time)
        kwargs["logs"]["steps"] = f"{state.global_step}/{state.max_steps}"
        kwargs["logs"]["elapsed"] = elapsed
        # deepspeed.comm.log_summary()
        return super().on_log(args, state, control, **kwargs)
    

class VLMDPOTrainer(DPOTrainer):
    @staticmethod
    def process_row(
        features: dict[str, str],
        processing_class,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer  # the processing class is a processor
        
        if "video" in features:
            raise NotImplementedError("Video processing in DPO dataset is not implemented yet.")

        # prompt
        prompt_messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': features['prompt']
            }
        ]
        prompt_chat_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        def _get_completion_chat_text(content: str):
            _messages = prompt_messages + [
                {
                    'role': 'assistant',
                    'content': content
                }
            ]
            _chat_text = processor.apply_chat_template(
                _messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if _chat_text.endswith("\n"):
                _chat_text = _chat_text[:-1]
                
            return _chat_text
        
        chosen_chat_text = _get_completion_chat_text(features['chosen'])
        prompt_chat_text = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_chat_text, chosen_chat_text, strict=False)))
        chosen_chat_text = chosen_chat_text[len(prompt_chat_text) :]
        
        rejected_chat_text = _get_completion_chat_text(features['rejected'])
        prompt_chat_text = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_chat_text, rejected_chat_text, strict=False)))
        rejected_chat_text = rejected_chat_text[len(prompt_chat_text) :]
        
        # image
        image_count = len(features.get("image", [])) if "image" in features else 0
        
        if image_count > 0:
            prompt_chat_text = prompt_chat_text.replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            
        has_images = image_count > 0
        if has_images:
            # image_folder = self.data_args.image_folder
            # images = [Image.open(os.path.join(image_folder, image_file)).convert("RGB") for image_file in data_dict["image"]]
            resized_images = [resize_if_too_small(img, min_size=30) for img in features["image"]]
            proc_out = processor(
                text=prompt_chat_text,
                images=resized_images,
                return_tensors="pt",
            )
            pixel_values = proc_out["pixel_values"]     # [N_patches_height * M_patches_width, in_channels * patch_size**2]
            image_grid_thw = proc_out["image_grid_thw"]  # [T, N_patches_height, M_patches_width,]
        else:
            proc_out = processor(
                text=prompt_chat_text,
                return_tensors="pt",
            )
                        
            pixel_values = None
            image_grid_thw = None
        
        prompt_input_ids = proc_out["input_ids"][0]
        chosen_input_ids = tokenizer(chosen_chat_text, add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(rejected_chat_text, add_special_tokens=False)["input_ids"]

        vision_start_id, image_token_id, vision_end_id = processor.tokenizer.convert_tokens_to_ids(
            [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
        )

        n_image_tokens = (prompt_input_ids == image_token_id).sum().item()
        possible_image_tokens = 8000
        if has_images and n_image_tokens > possible_image_tokens:
            image_grid_thw_prod =  image_grid_thw.prod(dim=-1)
            image_proportion = image_grid_thw_prod / image_grid_thw_prod.sum()
            new_num_image_token = torch.floor(possible_image_tokens * image_proportion)

            new_image_list = []
            for _num, _img in zip(new_num_image_token, sample.images):
                _max_pixel = 4 * _num * (processor.image_processor.patch_size ** 2)
                _min_pixels = processor.image_processor.min_pixels
                assert _max_pixel > _min_pixels # 이미지 줄이는 게 불가능한 경우
                new_image_list.append(_resize_image_max_pixel(image=_img, min_pixels=_min_pixels, max_pixels=_max_pixel))
                
            proc_out = processor(
                text=prompt_chat_text,
                images=new_image_list,
                return_tensors="pt",
            )

            # 이미지 re-프로세싱
            prompt_input_ids = proc_out["input_ids"][0]                         # [L]
            image_grid_thw = proc_out["image_grid_thw"]
            pixel_values = proc_out["pixel_values"]          # [num_tiles, C, H, W] or None

        if n_image_tokens > 0:
            assert image_grid_thw.prod(dim=-1).sum() / 4 == n_image_tokens, features
            
        # Truncate prompt and completion sequences
        # if max_prompt_length is not None:
        #     prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        return output
    
    
@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0] and examples[0]["pixel_values"] is not None:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0] and examples[0]["pixel_attention_mask"] is not None:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0] and examples[0]["pixel_values"] is not None:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0] and examples[0]["pixel_attention_mask"] is not None:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        if "token_type_ids" in examples[0]:
            token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
            output["token_type_ids"] = pad(token_type_ids, padding_value=0, padding_side="left")

        return output


def train():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    parser = TrlParser((DataArguments, ScriptArguments, DPOConfig, ModelConfig))
    data_args, script_args, training_args, model_args = parser.parse_args_and_config()
    
    training_args.save_safetensors = True
    training_args.attn_implementation = "flash_attention_2"

    rank0_print("Inspecting experiment hyperparameters:\n")
    rank0_print(f"model_args = {vars(model_args)}\n\n")
    rank0_print(f"data_args = {vars(data_args)}\n\n")
    rank0_print(f"training_args = {vars(training_args)}\n\n")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    
    ################
    # Model & Processor
    ################
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        low_cpu_mem_usage=False,
        dtype=dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    model.config.model_type = "llava"   # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES 에 있는 모델 타입이면 아무거나 ok. 
    
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        ref_model.config.model_type = "llava"
    else:
        ref_model = None
        
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    ################
    # Dataset
    ################
    train_dataset = load_from_disk(data_args.data_path, keep_in_memory=False).shuffle(seed=2275)
    data_collator = DataCollatorForPreference(pad_token_id=processor.tokenizer.pad_token_id)
    
    ################
    # Training
    ################
    # training_args.chunk_dpo_loss = False
    # if training_args.use_liger_kernel:
    #     training_args.use_liger_kernel = False
    #     training_args.chunk_dpo_loss = True

    trainer = VLMDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=None,
        processing_class=processor,
        callbacks=[CustomCallback],
    )

    trainer.train()
    
    trainer.save_model(training_args.output_dir)

    # model.config.use_cache = True
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()

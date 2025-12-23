""" Qwen2VLTaskEncoder class."""
import math
import re
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, TypeVar

import numpy as np
import torch
from megatron.energon import CaptioningSample, VQASample
from megatron.energon.flavors.webdataset import VideoData
from megatron.energon.task_encoder.base import stateless
from PIL import Image
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor
from typing_extensions import override

from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    PinMemoryMixin,
    Sample,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeSample, CrudeWebdataset
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.rng import SystemRng
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BlendDataset,
    ConcatDataset,
    BatchDataset,
    EpochizeDataset,
    GroupBatchDataset,
    LimitDataset,
    LogSampleDataset,
    MapDataset,
    PackingDataset,
    ShuffleBufferDataset,
)
# from .batch_dataset import BatchDataset
from megatron.energon.wrappers.repeat_dataset import RepeatDataset

from .flavors import MultiMixQASample, ImageConvSample
from .length_sort_dataset import LengthPoolSortDataset
from .task_encoder import (ImageTaskBatchPacked, ImageTaskSample,
                           ImageTaskSamplePacked, TaskEncoder, conv_to_hf_messages)

from vlm.utils import constants, get_chat_template


T = TypeVar("T")
V = TypeVar("V")
T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_raw_batch = TypeVar("T_raw_batch")
T_batch = TypeVar("T_batch")


IGNORE_INDEX = -100  # ID for labels that should be ignored.
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]

SYSTEM_PROMPT = "You are a helpful assistant."


def generate_generic_thinking_prompt(input_type: str = "multimodal"):
    """
    input_type: "multimodal" | "text" | "common"
    """
    assert input_type in ("multimodal", "text", "common")

    # 1. 도입부 (Starters)
    starters = [
        "First,", "Okay,", "Well,", "To begin with,", "Starting off,",
        "Let's see,", "Let me see,", "Basically,", "Process:", "Thought:"
    ]

    # 2. 행동 (Actions)
    common_actions = [
        "I will carefully analyze", "I need to examine", "I'll break down",
        "let's review", "checking", "investigating"
    ]

    visual_actions = [
        "I need to scan", "let's inspect", "I am looking at", "let's observe"
    ]

    text_actions = [
        "I will read", "I need to parse", "I'll interpret", "let's analyze"
    ]

    # 3. 대상 (Targets)
    visual_targets = [
        "the provided image", "this visual input", "the picture context",
        "the pixel data", "the image content", "the visual details",
        "what is shown here", "the overall scene", "the visible elements",
        "the image"
    ]

    text_targets = [
        "the given text", "the user's question", "the input prompt",
        "the textual content", "the sentence", "the provided description",
        "the written information", "the query"
    ]

    common_targets = [
        "the input", "the problem", "the task",
        "the given information", "the context",
        "the request", "what is provided"
    ]

    # 4. 목표 (Goals)
    goals = [
        "to find the correct answer.", "to derive a proper response.",
        "to understand the intent.", "to capture the key information.",
        "to answer the user's question.", "to address the query.",
        "to form a conclusion.", "to determine the result."
    ]

    # --- modality 선택 ---
    if input_type == "text":
        actions = common_actions + text_actions
        targets = text_targets

    elif input_type == "common":
        actions = common_actions
        targets = common_targets

    else:  # multimodal (default)
        actions = common_actions + visual_actions
        targets = visual_targets

    # 문장 구조 1: Full Sentence
    sentence_a = (
        f"{random.choice(starters)} "
        f"{random.choice(actions)} "
        f"{random.choice(targets)} "
        f"{random.choice(goals)}"
    )

    # 문장 구조 2: Short & Direct
    sentence_b = f"{random.choice(actions).capitalize()} {random.choice(targets)}."

    # 문장 구조 3: Self-Talk style
    sentence_c = f"Analyzing {random.choice(targets)}... {random.choice(goals)}"

    return random.choice([sentence_a, sentence_b, sentence_c])


def get_stateless(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is stateless."""
    return getattr(fn, "__stateless__", False)


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


@dataclass
class Qwen2VLImageTaskSample(ImageTaskSample):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, image_grid_thw: str, video_grid_thw=None, **kwargs):
        super().__init__(**kwargs)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskSamplePacked(ImageTaskSamplePacked):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**asdict(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskBatchPacked(ImageTaskBatchPacked):
    """ An image task sample with a grid of tokens and their corresponding pixel values."""
    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**asdict(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


class Qwen2VLTaskEncoder(TaskEncoder):
    """A simple task encoder for VLMs."""

    def __init__(self, args):
        super().__init__()
        if args.training_phase in ['sft']:
            self.chat_template = get_chat_template()
        self.processor = AutoProcessor.from_pretrained(self.args.hf_tokenizer_path, trust_remote_code=True)

        if args.image_resolution:
            setattr(self.processor, 'image_resolution', args.image_resolution)
        # video
        self.frame_min_pixels = args.frame_min_pixels
        self.frame_max_pixels = args.frame_max_pixels
        self.video_max_pixels = args.video_max_pixels
        self.fps = args.fps
        self.fps_min_frames = args.fps_min_frames
        self.fps_max_frames = args.fps_max_frames
        # image
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels
        
        self.use_think_template = getattr(args, "use_think_template", False)

    def _reisize_video(self, vision: VideoData, image_factor=28, frame_factor=2):
        """ Resize video: frame number, height, width """
        total_frames = len(vision.frames)
        video_fps = vision.info['video_fps']
        vision.info['fps'] = self.fps
        vision.info['min_frames'] = self.fps_min_frames
        vision.info['max_frames'] = self.fps_max_frames

        # resize frame
        nframes = smart_nframes(vision.info, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        video = vision.frames[idx]
        # resize height, width
        nframes, _, height, width = video.shape
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=int(self.frame_min_pixels * 1.05),
            max_pixels=min(self.frame_max_pixels, self.video_max_pixels / nframes * frame_factor),
        )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        return video

    def _resize_image(self, image, size_factor=28):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _resize_image_max_pixel(self, image, size_factor=28, max_pixel=None):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor,
            min_pixels=self.min_pixels,
            max_pixels=max_pixel,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _process(self, image, text):
        """" Process the data to get the model's input """
        inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs['input_ids'][0]
        attn_mask = inputs['attention_mask'][0].logical_not()
        image_grid_thw = None
        pixel = []
        if image is not None:
            image_grid_thw = inputs['image_grid_thw'] # [t,h,w]
            pixel = [inputs['pixel_values']] # [hw, 2*3*14*14]

        target = input_ids.clone()
        vision_start_id, img_pad_id, vision_end_id = self.tokenizer.convert_tokens_to_ids([
            VISION_TAGS[0],
            IMAGE_TOKEN,
            VISION_TAGS[1]
        ])
        target[target == vision_start_id] = IGNORE_INDEX
        target[target == img_pad_id] = IGNORE_INDEX
        target[target == vision_end_id] = IGNORE_INDEX

        return input_ids, target, pixel, image_grid_thw, attn_mask

    def process_sft_vqa(self, context, answer, image):
        """ process the data for sft vqa """
        text = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': context
            }, {
                'role': 'assistant',
                'content': answer
            }],
            tokenize=False
        ).replace(
            "<image>", IMAGE_TOKEN_WITH_TAGS
        )
        if text[-1] == '\n':
            text = text[:-1]
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answer = self.tokenizer.tokenize(answer)
        target[-len(answer) - 1: -1] = torch.tensor(answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def process_sft_qa(self, messages: list, system: str, raw_video: list, raw_image: list):
        """ process the data for sft qa """
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []

        if raw_image is not None:
            for i in raw_image:
                image.append(self._resize_image(i))

        if raw_video is not None:
            for v in raw_video:
                video.append(self._reisize_video(v))

        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image if image is not None else [],
            video if raw_video is not None else [],
            self.processor
        )
        # assert raw_image is not None, f'No image found in {messages}' 确实有纯文本对话
        if raw_video is not None:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs["pixel_values_videos"]]
        if raw_image is not None:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]

        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )
        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        return input_ids, target, attn_mask, pixel_values_images, image_grid_thw, \
                    pixel_values_videos, video_grid_thw

    def encode_captioning(self, sample: CaptioningSample) -> ImageTaskSample:
        """Encode CaptioningSample."""
        """Preprocessing function for datasets like COCO, containing image-caption pairs.
        See Energon codebase for more details on CaptioningSample.
        https://github.com/NVIDIA/Megatron-Energon/blob/develop/src/megatron/energon/flavors/captioning.py
        """

        # assert self.args.training_phase == constants.TrainingPhase.PRETRAIN, "Only support PRETRAIN phase"

        text = IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.tokenizer.eos_token

        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} thw {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_vqa4packing(self, sample: VQASample) -> ImageTaskSample:
        """Encode VQASample in Qwen2VL style."""
        
        # 构建 chat_template
        text = self.processor.apply_chat_template(
            [{
                'role': 'user',
                'content': sample.context
            }, {
                'role': 'assistant',
                'content': sample.answers
            }],
            tokenize=False
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)        

        if text[-1] == '\n':
            text = text[:-1]
            pass  
            
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1: -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]
        # print(target[-1])
        
        num_tiles = [len(image_grid_thw)]
        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"
            
        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )
    
    def encode_captioning_image_conv(self, sample: ImageConvSample) -> ImageTaskSample:
        # Text-Only X
        conversations = sample.conversations
        caption = conversations[1]['value']
        
        text = IMAGE_TOKEN_WITH_TAGS + caption + self.tokenizer.tokenizer.eos_token

        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.images, text)
        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} thw {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_vqa_image_conv(self, sample: ImageConvSample) -> ImageTaskSample:
        messages = conv_to_hf_messages(sample.conversations)

        if len(messages) == 0:
            raise ValueError("conversations is empty")

        first_role = messages[0]["role"]
        if first_role == "system":
            pass
        elif first_role == "user":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        else:
            raise ValueError("First message must be system or user role.")
        
        if self.use_think_template:
            for m_idx, m in enumerate(messages):
                if m['role'] == "assistant" and "<think>" not in m['content']:
                    if m_idx > 1 and messages[m_idx - 1]['role'] == "user":
                        if "<image>" in messages[m_idx - 1]['content']:
                            input_type = "multimodal"
                        else:
                            input_type = "text"
                    else:
                        input_type = "common"
                    
                    m['content'] = f"<think>{generate_generic_thinking_prompt(input_type)}</think>" + m['content']

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        if sample.image_count > 0:
            chat_text = chat_text.replace("<image>", IMAGE_TOKEN_WITH_TAGS)

        # 필요하면 마지막 개행 제거 (선택)
        if chat_text.endswith("\n"):
            chat_text = chat_text[:-1]
            
        has_images = sample.image_count > 0 and len(sample.images) > 0
        if has_images:
            resized_images = [resize_if_too_small(img, min_size=30) for img in sample.images]
            proc_out = self.processor(
                text=chat_text,
                images=resized_images,
                return_tensors="pt",
            )
            pixel_values = proc_out["pixel_values"]     # [N_patches_height * M_patches_width, in_channels * patch_size**2]
            image_grid_thw = proc_out["image_grid_thw"]  # [T, N_patches_height, M_patches_width,]
        else:
            proc_out = self.processor(
                text=chat_text,
                return_tensors="pt",
            )
            # make dummpy image inputs
            grid_h, grid_w = 24, 26
            patch_size = self.processor.image_processor.patch_size
            
            n_tokens = 1 * grid_h * grid_w
            dim = 3 * patch_size * patch_size
            pixel_values = torch.zeros(
                (n_tokens, dim), dtype=torch.float32
            )
            image_grid_thw = torch.tensor(
                [[1, grid_h, grid_w]], dtype=torch.long
            )

        input_ids = proc_out["input_ids"][0]                         # [L]
        attention_mask = proc_out["attention_mask"][0].logical_not()

        tokens = input_ids.tolist()
        L = len(tokens)

        unmask_tokens_ids = self.processor.tokenizer.convert_tokens_to_ids(["<|role_start|>", "<|role_end|>", "\n"])
        role_start_id, role_end_id, newline_id = unmask_tokens_ids

        vision_start_id, image_token_id, vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(
            [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
        )

        num_image_token = sum(input_ids == image_token_id)
        possible_image_tokens = self.args.possible_image_tokens
        if has_images and num_image_token > possible_image_tokens:
            image_grid_thw_prod =  image_grid_thw.prod(dim=-1)
            image_proportion = image_grid_thw_prod / image_grid_thw_prod.sum()
            new_num_image_token = torch.floor(possible_image_tokens * image_proportion)
            
            new_image_list = []
            for _num, _img in zip(new_num_image_token, sample.images):
                _max_pixel = 4 * _num * (self.processor.image_processor.patch_size ** 2)
                assert _max_pixel > self.min_pixels # 이미지 줄이는 게 불가능한 경우
                new_image_list.append(self._resize_image_max_pixel(image=_img, max_pixel=_max_pixel))
            
            proc_out = self.processor(
                text=chat_text,
                images=new_image_list,
                return_tensors="pt",
            )

            # 이미지 re-프로세싱
            input_ids = proc_out["input_ids"][0]                         # [L]
            attention_mask = proc_out["attention_mask"][0].logical_not()
            image_grid_thw = proc_out["image_grid_thw"]
            pixel_values = proc_out["pixel_values"]          # [num_tiles, C, H, W] or None

            tokens = input_ids.tolist()
            L = len(tokens)
            
            assert image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"
            assert self.args.seq_length >= len(input_ids)
        
        assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} [{sample.__subflavors__}] input length {len(input_ids)}"
        targets = torch.full_like(input_ids, IGNORE_INDEX)

        # assistant 블록(span)을 찾아 해당 구간만 label로 채우기
        # 여기서는 다음 <|role_start|>가 나오기 전까지를 하나의 assistant 답변으로 본다.
        assistant_token_ids = self.processor.tokenizer(
            "assistant", add_special_tokens=False
        )["input_ids"]  # [710, 9370]
                
        i = 0
        while i < L:
            if tokens[i] == role_start_id:
                j = i + 1
                if (
                    j + len(assistant_token_ids) < L
                    and tokens[j : j + len(assistant_token_ids)] == assistant_token_ids
                    and j + len(assistant_token_ids) < L
                    and tokens[j + len(assistant_token_ids)] == role_end_id
                ):
                    # header 구간: <|role_start|> assistant <|role_end|>
                    header_end = j + len(assistant_token_ids) + 1

                    # 답변 끝 위치: 다음 role_start 또는 시퀀스 끝
                    k = header_end
                    while k < L and tokens[k] != role_start_id:
                        k += 1

                    targets[header_end:k] = input_ids[header_end:k]

                    i = k
                    continue

            i += 1

        # # <|role_start|>, <|role_end|> 는 모든 role 에서 targets 에 포함
        # # newline(\n) 은 assistant 에서만 포함
        # for idx, tid in enumerate(tokens):
        #     if tid == role_start_id or tid == role_end_id:
        #         targets[idx] = input_ids[idx]
        #     elif tid == newline_id and targets[idx] != IGNORE_INDEX:
        #         targets[idx] = input_ids[idx]
                
        # # <|role_start|>, <|role_end|>, newline(\n) 를 targets 에 포함
        # for idx, tid in enumerate(tokens):
        #     if tid == role_start_id or tid == role_end_id or tid == newline_id :
        #         targets[idx] = input_ids[idx]

        for idx, tid in enumerate(tokens):
            if tid in (vision_start_id, image_token_id, vision_end_id):
                targets[idx] = IGNORE_INDEX
            
        num_tiles = [len(image_grid_thw)]
        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"
        
        n_image_tokens = (input_ids == image_token_id).sum().item()
        if n_image_tokens > 0:
            assert image_grid_thw.prod(dim=-1).sum() / 4 == n_image_tokens, sample
        
        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=[pixel_values],
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=targets,
            attn_mask=attention_mask,
            total_len=len(input_ids),
        )

    def encode_multi_vid_qa(self, sample: VQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            input_ids, target, attn_mask, imgs, image_grid_thw, video, video_grid_thw = \
                        self.process_sft_qa(sample.messages, sample.system, sample.video, None)
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                    f"{sample.__key__} grid_thw: {video_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=video,
            video_grid_thw=video_grid_thw,
            num_tiles=[len(video_grid_thw)],
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = []

            input_ids, target, attn_mask, imgs, image_grid_thw, pixel_values_videos, video_grid_thw = \
                        self.process_sft_qa(sample.messages, sample.system, sample.video, sample.image)
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)]
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)]
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")


        if len(input_ids) == 0:
            raise ValueError(f"input_ids is empty in {sample.__key__}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                        f"{sample.__key__} grid_thw: {video_grid_thw}"
        elif sample.image is not None:
            assert image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, \
                        f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_vaq(self, sample: VQASample) -> ImageTaskSample:
        """Encode pretrain sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace(
                    "<image>",
                    IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        elif self.args.training_phase == constants.TrainingPhase.SFT:


            if len(sample.answers) < 1:
                raise ValueError("sample.answers < 1!")

            # Add image resize check for PIL.Image
            if sample.image is not None:

                img_arr = np.array(sample.image)
                if np.sum(img_arr) == 0:
                    raise ValueError("Image pixels are all zero!")

            # Truncate answer to the last full sentence if it exceeds the max length.
            max_answer_length = self.args.training_rice_vl_max_answer_length
            if len(sample.answers) > max_answer_length:
                original_length = len(sample.answers)

                # Perform a preliminary cut at the maximum allowed length.
                preliminary_cut = sample.answers[:max_answer_length]

                # Clean up trailing punctuation and whitespace from the preliminary cut
                cleaned_cut = preliminary_cut.rstrip('.。 \t\n')

                # Find the last occurrence of a sentence-ending punctuation mark followed by a space or the end of the string.
                # This pattern looks for sentence enders (. or 。)
                sentence_enders_pattern = r'[.。]'

                # Find all matches and get the end position of the last match
                matches = list(re.finditer(sentence_enders_pattern, cleaned_cut))

                if matches:
                    # Get the end position of the last match
                    last_end_index = matches[-1].end()
                    # Truncate at the end of the last full sentence.
                    sample.answers = cleaned_cut[:last_end_index]
                else:
                    # Fallback to a hard cut of the original preliminary string if no sentence ender is found.
                    sample.answers = preliminary_cut

                print(
                    f"Answer truncated to a full sentence. "
                    f"Original length: {original_length}, New length: {len(sample.answers)}"
                )

            text = self.processor.apply_chat_template(
                [{
                    'role': 'user',
                    'content': sample.context
                }, {
                    'role': 'assistant',
                    'content': sample.answers
                }],
                tokenize=False
            ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            if text[-1] == '\n':
                text = text[:-1]
            input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
            target = torch.ones_like(input_ids) * IGNORE_INDEX
            answers = self.tokenizer.tokenize(sample.answers)
            target[-len(answers) - 1: -1] = torch.tensor(answers)
            target[-1] = input_ids[-1]
            # print(target[-1])
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def process_samples_grid(self, samples):
        """ concat grid_thw for image and video """
        image_grid_thw = [x.image_grid_thw for x in samples if x.image_grid_thw is not None]
        video_grid_thw = [x.video_grid_thw for x in samples if x.video_grid_thw is not None]

        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw).to(dtype=torch.int32)
        else:
            image_grid_thw = None

        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw).to(dtype=torch.int32)
        else:
            video_grid_thw = None

        return image_grid_thw, video_grid_thw

    @override
    @stateless
    def pack_selected_samples(self, samples: List[Qwen2VLImageTaskSample]) -> List[Qwen2VLImageTaskSamplePacked]:
        """ Pack selected samples into one big sample."""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskSamplePacked(
            super().pack_selected_samples(samples),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )

    @override
    def batch(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> Qwen2VLImageTaskBatchPacked:
        """ Batch samples together """
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskBatchPacked(
            super().batch(samples),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )

    @override
    def process_images(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """" Process the data to get the model's input """
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def process_videos(self, samples: List[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """" Process the data to get the model's input """
        pixel_values_videos = [pixel_values_video for s in samples if s.pixel_values_videos is not None \
                for pixel_values_video in s.pixel_values_videos]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def build_train_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDatasetFactory[T_sample], Union[float, int, None]]],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
        blend_mode: DatasetBlendMode = DatasetBlendMode.NONE,
        repeat: bool = True,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""
        

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset) for dataset, _ in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [
            rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]
        ]

        if repeat:
            inner_datasets = [  
                (
                    RepeatDataset(
                        dataset.build(worker_rotation_offset=worker_rotation_offset),
                        worker_config=worker_config,
                    ),
                    1.0 if weight is None else float(weight),
                )
                for (dataset, weight), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
            ]
        else:
            assert blend_mode in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ) and all(
                isinstance(repetitions, int) for _dataset, repetitions in datasets
            ), "If repeat is False, the datasets must be repeated with integer weights."
            inner_datasets = [
                (
                    (
                        dataset.build(worker_rotation_offset=worker_rotation_offset)
                        if repetition is None or repetition == 1
                        else RepeatDataset(
                            dataset.build(worker_rotation_offset=worker_rotation_offset),
                            repeats=int(repetition),
                            worker_config=worker_config,
                        )
                    ),
                    len(dataset) * (1 if repetition is None else int(repetition)),
                )
                for (dataset, repetition), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
            ]

        if len(inner_datasets) > 1:
            # The worker offset for each dataset is the cumsum of the dataset lengths, but modulo the
            # global number of workers.
            dataset = BlendDataset(
                *inner_datasets,
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = inner_datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)

         # BatchDataset 으로 넘기기 전에 pooled 정렬 단계를 넣는다
        if getattr(self.args, "length_sort_pool_size", 0) and self.args.length_sort_pool_size > 0:
            dataset = LengthPoolSortDataset(
                dataset,
                pool_size=self.args.length_sort_pool_size,
                key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens"))),
                ascending=not getattr(self.args, "length_sort_desc", False),
                worker_config=worker_config,
            )
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, mode="train", worker_config=worker_config)
        return dataset
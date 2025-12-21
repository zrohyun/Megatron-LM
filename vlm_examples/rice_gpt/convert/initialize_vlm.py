#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Dict
from os.path import dirname
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict

import torch
import torch.distributed as dist

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.dist_checkpointing import load as dist_load
from megatron.core.dist_checkpointing import save as dist_save
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args, initialize_megatron
from megatron.training import print_rank_0

from megatron.core.models.wbl_moe_gpt.model import (
    get_wbl_moe_gpt_decoder_block_spec,
    get_wbl_moe_gpt_layer_with_transformer_engine_spec,
)
from vlm.models.vision.rice_vit_spec import (
    get_vit_layer_with_transformer_engine_spec, 
    get_vit_layer_with_local_spec
)
from vlm.models.rice_gpt.rice_gpt_spec import (
    get_vision_projector_layer_with_spec
)
from vlm.models.rice_gpt.rice_gpt_model import RiceGPTModel
from vlm.models.rice_gpt.rice_gpt_provider import rice_gpt_model_provider
from vlm.models.rice_gpt.rice_gpt_config import get_vision_config, get_vision_projection_config


def add_state_dict_args(parser):
    """Static inference arguments."""
    group = parser.add_argument_group(title="stage-dict")
    
    group.add_argument("--language-model-path", type=str, help="Path to language model.")
    group.add_argument("--vision-model-path", type=str, help="Path to vision model.")
    group.add_argument("--vision-patch-path", type=str, help="Path to vision patch.")

    group.add_argument("--save-ckpt-path", type=str, help="Path to save checkpoint.")
    
    group.add_argument("--model-name", default="rice-gpt-7b-a1b")
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument('--trainable-modules', default=['all'], nargs='*',
                    help='choices: all, language_model, vision_projection, vision_model, '
                        'language_expert_linear, vision_expert_linear')

    return parser


def load_megatron_checkpoint(load_path):
    """ load ckpt """
    state_dict = []
    sub_dirs = sorted([x for x in os.listdir(load_path) if x.startswith("mp_rank")])
    last_dir = sub_dirs[-1].split('_')
    if len(last_dir) == 4:
        tp = int(last_dir[-2]) + 1
        pp = int(last_dir[-1]) + 1
        for p in range(pp):
            state_dict.append([])
            for t in range(tp):
                checkpoint_name = f"mp_rank_{t:02d}_{p:03d}/model_optim_rng.pt"
                ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location='cpu', weights_only=False)
                state_dict[p].append(ckpt)
        return state_dict
    else:
        for t in range(len(sub_dirs)):
            checkpoint_name = f"mp_rank_{t:02d}/model_optim_rng.pt"
            ckpt = torch.load(os.path.join(load_path, checkpoint_name), map_location='cpu', weights_only=False)
            state_dict.append(ckpt)
        return state_dict


@torch.no_grad()
def load_llm_into_vlm_language_model(vlm: RiceGPTModel, llm_checkpoint_dir: Path):
    lang_sharded = vlm.language_model.sharded_state_dict(prefix="")
    dist_load(
        sharded_state_dict=lang_sharded, 
        checkpoint_dir=str(llm_checkpoint_dir), 
        strict="log_all", # "log_all"
    )
    
    
def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value
        
        
@torch.no_grad()
def load_vit_into_vlm(vlm: RiceGPTModel, vision_model_path: Path, vision_patch_path: Path):
    vit = load_megatron_checkpoint(vision_model_path)
    vit_state_dict = vit[0]['model']
    
    patch = load_megatron_checkpoint(vision_patch_path)
    patch_state_dict = patch[0]['model']
    
    for k, v in patch_state_dict.items():
        assert k not in vit_state_dict
        vit_state_dict[k] = v
    
    def strip_vision_prefix(state_dict: dict, prefix="vision_model."):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict
        
    vision_state_dict = strip_vision_prefix(vit_state_dict)
    
    missing, unexpected = vlm.vision_model.load_state_dict(vision_state_dict, strict=True)
    if len(missing) > 0:
        print("[HF-ViT->Rice] missing keys:")
        for k in missing:
            print("  -", k)

    if len(unexpected) > 0:
        print("[HF-ViT->Rice] unexpected keys:")
        for k in unexpected:
            print("  -", k)
            
    # known_missing = {}  # 전부 다 로드되어야 정상, (missing, unexpected 모두 없어야 함)
    # real_missing = [k for k in missing if k not in known_missing]
    # if real_missing:
    #     raise RuntimeError(f"[HF-ViT->Rice] 예상하지 못한 missing keys: {real_missing}")
    # if unexpected:
    #     raise RuntimeError(f"[HF-ViT->Rice] 예상하지 못한 unexpected keys: {unexpected}")
    
    
def write_latest_tracker(dst_root: Path, dst_iter: int):
    (dst_root / "latest_checkpointed_iteration.txt").write_text(f"{dst_iter}\n", encoding="utf-8")


def main():
    initialize_megatron(
        extra_args_provider=add_state_dict_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()
    
    vlm = rice_gpt_model_provider(True, True, add_encoder=True, add_decoder=True)
    vlm.eval()
    
    # TODO: 추후에 logits 검증 코드 작성
    
    print("origin lang norm:", vlm.language_model.embedding.word_embeddings.weight.data.sum())
    import time
    time.sleep(1)

    load_llm_into_vlm_language_model(vlm, args.language_model_path)

    print("lang norm:", vlm.language_model.embedding.word_embeddings.weight.data.sum())
    
    load_vit_into_vlm(vlm, args.vision_model_path, args.vision_patch_path)
    
    dst_iter = 1
    dst_iter_dir = Path(args.save_ckpt_path) / f"iter_{dst_iter:07d}"
    dst_iter_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # full_state = get_model_state_dict(vlm, exclude_rng_states=True)
        # full_sharded = ShardedStateDict(full_state)

        state_dict = {}
        state_dict["args"] = args
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = dst_iter
        state_dict["model"] = vlm.sharded_state_dict()
        
        dist_save(
            sharded_state_dict=state_dict,
            checkpoint_dir=str(dst_iter_dir),
            # checkpoint_name="model",
            # rank0_only=False,
        )
    
    write_latest_tracker(Path(args.save_ckpt_path), dst_iter)
    
    if torch.distributed.get_rank() == 0:
        # Megatron checkpoint 포맷에 맞는 최소한의 stub
        ckpt_state = {
            "args": args,                    # get_args()로 받은 Arguments 객체
            "checkpoint_version": 3.0,       # 보통 3.0 사용
            "iteration": dst_iter,           # 우리가 지정한 iteration 번호
            "model": {},                     # dist-ckpt 사용 시 실제 파라미터는 여기 안씀
            "optimizer": None,
            "lr_scheduler": None,
            "skipped_steps": 0,
            "rng_state": {},                 # 초기 pretrain이면 비워둬도 됨
        }

        torch.save(ckpt_state, dst_iter_dir / "model_optim_rng.pt")

    # dst_iter_name = "release"
    # dst_iter_dir = Path(args.save_ckpt_path) / dst_iter_name / "mp_rank_00"
    # dst_iter_dir.mkdir(parents=True, exist_ok=True)
    
    # if torch.distributed.get_rank() == 0:
    #     with torch.no_grad():
    #         state_dict = {}
    #         state_dict["args"] = args
    #         state_dict["checkpoint_version"] = 3.0
    #         state_dict["iteration"] = 0
    #         state_dict["model"] = {k: v.cpu() if v is not None else None for k, v in vlm.state_dict().items()}
            
    #         torch.save(state_dict, dst_iter_dir / "model_optim_rng.pt")
            
    # write_latest_tracker(Path(args.save_ckpt_path), dst_iter_name)
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

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
from vlm.models.rice_gpt.rice_gpt_config import get_vision_config, get_vision_projection_config


def init_distributed_singleton():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )


def build_vlm(args) -> RiceGPTModel:
    config = core_transformer_config_from_args(args)

    assert args.transformer_impl == "transformer_engine"

    language_config = deepcopy(config)
    vision_config = deepcopy(config)
    vision_projection_config = deepcopy(config)

    for cfg in (language_config, vision_config, vision_projection_config):
        cfg.tensor_model_parallel_size = 1
        cfg.pipeline_model_parallel_size = 1
        cfg.expert_model_parallel_size = 8
        cfg.context_parallel_size = 1

    language_transformer_layer_spec = get_wbl_moe_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        normalization=config.normalization,
        qk_l2_norm=False,
        vp_stage=None,
    )

    for k, v in asdict(get_vision_config("", "rice")).items():
        setattr(vision_config, k, v)

    for k, v in asdict(get_vision_projection_config("")).items():
        setattr(vision_projection_config, k, v)

    vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

    # Make sure vision model pipeline parallel size is not inherited from the language model pipeline parallel size.
    vision_config.pipeline_model_parallel_size = 1
    vision_projection_config.pipeline_model_parallel_size = vision_config.pipeline_model_parallel_size

    # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
    vision_config.first_pipeline_num_layers = vision_config.last_pipeline_num_layers = None

    # if vision_projection_config.normalization:
    #     vision_projection_layer_spec = get_norm_multimodal_projector_module_spec_te().submodules
    # else:
    #     vision_projection_layer_spec = get_multimodal_projector_module_spec(use_te=True).submodules
    vision_projection_layer_spec = get_vision_projector_layer_with_spec()

    vision_config.recompute_granularity = None
    vision_config.recompute_method = None
    vision_config.recompute_num_layers = None

    vision_projection_config.recompute_granularity = None
    vision_projection_config.recompute_method = None
    vision_projection_config.recompute_num_layers = None

    # TODO: Vision model and projection do not use SP/CP yet.
    vision_config.sequence_parallel = False
    vision_config.context_parallel_size = 1
    vision_config.tp_comm_overlap = False

    vision_projection_config.sequence_parallel = False
    vision_projection_config.context_parallel_size = 1
    vision_projection_config.tp_comm_overlap = False

    vlm = RiceGPTModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        allow_missing_vision_projection_checkpoint=True,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy
    )
    return vlm


def load_llm_into_vlm_language_model(vlm: RiceGPTModel, llm_checkpoint_dir: Path):
    with torch.no_grad():
        lang_sharded = vlm.language_model.sharded_state_dict(prefix="")
        dist_load(
            sharded_state_dict=lang_sharded, 
            checkpoint_dir=str(llm_checkpoint_dir), 
            strict="assume_ok_unexpected", # "log_all"
        )


@torch.no_grad()
def load_hf_vit_into_rice(vlm: RiceGPTModel, vit_path: str):
    """
    HF ViT를 로드하고 키 매핑 후 RiceViTModel에 주입.
    """
    print_rank_0(f"Loading weights from Hugging Face Hub: {vit_path}")
    cache_path = hf_hub_download(vit_path, "model.safetensors")

    vit_weights = load_file(cache_path)
    loaded_keys = 0
    VIT_KEYS_TO_MODIFY_MAPPING = {
        "vision_model.": "",
        "embeddings.": "",
        "patch_embedding.weight": "patch_embed.proj.weight",
        "encoder.layers.": "decoder.layers.",
        "self_attn": "self_attention",
        "self_attention.out_proj.": "self_attention.linear_proj.",
        "layer_norm1.": "self_attention.linear_qkv.layer_norm_",
        "mlp.fc1.": "mlp.linear_fc1.",
        "mlp.fc2.": "mlp.linear_fc2.",
        "layer_norm2.": "mlp.linear_fc1.layer_norm_",
        "pre_layrnorm": "pre_layernorm",
    }

    def merge_qkv_weights(state_dict, block_prefix):
        # Merge q_proj, k_proj, v_proj weights and biases
        q_w = state_dict[f"{block_prefix}.self_attention.q_proj.weight"]
        k_w = state_dict[f"{block_prefix}.self_attention.k_proj.weight"]
        v_w = state_dict[f"{block_prefix}.self_attention.v_proj.weight"]
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{block_prefix}.self_attention.q_proj.bias"]
        k_b = state_dict[f"{block_prefix}.self_attention.k_proj.bias"]
        v_b = state_dict[f"{block_prefix}.self_attention.v_proj.bias"]
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)
        return {f"{block_prefix}.self_attention.linear_qkv.weight": qkv_weight, f"{block_prefix}.self_attention.linear_qkv.bias": qkv_bias}

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in VIT_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value

        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if key.startswith("decoder.layers.") and "self_attention" in key and ("q_proj" in key or "k_proj" in key or "v_proj" in key):
                block_index = key.split('.')[2]
                block_prefix = f"decoder.layers.{block_index}"
                if f"{block_prefix}.self_attention.q_proj.weight" in new_state_dict:
                    merge_res = merge_qkv_weights(new_state_dict, block_prefix)
                    new_state_dict2.update(merge_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2

    vit_weights = convert_state_dict(vit_weights)
    vit_weights.pop("post_layernorm.weight")
    vit_weights.pop("post_layernorm.bias")

    missing, unexpected = vlm.vision_model.load_state_dict(vit_weights, strict=False)
    if len(missing) > 0:
        print("[HF-ViT->Rice] missing keys:")
        for k in missing:
            print("  -", k)

    if len(unexpected) > 0:
        print("[HF-ViT->Rice] unexpected keys:")
        for k in unexpected:
            print("  -", k)
            
    known_missing = {}  # 전부 다 로드되어야 정상, (missing, unexpected 모두 없어야 함)
    real_missing = [k for k in missing if k not in known_missing]
    if real_missing:
        raise RuntimeError(f"[HF-ViT->Rice] 예상하지 못한 missing keys: {real_missing}")
    if unexpected:
        raise RuntimeError(f"[HF-ViT->Rice] 예상하지 못한 unexpected keys: {unexpected}")
    

def write_latest_tracker(dst_root: Path, dst_iter: int):
    (dst_root / "latest_checkpointed_iteration.txt").write_text(f"{dst_iter}\n", encoding="utf-8")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--llm-checkpoint-dir", required=True, help="기존 LLM Megatron 체크포인트의 iter_xxxxxx 디렉토리")
    # parser.add_argument("--dst-root", required=True, help="VLM 초기 체크포인트가 저장될 루트 (이하 iter_xxxxxx 생성)")
    # parser.add_argument("--dst-iter", type=int, default=0)

    # parser.add_argument("--vit-model-id", type=str, default="DeepGlint-AI/rice-vit-large-patch14-560", help="비전 백본 모델 id")

    # args_cli = parser.parse_args()

    # TODO: 여기서 위처럼 해버리면 megatron 의 기본 인자들을 받을 수가 없음

    initialize_megatron()
    args = get_args()

    llm_checkpoint_dir = Path("/mnt/output2/kks/checkpoints/test/sft_lr_3e6_16384/iter_0008300")
    dst_root = Path("/mnt/output/ncai/multi_modal/Megatron-LM/checkpoints/vlm_merged_wbl_llm_sft_lr_3e6_seqlen_16384_iter_0008300")
    dst_iter = 1
    vit_model_id = "DeepGlint-AI/rice-vit-large-patch14-560"

    # llm_checkpoint_dir = Path(args_cli.llm_checkpoint_dir if hasattr(args_cli, "llm-checkpoint-dir") else args_cli.llm_checkpoint_dir).resolve()
    # dst_root = Path(args_cli.dst_root).resolve()
    # dst_iter_dir = dst_root / f"iter_{args_cli.dst_iter:07d}"
    dst_iter_dir = dst_root / f"iter_{dst_iter:07d}"
    dst_iter_dir.mkdir(parents=True, exist_ok=True)

    vlm = build_vlm(args)
    vlm.eval()

    print("origin lang norm:", vlm.language_model.embedding.word_embeddings.weight.data.sum())
    import time
    time.sleep(1)
    print("origin vision norm:", vlm.vision_model.decoder.layers[1].self_attention.linear_qkv.weight.data.sum())
    time.sleep(1)

    load_llm_into_vlm_language_model(vlm, llm_checkpoint_dir)

    load_hf_vit_into_rice(vlm, vit_model_id)
    
    print("lang norm:", vlm.language_model.embedding.word_embeddings.weight.data.sum())
    import time
    time.sleep(1)
    print("vision norm:", vlm.vision_model.decoder.layers[1].self_attention.linear_qkv.weight.data.sum())

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

    write_latest_tracker(dst_root, dst_iter)
    
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
    
    (dst_iter_dir / "vision_init.json").write_text(
        json.dumps(
            {
                "vision_backbone_hf_id": vit_model_id,
                "note": "Vision projector are randomly initialized here;"
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print_rank_0(f"[OK] VLM dist checkpoint saved: {dst_iter_dir}")

if __name__ == "__main__":
    main()

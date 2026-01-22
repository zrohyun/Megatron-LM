import torch
import shutil
import argparse
import sys
from pathlib import Path
import pdb
from typing import Any, Callable, Literal, Optional, Union
from megatron.core import parallel_state, dist_checkpointing
from megatron.core.enums import ModelType
from megatron.core.transformer import MegatronModule
from megatron.core.dist_checkpointing.serialization import (
    StateDict,
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.utils import unwrap_model
from modelopt.torch.opt.plugins import (
    restore_modelopt_state,
    save_modelopt_state,
    save_sharded_modelopt_state,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
from megatron.bridge.training.checkpointing import _generate_model_state_dict, _load_model_weights_from_checkpoint
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import (
    get_rank_safe,
    is_last_rank,
    print_rank_0,
)
import torch.nn as nn
# from pretrain_gpt_for_wbl import model_provider_with_args
from vlm.bridge.rice_gpt_provider import rice_gpt_model_provider_with_args
# from vlm.train.pretrain.pretrain_rice_gpt import model_provider

from vlm.bridge.vaetki_vl_bridge import VaetkiBridge # register bridge

dim = 1024
num_heads = 16
head_dim = dim // num_heads # 1024 / 16 = 64

num_repeats = head_dim // num_heads
num_splits = num_repeats + 2 # repeats*Q + K + V



def megatron_to_hf_vit_attn(param, n_heads, h_dim, d_in, is_bias=False):

    if is_bias:
        b = param.view(n_heads, 3, h_dim)
        bq = b[:, 0].reshape(dim)
        bk = b[:, 1].reshape(dim)
        bv = b[:, 2].reshape(dim)
        b = torch.concat([bq, bk, bv], dim=0)
        return b
    else:
        W = param.view(n_heads, 3, h_dim, dim)
        Wq = W[:, 0].reshape(dim, dim)
        Wk = W[:, 1].reshape(dim, dim)
        Wv = W[:, 2].reshape(dim, dim)
        W = torch.concat([Wq,Wk,Wv], dim=0)
        return W

def delete_extra_state(state_dict):
    """Delete all extra state keys from the model state dictionary.

    This function removes all keys containing '_extra_state' from the model
    portion of the state dictionary. This is useful for cleaning up corrupted
    or problematic extra state that can cause issues during model loading.

    Args:
        state_dict: The state dictionary. Can be either:
                   - A full checkpoint dict with a "model" key, or
                   - A model state dict directly

    Returns:
        The modified state dictionary with extra state keys removed.
    """
    # Handle both cases: full checkpoint dict with "model" key or direct model state dict
    if isinstance(state_dict, dict) and "model" in state_dict:
        # Full checkpoint dict case
        target_dict = state_dict["model"]
    else:
        # Direct model state dict case
        target_dict = state_dict

    # If target is not a mapping-like object, nothing to clean
    if not hasattr(target_dict, "keys"):
        return state_dict

    # Some objects may implement keys() but not be directly iterable into a list (e.g., mocks)
    try:
        keys = list(target_dict.keys())
    except Exception:
        return state_dict

    for key in keys:
        if isinstance(key, str) and "_extra_state" in key:
            del target_dict[key]
    return state_dict


def _load_model_state_dict(module: torch.nn.Module, state_dict: dict[str, Any], strict: bool):
    """Helper function to load state dict with fallback for missing extra states."""
    try:
        module.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        if strict:
            # Fallback support for backward compatibility breaking changes in TransformerEngine
            print_rank_0(f"Warning: Exception during strict loading: {e}")
            load_return = module.load_state_dict(state_dict, strict=True)
            print_rank_0(f"load_return: {load_return}")
        else:
            # Re-raise if we were already in non-strict mode
            raise


def _load_model_weights_from_checkpoint_fixed(
    checkpoint_path: str,
    model: list[MegatronModule],
    fully_parallel_load: bool = False,
    return_state_dict: bool = False,
    dist_ckpt_strictness: Literal[
        "assume_ok_unexpected",
        "log_unexpected",
        "log_all",
        "raise_unexpected",
        "raise_all",
        "return_unexpected",
        "return_all",
        "ignore_all",
    ] = "assume_ok_unexpected",
    strict: bool = True,
    ) -> Optional[Union[StateDict, tuple[StateDict, set[str], set[str]]]]:
    """Load model weights from a checkpoint.

    MCore distributed checkpoints from both Megatron Bridge and MegatronLM are supported.
    This function duplicates some logic from load_checkpoint() to simplify model
    loading for inference.

    Args:
        checkpoint_path: path to a distributed checkpoint.
        model: The model module(s) to load weights into.
        fully_parallel_load: Apply full load parallelization across DP.
        return_state_dict: Skips loading state dict into model and returns model state dict
            itself. Default False.
        dist_ckpt_strictness: Determine handling of key mismatch during checkpoint load.
        strict: Whether to enforce strict loading (see torch.nn.Module.load_state_dict).
    """

    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_path)
    assert state_dict is not None

    sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
    print_rank_0(f"sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}")
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)

    # [ModelOpt]: Restore state
    restore_modelopt_state(model, state_dict)

    model = unwrap_model(model)
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)

    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    if fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_path, load_strategy, strict=dist_ckpt_strictness
    )

    # state_dict = torch.load("/workspace/vlm/bridge/sd.ckpt", weights_only=False)
    delete_extra_state(state_dict)
    
    if return_state_dict:
        return state_dict
    
    print_rank_0("Loading the state dict of the model...")
    if len(model) == 1:
        for k, v in state_dict["model"].items():
            if "linear_qkv" in k and (".weight" in k or ".bias" in k):
                is_bias = True if "bias" in k else False
                state_dict["model"][k] = megatron_to_hf_vit_attn(v, n_heads=num_heads, h_dim=head_dim, d_in=dim, is_bias=is_bias)
            # if "vision_projection.layernorm.weight" in k:
            #     state_dict["model"][k] = nn.Parameter(v.detach().clone() + 1.0)
        _load_model_state_dict(model[0], state_dict["model"], strict=True)
    else:
        for i in range(len(model)):
            # If there is no corresponding model in the state_dict, it will be ignored.
            # It means that this is an empty stage.
            model_key = "model%d" % i
            if model_key not in state_dict:
                continue
            for k, v in state_dict[model_key].items():
                if "linear_qkv" in k:
                    is_bias = True if "bias" in k else False
                    state_dict[model_key][k] = megatron_to_hf_vit_attn(v, n_heads=num_heads, h_dim=head_dim, d_in=dim, is_bias=is_bias)
                # if "vision_projection.layernorm.weight" in k:
                #     state_dict["model"][k] = nn.Parameter(v.detach().clone() + 1.0)
            _load_model_state_dict(model[i], state_dict[model_key], strict=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def load_megatron_model(megatron_path):
    _, mlm_args = load_model_config(megatron_path)
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
    _load_model_weights_from_checkpoint_fixed(megatron_path, [model])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--megatron-path", type=str)
    parser.add_argument("--hf-path", type=str)
    parser.add_argument('--no-use-cpu-initialization', action='store_false', dest='use_cpu_initialization')
    parser.add_argument('--save_megatron', action='store_true')
    args = parser.parse_args()

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    backend = "gloo" if args.use_cpu_initialization else "nccl"
    with temporary_distributed_context(backend):
        megatron_model = load_megatron_model(args.megatron_path)
        if args.save_megatron:
            torch.save(megatron_model.state_dict(), "/workspace/vlm/bridge/megatron.ckpt")
            print("Megatron state dict saved! at /workspace/vlm/bridge/megatron.ckpt")
        bridge.save_hf_pretrained([megatron_model], args.hf_path)
        shutil.copy(f"{args.hf_model}/modeling_vaetki_vl.py", args.hf_path)
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
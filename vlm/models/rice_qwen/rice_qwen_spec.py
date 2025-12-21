import torch
from typing import Optional

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer, TransformerLayerSubmodules)


try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_qwen_layer_with_transformer_engine_default_spec(
    num_experts: Optional[int] = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False, padding=False
) -> ModuleSpec:

    attn_mask_type = AttnMaskType.causal
    # Padding mask is needed for e.g. Context Parallel.
    if padding:
        attn_mask_type = AttnMaskType.padding_causal

    mlp = get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        )
    )


def get_qwen_layer_with_local_default_spec(
    num_experts: Optional[int] = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:

    mlp = get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        )
    )


def get_multimodal_projector_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )


def get_norm_multimodal_projector_module_spec_te() -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )

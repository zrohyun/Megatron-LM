import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TELinear,
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

from .vision_projector import VisionProjectorSubmodules


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


def get_vision_projector_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return VisionProjectorSubmodules(
        layernorm=TENorm,
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )


# def get_multimodal_projector_module_spec(use_te: bool = True) -> ModuleSpec:
#     # Dense MLP w/ or w/o TE modules.
#     return ModuleSpec(
#         module=MLP,
#         submodules=MLPSubmodules(
#             linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
#             linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
#         ),
#     )


# def get_norm_multimodal_projector_module_spec_te() -> ModuleSpec:
#     return ModuleSpec(
#         module=MLP,
#         submodules=MLPSubmodules(
#             linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
#         ),
#     )

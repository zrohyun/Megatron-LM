import torch
from dataclasses import dataclass

from vlm.utils.constants import VisionLanguageModelFamilies
from vlm.models.factory import register_model_config


@dataclass
class MultimodalProjectorConfig:
    """configuration for multimodal projector model
    The fields need to be consistent with the definitions in args
    """
    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06


@dataclass
class RiceGPTConfig:
    """config for rice gpt model"""
    ...
    # # MODEL_ARGS
    # num_layers: int
    # hidden_size: int
    # ffn_hidden_size: int
    # num_attention_heads: int
    # add_bias_linear: bool = False
    # position_embedding_type: str = "none"
    # init_method_std: float = 0.0134
    # attention_dropout: float = 0
    # hidden_dropout: float = 0
    # swiglu: bool = True
    # untie_embeddings_and_output_weights: bool = True
    # masked_softmax_fusion: bool = False

    # # MOE_ARGS
    # num_experts: int = 128
    # moe_layer_freq: str = '([0]*1+[1]*16)'
    # moe_ffn_hidden_size: int = 512
    # moe_shared_expert_intermediate_size: int = 512
    # moe_shared_expert_overlap: bool = True
    # moe_router_padding_for_fp8: bool = True
    # moe_router_load_balancing_type: str = "global_aux_loss"
    # moe_router_topk: int = 6
    # moe_grouped_gemm: bool = True
    # moe_aux_loss_coeff: float = 1e-2
    # moe_z_loss_coeff: float = 1e-3
    # moe_router_topk_scaling_factor: float = 1.0
    # moe_router_score_function: str = "sigmoid"
    # moe_token_dispatcher_type: str = "alltoall"
    # moe_permute_fusion: bool = True
    # moe_router_dtype: str = "fp32"
    # overlap_param_gather: bool = True
    # overlap_grad_reduce: bool = True

    # # MLA_ARGS
    # multi_latent_attention: bool = True
    # q_lora_rank: int = 1024
    # kv_lora_rank: int = 512
    # qk_head_dim: int = 128
    # qk_pos_emb_head_dim: int = 64
    # v_head_dim: int = 128
    # rotary_scaling_factor: float = 1.0
    # normalization: str = "RMSNorm"
    # rope_type: str = "rope"
    # apply_layernorm_1p: bool = True
    # rotary_base: int = 10000
    # rotary_base_global: int = 1000000
    # qk_layernorm: bool = True


@register_model_config(model_family=VisionLanguageModelFamilies.RICE_GPT, model_arch="rice-gpt-7b-a1b")
def rice_gpt_7b_a1b():
    # _validate_extra_model_args 이 함수에서 config 가 args 로 들어감
    # wbl-llm-7b-a1b 의 설정은 이미 정해져 있으므로 여기에서 고정한다.
    """rice-gpt-7b-a1b"""
    return RiceGPTConfig(
        # # MODEL_ARGS
        # add_bias_linear=False,
        # num_layers=24,
        # position_embedding_type="none",
        # hidden_size=1536,
        # ffn_hidden_size=5760,
        # num_attention_heads=12,
        # init_method_std=0.0134,
        # attention_dropout=0.0,
        # hidden_dropout=0.0,
        # swiglu=True,
        # untie_embeddings_and_output_weights=True,
        # masked_softmax_fusion=False,

        # # MOE_ARGS
        # num_experts=64,
        # moe_layer_freq='([0]*1+[1]*23)',
        # moe_ffn_hidden_size=960,
        # moe_shared_expert_intermediate_size=960,
        # moe_shared_expert_overlap=False,
        # moe_router_padding_for_fp8=True,
        # moe_router_load_balancing_type="global_aux_loss",
        # moe_router_topk=5,
        # moe_grouped_gemm=True,
        # moe_aux_loss_coeff=2e-2,
        # moe_z_loss_coeff=1e-3,
        # moe_router_topk_scaling_factor=1.0,
        # moe_router_score_function="sigmoid",
        # moe_token_dispatcher_type="alltoall",
        # moe_permute_fusion=True,
        # moe_router_dtype="fp32",
        # overlap_param_gather=True,
        # overlap_grad_reduce=True,

        # # MLA_ARGS
        # multi_latent_attention=True,
        # q_lora_rank=768,
        # kv_lora_rank=512,
        # qk_head_dim=128,
        # qk_pos_emb_head_dim=64,
        # v_head_dim=128,
        # rotary_scaling_factor=1.0,
        # normalization="RMSNorm",
        # rope_type="rope",
        # apply_layernorm_1p=True,
        # rotary_base=10000,
        # rotary_base_global=1000000,
        # qk_layernorm=True,
    )


@dataclass
class VisionConfig:
    """configuration for vision model
    
    The fields need to be consistent with the definitions in args
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: tuple[int]
    image_size: tuple[int]
    kv_channels: int
    normalization: str
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    layernorm_epsilon: float = 1e-05
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    layernorm_zero_centered_gamma: bool = False


def get_vision_config(model_family, model_name):
    """ get vision config """
    config = VisionConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        patch_size=14,
        image_size=(1344, 1344),
        kv_channels=64,
        normalization="LayerNorm",
        swiglu=False,
        class_token_len=0,
        group_query_attention=False,
        attention_dropout=0,
        hidden_dropout=0,
        layernorm_epsilon=1e-5,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=False,
        gated_linear_unit=False,
        in_channels=3,
        num_query_groups=16,
        add_bias_linear=True,
        add_qkv_bias=True,
        position_embedding_type="rope"
    )
    if "vision-2b" in model_name:
        config.num_layers = 48
        config.hidden_size = 1664
        config.ffn_hidden_size = 8192
        config.kv_channels = 104

    return config


def get_vision_projection_config(model_family):
    """ get vision projection config """
    return MultimodalProjectorConfig(
        normalization="LayerNorm",
        add_bias_linear=True,
    )

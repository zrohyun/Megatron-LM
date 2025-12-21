"""rice-gpt model provider"""

import warnings
from typing import Optional
from copy import deepcopy
from dataclasses import asdict

from megatron.training.utils import print_rank_0
from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from megatron.core.models.wbl_moe_gpt.model import (
    get_wbl_moe_gpt_decoder_block_spec,
    get_wbl_moe_gpt_layer_with_transformer_engine_spec,
)

from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)

from vlm.models.factory import register_model_provider
from vlm.models.rice_gpt.rice_gpt_config import (
    get_vision_config,
    get_vision_projection_config,
)
from vlm.models.vision.rice_vit_spec import (
    get_vit_layer_with_transformer_engine_spec,
    get_vit_layer_with_local_spec,
)
from vlm.utils import build_transformer_config, get_args
from vlm.utils.constants import VisionLanguageModelFamilies
from vlm.models.factory import get_model_family

from .rice_gpt_model import RiceGPTModel
from .rice_gpt_spec import get_vision_projector_layer_with_spec


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        # return get_gpt_layer_with_transformer_engine_spec(
        return get_wbl_moe_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


@register_model_provider(model_family=[VisionLanguageModelFamilies.RICE_GPT])
def rice_gpt_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    vp_stage: Optional[int] = None,
    parallel_output: bool = True,
) -> RiceGPTModel:
    """Builds the model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A multimodal model.
    """
    args = get_args()

    # if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
    #     return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0(f"building Rice-GPT model ...")

    assert args.decoder_seq_length is not None, (
        "Please provide --decoder-seq-length to set the language model sequence length"
    )
    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn(
            f"Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the maximum language model sequence length"
        )

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    config = build_transformer_config(args)

    language_config = deepcopy(config)
    vision_config = deepcopy(config)
    vision_projection_config = deepcopy(config)

    model_family = get_model_family(args.model_name)
    for k, v in asdict(get_vision_config(model_family, args.model_name)).items():
        setattr(vision_config, k, v)
    for k, v in asdict(get_vision_projection_config(model_family)).items():
        setattr(vision_projection_config, k, v)

    # # print(vision_config)
    setattr(language_config, "image_token_id", 137179)  # "<|reserved_2020|>"
    setattr(language_config, "video_token_id", 137180)  # "<|reserved_2021|>"

    # Language Model Layer Spec
    # NOTE:args.use_legacy_models 및 args.spec 옵션은 일단 미구현
    if args.num_experts:
        # Define the decoder block spec
        language_transformer_layer_spec = get_wbl_moe_gpt_decoder_block_spec(
            config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
        )
    elif args.heterogeneous_layers_config_path is not None:
        language_transformer_layer_spec = get_gpt_heterogeneous_layer_spec(
            config, use_te
        )
    else:
        # Define the decoder layer spec
        language_transformer_layer_spec = _get_transformer_layer_spec(use_te, config)

    # Vision Model Layer Spec
    if use_te:
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    else:
        vision_transformer_layer_spec = get_vit_layer_with_local_spec()

    # Make sure vision model pipeline parallel size is not inherited from the language model pipeline parallel size.
    vision_config.pipeline_model_parallel_size = 1
    vision_projection_config.pipeline_model_parallel_size = (
        vision_config.pipeline_model_parallel_size
    )

    # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
    vision_config.first_pipeline_num_layers = vision_config.last_pipeline_num_layers = (
        None
    )

    # if vision_projection_config.normalization:
    #     vision_projection_layer_spec = get_norm_multimodal_projector_module_spec_te().submodules
    # else:
    #     vision_projection_layer_spec = get_multimodal_projector_module_spec(use_te=use_te).submodules
    vision_projection_layer_spec = get_vision_projector_layer_with_spec()

    # # Toggle --recompute* for the vision and language model separately.
    # if getattr(args, "recompute_vision", False):
    #     if (
    #         vision_config.recompute_method is not None
    #         and vision_config.recompute_granularity is not None
    #     ):
    #         vision_config.recompute_num_layers = vision_config.num_layers
    # else:
    #     vision_config.recompute_granularity = None
    #     vision_config.recompute_method = None
    #     vision_config.recompute_num_layers = None

    # vision_projection_config.recompute_granularity = None
    # vision_projection_config.recompute_method = None
    # vision_projection_config.recompute_num_layers = None
    
    vision_config.layernorm_zero_centered_gamma = False
    
    # LEEJH ADDED
    vision_config.window_size = None

    # TODO: Vision model and projection do not use SP/CP yet.
    vision_config.sequence_parallel = False
    vision_config.context_parallel_size = 1
    vision_config.tp_comm_overlap = False

    vision_projection_config.sequence_parallel = False
    vision_projection_config.context_parallel_size = 1
    vision_projection_config.tp_comm_overlap = False

    model = RiceGPTModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,  # args.decoder_seq_length
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        # drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        allow_missing_vision_projection_checkpoint=getattr(
            args, "allow_missing_vision_projection_checkpoint", False
        ),
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
    )

    if getattr(args, "trainable_modules", ["all"]) != ["all"]:
        train_language_model = "language_model" in args.trainable_modules
        train_vision_model = "vision_model" in args.trainable_modules
        train_vision_projection = "vision_projection" in args.trainable_modules
        model.freeze(
            freeze_language_model=not train_language_model,
            freeze_vision_model=not train_vision_model,
            freeze_vision_projection=not train_vision_projection,
        )

    return model

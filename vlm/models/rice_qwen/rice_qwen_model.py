import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch
from megatron.core import InferenceParams, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.utils import make_viewless_tensor, log_single_rank

from vlm.models.vision.rice_vit_model import RiceViTModel
from vlm.models.qwen.qwen_model import QwenModel
from vlm.models.qwen_vl.utils import get_inputs_on_this_cp_rank


class RiceQwenModel(MegatronModule):
    """Rice-Qwen multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Language model spec.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Vision model spec.
        drop_vision_class_token (bool): Drop vision class token(s) before the language model.
        vision_projection_config (TransformerConfig): Vision projection config.
        vision_projection_layer_spec (ModuleSpec): Vision projection spec.
        vision_projection_type (str): Type of the vision projection. Default: 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be
            missing when loading a checkpoint. Default False.
        parallel_output (bool): Keep outputs split across tensor parallel ranks.
            This is typically True for training and False for inference.
        share_embeddings_and_output_weights (bool): Input embedding and output layer share weights.
        language_position_embedding_type (str): Language model position embedding type.
        language_rotary_percent (float): RoPE percent. Defaults to 1.0.
        pre_process (bool): Include embedding layer in the decoder (used with pipeline parallel).
        post_process (bool): Include output layer in the decoder (used with pipeline parallel).
        add_encoder (bool): Construct the encoder (used with pipeline parallel).
            When we use pipelining, the encoder will live on only the first stage
        add_decoder (bool): Construct the decoder (used with pipeline parallel).
            When we use pipelining, the decoder will live on every stage after the first one.
        language_rotary_base (int): RoPE base.
        language_rope_scaling (bool): Toggle RoPE scaling.
        language_rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.
        image_token_index (int): Token ID for image token such as <image>.
        pixel_shuffle (bool): Enable pixel shuffle.
        tile_tags (list): Optional tile tags.
        model_comm_pgs (ModelCommProcessGroups): Model communication process groups.
        vp_stage (int): Virtual pipeline stage.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "rope",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        fp16_lm_cross_entropy: bool = False,
        tile_tags: Optional[list] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        max_num_tiles: int = 0,
        tokenizer_type: str = "",
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "RiceQwenModel is work in progress. Features are missing and methods can change.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None

        if model_comm_pgs is None:
            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()
        self.model_comm_pgs = model_comm_pgs

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            if self.context_parallel_lm > 1:
                self.cp_group = self.model_comm_pgs.cp
                assert (
                    self.cp_group.size() == self.context_parallel_lm
                ), "CP Group size should match the Language Model CP size"
                assert is_te_min_version(
                    "1.10.0"
                ), "Context Parallelism in LLaVA requires TE v1.10 or higher"
            else:
                self.cp_group = None
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        if self.add_decoder:
            language_model_type = getattr(language_transformer_config, "language_model_type", "")
            assert "qwen" in language_model_type, "RiceQwenModel only supports Qwen model"  

            # self.rotary_emb = Qwen2VLRotaryEmbedding(
            #     dim=language_config.hidden_size // language_config.num_attention_heads,
            #     theta=language_rotary_base
            # )
            self.language_model = QwenModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            )

            self._language_max_sequence_length = language_max_sequence_length
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

            # Newer Transformer Engine versions add _extra_state keys in state_dict when using FP8.
            # Older models may not have _extra_state and can be ignored.
            self.language_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

        #  define the vision model and the projection from vision model outputs to language model inputs.
        if self.add_encoder:
            vision_model_type = getattr(vision_transformer_config, "vision_model_type", "")
            assert "rice" in vision_model_type, "RiceQwenModel only supports RiceViT model"  

            self.vision_model = RiceViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                patch_size=14,    # TODO: get from config
                in_channels=3,    # TODO: get from config
            )

            self.vision_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

            vision_projection_input_size = vision_transformer_config.hidden_size
            # vision_projection_input_size *= 4 if pixel_shuffle else 1

            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_projection_input_size,
                tp_group=self.model_comm_pgs.tp,
            )

           # Ignore missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains
            # pretrained vision and language models but not the projection from vision model
            # outputs to language model inputs.
            if allow_missing_vision_projection_checkpoint:
                vision_projection_param_names = [
                    f"vision_projection.{name}"
                    for name in self.vision_projection.state_dict().keys()
                ]
                self.vision_projection.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, vision_projection_param_names)
                )

            self.vision_projection.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        inference_params: InferenceParams = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward function of the Qwen-VL model.

        Args:
            images (torch.Tensor): input image of shape [image_size / 3d_patch_size, in_channels * 3d_patch_size].
            image_grid_thw (torch.Tensor): image grid tensor of shape [num_images, 3]
            pixel_values_videos (torch.Tensor): The tensors corresponding to the input videos, 
                            shape:[seq_length, num_channels * temporal_size * patch_size * patch_size]
            video_grid_thw (torch.Tensor): video grid tensor of shape [num_videos, 3]
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """
        # from megatron.training import print_rank_0
        # print_rank_0(
        #     f"> forward step: input_ids shape {input_ids.shape}, "
        #     f"images shape {images.shape}, "
        #     f"image_grid_thw shape {image_grid_thw.shape}, "
        #     # f"labels shape {labels.shape}, "
        #     f"attn_mask_type {attn_mask_type}, "
        #     f"position_ids shape {position_ids.shape if position_ids is not None else None}"
        # )
        # print_rank_0(input_ids)
        # print_rank_0(position_ids)
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device).reshape(
                0, 0, 0
            )
        elif self.add_encoder:
            if images is not None:
                image_embeddings, window_index = self.vision_model(images, grid_thw=image_grid_thw) # [img_len, h_vision]

                # if self._pixel_shuffle:
                #     image_embeddings = pixel_shuffle(
                #         image_embeddings
                #     )  # [num_tiles, img_seq_len_shuffled, h_vision_shuffled]

                # map vision model output size to language model input size.
                image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeddings.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features {n_image_features} != image tokens {n_image_tokens}"
                    )

                # If running inference, the language model KV cache will be updated for image token positions.
                # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                if inference_params is not None:
                    inference_params.key_value_memory_dict["image_tokens_count"] = (
                        image_embeddings.shape[0]
                    )
            if pixel_values_videos is not None:
                raise NotImplementedError(
                    "Video input is not supported in RiceVLModel. "
                    "Please use a different model that supports video input."
                )
                # pixel_values_videos: [video_seq_len, num_channels * temporal_size * patch_size * patch_size]
                # video_grid_thw: [num_videos, 3]
                # # Get the video embeddings from the vision model.
                # video_embeddings, window_index = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw)
                # video_embeddings = self.adapter(video_embeddings, window_index)
                # n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                # n_video_features = video_embeddings.shape[0]
                # if n_video_tokens != n_video_features:
                #     raise ValueError(
                #         f"video features {n_video_features} != video tokens {n_video_tokens}"
                #     )

                # # If running inference, the language model KV cache will be updated for image token positions.
                # # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
                # if inference_params is not None:
                #     inference_params.key_value_memory_dict["video_tokens_count"] = (
                #         video_embeddings.shape[0]
                #     )
        else:
            vision_embeddings = self.encoder_hidden_state.squeeze(0) if self.encoder_hidden_state is not None else None

        if not self.add_decoder:
            # p2p_communicate_shapes requires dim=3
            vision_embeddings = make_viewless_tensor(
                inp=vision_embeddings.unsqueeze(0),
                requires_grad=True,
                keep_graph=True
            )
            return vision_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already
            # earlier for this sample.
            if use_inference_kv_cache or (images is None and pixel_values_videos is None):
                combined_embeddings = language_embeddings
            else:
                if images is not None and self.config.image_token_id in input_ids:
                    image_token_id = self.config.image_token_id
                    images_mask = (
                        (input_ids == image_token_id).transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    image_embeddings = image_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(images_mask, image_embeddings)

                if pixel_values_videos is not None and self.config.video_token_id in input_ids:
                    video_token_id = self.config.video_token_id
                    videos_mask = (
                        (input_ids == video_token_id).transpose(0, 1)
                        .unsqueeze(-1)
                        .expand_as(language_embeddings)
                        .to(language_embeddings.device)
                    )
                    video_embeddings = video_embeddings.to(language_embeddings.device, language_embeddings.dtype)
                    combined_embeddings = language_embeddings.masked_scatter(videos_mask, video_embeddings)

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings)

        else:
            combined_embeddings = None
            input_tensor = self.language_model.decoder.input_tensor

        # rotary_pos_emb = self.rotary_emb(position_ids).transpose(0, 2).contiguous()

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            # rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb=None,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={},
        )

        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this if you want to load a checkpoint that contains vision and language
    model weights but not the vision projection weights.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    f"_extra_state key {key} being removed from {name}"
                )
                keys.remove(key)


# pylint: disable-next=line-too-long
# Based on https://github.com/OpenGVLab/InternVL/blob/c7c5af1a8930b4862afe8ed14672307082ef61fa/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L218
# Copyright (c) 2023 OpenGVLab.
def pixel_shuffle(x, scale_factor=0.5, version=2):
    """Pixel shuffle based on InternVL but adapted for our use case.

    Args:
        x (torch.Tensor): Vision model outputs [num_tiles, img_seq_len, h_vision]
        version (int): Implementation version.

    Returns:
        Shuffled vision model outputs [num_tiles, (sq ** 2) * (scale ** 2), h_vision / (scale ** 2)]
    """
    h = w = int(x.shape[1] ** 0.5)  # sq
    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, sq, sq, h_vision]

    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(
        n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
    )

    if version == 2:
        x = x.permute(0, 2, 1, 3).contiguous()

    x = x.reshape(x.shape[0], -1, x.shape[-1])

    return x
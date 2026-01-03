# SPDX-License-Identifier: Apache-2.0
"""
vLLM-native implementation of WBL VLM (Vision Language MoE) model.

This implementation follows vLLM's architecture patterns for multimodal models,
similar to Qwen2.5-VL, with the following key components:

1. Rice Vision Encoder - Custom vision transformer with vLLM attention
2. Language Model - Uses vLLM's registered Qwen2 MoE model
3. Multimodal Integration - Efficient image/video embedding and merging
"""

from typing import Annotated, Iterable, Literal, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import maybe_get_vit_flash_attn_backend
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)

from .configuration_wbl_vl_moe import WBLVLMoEConfig, RiceConfig
from .utils import maybe_prefix

# Import WBL language model (copied from LLM plugin)
from .wbl_llm_model import WBLForCausalLM

logger = init_logger(__name__)


# ============================================================================
# Vision Input Schemas (similar to Qwen2.5-VL)
# ============================================================================

class WBLImagePixelInputs(TensorSchema):
    """
    Image input schema for WBL VLM.

    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Channels * patch_size * patch_size
    """
    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),  # (temporal, height, width)
    ]


class WBLImageEmbeddingInputs(TensorSchema):
    """Pre-computed image embedding inputs."""
    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),  # (num_features, hidden_size)
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


# ============================================================================
# Rice Vision Encoder Components (vLLM Native)
# ============================================================================

class RiceRotaryEmbedding(nn.Module):
    """Rotary position embedding for Rice vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to vision Q/K."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class RicePatchEmbed(nn.Module):
    """Patch embedding layer for Rice vision encoder."""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        return hidden_states.view(-1, self.embed_dim)


class RiceAttention(nn.Module):
    """
    Vision attention using vLLM's attention wrappers.

    This replaces the transformers-based attention with vLLM native implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        use_upstream_fa: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Use vLLM's QKV parallel linear for efficiency
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=True,  # No tensor parallelism for vision encoder
        )

        self.proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=True,
        )

        # Setup attention backend
        self.attn_backend = attn_backend
        self.use_upstream_fa = use_upstream_fa
        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                self.use_upstream_fa,
            )
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (seq_length, hidden_size)
            cu_seqlens: Cumulative sequence lengths for each image
            position_embeddings: (cos, sin) tuple for RoPE
        """
        seq_length = hidden_states.shape[0]

        # QKV projection
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Apply attention
        # Try flash_attn first, fallback to manual if not available
        try:
            attn_output = self._flash_attention(q, k, v, cu_seqlens)
        except Exception as e:
            # Fallback to manual attention
            attn_output = self._manual_attention(q, k, v, cu_seqlens)

        attn_output = attn_output.reshape(seq_length, -1)
        output, _ = self.proj(attn_output)
        return output

    def _flash_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Flash Attention implementation using flash_attn library directly."""
        from flash_attn import flash_attn_varlen_func

        # flash_attn expects: (total_seq_len, num_heads, head_dim)
        # Our input is already in this format

        # Compute max sequence length
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # Use flash attention
        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,  # Vision attention is not causal
        )

        return attn_output

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Fallback manual attention implementation."""
        import math

        # Process each sequence separately to avoid OOM
        outputs = []
        for i in range(1, len(cu_seqlens)):
            start = cu_seqlens[i - 1]
            end = cu_seqlens[i]

            # Extract this sequence's q, k, v
            q_i = q[start:end]  # [seq_i, num_heads, head_dim]
            k_i = k[start:end]
            v_i = v[start:end]

            # Transpose for batch matmul: [num_heads, seq_i, head_dim]
            q_i = q_i.transpose(0, 1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)

            # Compute attention
            attn_weights = torch.matmul(q_i, k_i.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_i.dtype)
            attn_output = torch.matmul(attn_weights, v_i)

            # Transpose back: [seq_i, num_heads, head_dim]
            attn_output = attn_output.transpose(0, 1)
            outputs.append(attn_output)

        # Concatenate all outputs
        return torch.cat(outputs, dim=0)


class RiceMLP(nn.Module):
    """MLP layer for Rice vision encoder."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        # Use standard activation, not gated activation
        if hidden_act == "gelu":
            self.act = nn.GELU()
        elif hidden_act == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = nn.GELU()  # Default to GELU
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RiceBlock(nn.Module):
    """Transformer block for Rice vision encoder."""

    def __init__(
        self,
        config: RiceConfig,
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)

        self.attn = RiceAttention(
            config.hidden_size,
            num_heads=config.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=attn_backend,
        )

        mlp_hidden_dim = int(config.intermediate_size)
        self.mlp = RiceMLP(
            dim=config.hidden_size,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class RicePatchMerger(nn.Module):
    """Merge vision features to language model dimension."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class RiceVisionTransformer(nn.Module):
    """
    Rice Vision Transformer - vLLM native implementation.

    This is the vision encoder that processes images/videos before
    feeding to the language model.
    """

    def __init__(
        self,
        config: RiceConfig,
        quant_config: Optional[QuantizationConfig] = None,
        text_hidden_size: int = 3584,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        # Patch embedding
        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        # Rotary position embeddings
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)

        # Class token and position embeddings
        scale = config.hidden_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))

        # Transformer blocks
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-4)

        # Determine attention backend
        # Use TORCH_SDPA for vLLM backend, but we'll use flash_attn directly in RiceAttention
        # This avoids vLLM v1's CUDA initialization issues
        attn_backend = AttentionBackendEnum.TORCH_SDPA

        self.blocks = nn.ModuleList([
            RiceBlock(
                config,
                attn_backend=attn_backend,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.depth)
        ])

        # Patch merger to project to language model dimension
        self.merger = RicePatchMerger(
            dim=text_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=1e-6,
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for image patches."""
        # Ensure grid_thw is 2D: (num_images, 3)
        # Handle various input shapes
        if grid_thw.dim() == 3:
            # Shape: (batch, 1, 3) -> squeeze middle dimension
            grid_thw = grid_thw.squeeze(1)
        elif grid_thw.dim() == 1:
            # Shape: (3,) -> add batch dimension
            grid_thw = grid_thw.unsqueeze(0)

        pos_ids = []
        for t, h, w in grid_thw:
            # Height position IDs
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            # Width position IDs
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Pixel values tensor
            grid_thw: (num_images, 3) tensor of [temporal, height, width]

        Returns:
            Image features ready for language model
        """
        # Normalize grid_thw to 2D
        if grid_thw.dim() == 3:
            grid_thw = grid_thw.squeeze(1)
        elif grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        # Patch embedding
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        img_feats = hidden_states.shape[0]

        # Compute cumulative sequence lengths
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        cu = cu_seqlens.to(torch.long)
        num_segments = cu.numel() - 1

        # Prepend class tokens
        cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)
        total_patches = cu[-1].item()
        new_total = total_patches + num_segments
        D = hidden_states.size(-1)

        new_hidden = hidden_states.new_empty((new_total, D))
        new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

        write_ptr = 0
        new_cu = [0]
        for i in range(1, num_segments + 1):
            seg_start = cu[i-1].item()
            seg_end = cu[i].item()
            seg_len = seg_end - seg_start

            new_hidden[write_ptr] = cls_token
            new_rotary_pos_emb[write_ptr] = self.class_pos_emb
            new_hidden[write_ptr + 1: write_ptr + 1 + seg_len] = hidden_states[seg_start:seg_end]
            new_rotary_pos_emb[write_ptr + 1: write_ptr + 1 + seg_len] = rotary_pos_emb[seg_start:seg_end]

            write_ptr += 1 + seg_len
            new_cu.append(write_ptr)

        hidden_states = new_hidden
        cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32)
        rotary_pos_emb = new_rotary_pos_emb

        # Pre-layernorm
        hidden_states = self.pre_layernorm(hidden_states)

        # Prepare position embeddings
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Transformer blocks
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    blk, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings
                )

        # Remove class tokens
        new_hidden = hidden_states.new_empty((img_feats, D))
        for i in range(1, num_segments + 1):
            seg_start = cu[i-1].item()
            seg_end = cu[i].item()
            new_hidden[seg_start:seg_end] = hidden_states[seg_start+1:seg_end+1]

        hidden_states = new_hidden

        # Merge patches and project to language model dimension
        return self.merger(hidden_states)


# ============================================================================
# Multimodal Processing (Required for vLLM)
# ============================================================================

from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.multimodal.inputs import MultiModalFieldConfig


class WBLProcessingInfo(BaseProcessingInfo):
    """Processing information for WBL VLM."""

    def get_hf_processor(self, **kwargs):
        """Get HuggingFace processor."""
        # WBL uses Qwen2.5-VL processor
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            **kwargs,
        )

    def get_image_processor(self, **kwargs):
        """Get image processor from HF processor."""
        return self.get_hf_processor(**kwargs).image_processor

    def get_tokenizer(self):
        """Get tokenizer."""
        return self.ctx.tokenizer

    def get_supported_mm_limits(self) -> dict[str, Optional[int]]:
        """Return maximum number of items per modality."""
        return {"image": 10, "video": 10}  # Support up to 10 images/videos


class WBLDummyInputsBuilder(BaseDummyInputsBuilder):
    """Build dummy inputs for memory profiling."""

    def get_dummy_text(self, mm_counts: dict[str, int]) -> str:
        """Return dummy text with image tokens."""
        num_images = mm_counts.get("image", 0)
        # Use vision tokens as placeholders
        return "<|vision_start|><|image_pad|><|vision_end|>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
        mm_options: Optional[dict] = None,
    ) -> dict:
        """Return dummy multimodal data for profiling."""
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {}

        # Create dummy images (small size for profiling)
        # Use a size that matches Rice encoder expectations
        import PIL.Image
        dummy_image = PIL.Image.new("RGB", (448, 448), color="white")

        return {
            "image": [dummy_image] * num_images
        }


class WBLMultiModalProcessor(BaseMultiModalProcessor):
    """Multimodal processor for WBL VLM."""

    def _get_mm_fields_config(
        self,
        hf_inputs: dict,
        hf_processor_mm_kwargs: dict,
    ) -> dict[str, MultiModalFieldConfig]:
        """Return schema of multimodal tensors."""
        # pixel_values: (num_patches, features) - shared across images
        # image_grid_thw: (num_images, 3) - batched by image

        # Get grid info to calculate patch slices
        if "image_grid_thw" in hf_inputs:
            grid_thw = hf_inputs["image_grid_thw"]  # (num_images, 3)
            # Each image has T*H*W patches
            num_patches_per_image = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]  # Keep as tensor

            return {
                "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                    "image", num_patches_per_image
                ),
                "image_grid_thw": MultiModalFieldConfig.batched("image"),
            }
        else:
            # Fallback - assume 1 image
            return {
                "pixel_values": MultiModalFieldConfig.shared("image", 1),
                "image_grid_thw": MultiModalFieldConfig.batched("image"),
            }

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        """Return prompt replacements for image tokens."""
        # WBL uses the Qwen2.5-VL processor, so we need the same logic
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_wbl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_wbl, modality=modality),
            )
            for modality in ("image", "video")
        ]


# ============================================================================
# Main WBL VLM Model
# ============================================================================

@MULTIMODAL_REGISTRY.register_processor(
    WBLMultiModalProcessor,
    info=WBLProcessingInfo,
    dummy_inputs=WBLDummyInputsBuilder,
)
class WBLVLMForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    WBL Vision Language Model for Conditional Generation.

    Architecture:
        - Rice Vision Encoder (custom vLLM implementation)
        - Language Model (vLLM's registered Qwen2 MoE model)
        - Multimodal integration
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Vision encoder
        if hasattr(config, 'vision_config') and config.vision_config is not None:
            self.visual = RiceVisionTransformer(
                config=config.vision_config,
                quant_config=quant_config,
                text_hidden_size=config.text_config.hidden_size,
                prefix=maybe_prefix(prefix, "visual"),
            )
        else:
            self.visual = None

        # Language model - use WBL's native implementation with MLA and MoE
        # Temporarily replace hf_config with text_config for language model initialization
        text_config = config.text_config if hasattr(config, 'text_config') else config
        original_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = text_config
        self.language_model = WBLForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        vllm_config.model_config.hf_config = original_hf_config

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Return placeholder string for multimodal inputs."""
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        return None

    def get_language_model(self) -> nn.Module:
        """Return the underlying language model."""
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode images into continuous embeddings."""
        if self.visual is None:
            raise ValueError("Vision encoder is not initialized")

        pixel_values = pixel_values.type(self.visual.patch_embed.proj.weight.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Embed multimodal inputs (images) into continuous embeddings."""
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        if pixel_values is None or image_grid_thw is None:
            return []

        # Debug: Print input types and shapes
        print(f"[DEBUG] pixel_values type: {type(pixel_values)}")
        print(f"[DEBUG] image_grid_thw type: {type(image_grid_thw)}")

        # Handle list of tensors (vLLM V1 API with multiple images)
        if isinstance(pixel_values, list):
            print(f"[DEBUG] pixel_values is list, length: {len(pixel_values)}")
            for i, pv in enumerate(pixel_values):
                print(f"[DEBUG] pixel_values[{i}] shape: {pv.shape}")
            pixel_values = torch.cat(pixel_values, dim=0)
            print(f"[DEBUG] After cat, pixel_values shape: {pixel_values.shape}")

        if isinstance(image_grid_thw, list):
            print(f"[DEBUG] image_grid_thw is list, length: {len(image_grid_thw)}")
            for i, gt in enumerate(image_grid_thw):
                print(f"[DEBUG] image_grid_thw[{i}]: {gt}, shape: {gt.shape if torch.is_tensor(gt) else 'N/A'}")
            image_grid_thw = torch.stack(image_grid_thw, dim=0)
            print(f"[DEBUG] After stack, image_grid_thw shape: {image_grid_thw.shape}")

        # Normalize grid_thw shape to 2D: [num_images, 3]
        if image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.squeeze(1)
        elif image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        print(f"[DEBUG] Final pixel_values shape: {pixel_values.shape}")
        print(f"[DEBUG] Final image_grid_thw shape: {image_grid_thw.shape}")

        # Get image features from vision encoder
        # Returns concatenated embeddings: [total_tokens, hidden_dim]
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)

        # Split embeddings by image
        # Each image has t*h*w tokens where grid_thw = [t, h, w]
        embeddings_list = []
        start_idx = 0
        for t, h, w in image_grid_thw:
            num_tokens = t.item() * h.item() * w.item()
            end_idx = start_idx + num_tokens
            embeddings_list.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx

        return embeddings_list

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for WBL VLM.

        Args:
            input_ids: Input token IDs
            positions: Position IDs
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed embeddings (optional)
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions (T, H, W)
        """
        # Handle image embeddings
        if inputs_embeds is None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)

            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)

                # Replace image tokens with image embeddings
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and tokens mismatch: {n_image_features} features vs {n_image_tokens} tokens"
                    )

                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Forward through language model
        hidden_states = self.language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute output logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        This handles weight loading for both vision and language components.
        """
        # Separate weights into vision and language
        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            original_name = name

            # Route weights to appropriate component
            if name.startswith("model.visual.") or name.startswith("visual."):
                # Vision encoder weights
                if name.startswith("model.visual."):
                    name = name.replace("model.visual.", "visual.")
                vision_weights.append((name, loaded_weight))
            elif (name.startswith("model.language_model.") or
                  name.startswith("language_model.") or
                  name.startswith("lm_head.")):
                # Language model weights
                # WBLForCausalLM expects weights without "language_model." prefix
                # It internally has "model." and "lm_head." structure
                if name.startswith("model.language_model."):
                    # "model.language_model.layers.X..." -> "model.layers.X..."
                    name = name.replace("model.language_model.", "model.")
                elif name.startswith("language_model."):
                    # "language_model.model.layers.X..." -> "model.layers.X..."
                    name = name.replace("language_model.", "")
                elif name.startswith("lm_head."):
                    # "lm_head.weight" -> "lm_head.weight" (no change)
                    pass
                language_weights.append((name, loaded_weight))
            else:
                logger.warning(f"Unknown weight: {original_name}")

        # Load vision weights directly
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, loaded_weight in vision_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                logger.debug(f"Vision parameter {name} not found, skipping")

        # Load language model weights using WBLForCausalLM's load_weights
        # This handles the complex MoE weight fusion logic
        language_loaded = self.language_model.load_weights(language_weights)
        # Add "language_model." prefix to the returned parameter names
        loaded_params.update(f"language_model.{name}" for name in language_loaded)

        return loaded_params

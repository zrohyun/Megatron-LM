import torch
from dataclasses import dataclass
from typing import Union, Optional

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module, ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push


@dataclass
class VisionProjectorSubmodules:
    """Adapter sub-modules."""
    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class VisionProjector(MegatronModule):
    """ Adaptor  """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: VisionProjectorSubmodules,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__(config=config)
        
        self.hidden_size = input_size * (spatial_merge_size**2)

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )
        
        self.activation_func = config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )


    def forward(self, hidden_states: torch.Tensor, window_index: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """ Forward pass."""
        hidden_states = self.layernorm(hidden_states).view(-1, self.hidden_size)
        
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, _ = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")
        
        nvtx_range_push(suffix="activation")
        intermediate_parallel = self.activation_func(intermediate_parallel)
        nvtx_range_pop(suffix="activation")
        
        nvtx_range_push(suffix="linear_fc2")
        output, _ = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")
        
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            output = output[reverse_indices, :].contiguous()

        # the encoder produces "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        encoder_output = make_viewless_tensor(
            inp=output, requires_grad=True, keep_graph=True
        )

        return encoder_output

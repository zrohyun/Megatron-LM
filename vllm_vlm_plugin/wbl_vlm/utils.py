"""Utility functions for WBL VLM vLLM plugin."""

# Import utility functions from vLLM
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)

__all__ = [
    "init_vllm_registered_model",
    "maybe_prefix",
]

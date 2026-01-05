def _patch_transformers_version_check():
    """
    Patch vLLM's transformers version check for MoE models.

    vLLM requires transformers>=5.0.0.dev0 for MoE models, but we're using
    transformers 4.57.1. This monkey patch disables the version check since
    our custom WBL implementation handles MoE correctly.
    """
    try:
        from vllm.model_executor.models.transformers import Base

        # Store original check_version method
        original_check_version = Base.check_version

        # Create patched version that skips MoE check
        def patched_check_version(self, min_version: str, feature_name: str):
            if "MoE models support" in feature_name:
                # Skip the version check for MoE models
                return
            # Call original check for other features
            return original_check_version(self, min_version, feature_name)

        # Apply patch
        Base.check_version = patched_check_version

    except Exception as e:
        # If patch fails, log warning but don't crash
        import warnings
        warnings.warn(f"Failed to patch transformers version check: {e}")


def _patch_processor_tokenizer_check():
    """
    Patch Qwen2VL Processor's tokenizer type check to accept PreTrainedTokenizerFast.

    The Qwen2VL processor expects Qwen2Tokenizer or Qwen2TokenizerFast, but our
    checkpoint uses PreTrainedTokenizerFast. This patch allows the processor to
    accept any tokenizer type that has the required functionality.
    """
    try:
        from transformers.processing_utils import ProcessorMixin

        # Store original method
        original_check_argument = ProcessorMixin.check_argument_for_proper_class

        def patched_check_argument(self, attribute_name: str, arg):
            """Skip tokenizer type check - trust that it has the right methods"""
            if attribute_name == "tokenizer":
                # Skip the strict type check for tokenizer
                return
            # Call original check for other attributes
            return original_check_argument(self, attribute_name, arg)

        # Apply patch
        ProcessorMixin.check_argument_for_proper_class = patched_check_argument

    except Exception as e:
        import warnings
        warnings.warn(f"Failed to patch processor tokenizer check: {e}")


def register():
    # Apply patches before importing vLLM registry
    _patch_transformers_version_check()
    _patch_processor_tokenizer_check()

    from vllm import ModelRegistry

    # Register the vLLM-native WBL VLM model
    ModelRegistry.register_model(
        "WBLVLMoEForCausalLM",
        "wbl_vlm.wbl_vllm:WBLVLMForConditionalGeneration",
    )

    # Register VaetkiVL model (same implementation)
    ModelRegistry.register_model(
        "VaetkiVLForCausalLM",
        "wbl_vlm.wbl_vllm:WBLVLMForConditionalGeneration",
    )


# Auto-register when module is imported
register()

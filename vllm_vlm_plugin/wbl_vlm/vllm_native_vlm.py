from vllm.model_executor.models.transformers import (
    TransformersMultiModalMoEForCausalLM,
)


class WBLVLMoEForCausalLMNative(TransformersMultiModalMoEForCausalLM):
    @staticmethod
    def check_version(min_version: str, feature: str) -> None:
        # WBL VLM uses a custom HF model implementation, so we do not rely on
        # transformers' built-in MoE support. Skip the version gate.
        return None

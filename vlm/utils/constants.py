"""constants"""

from typing import List

IGNORE_INDEX = -100

######### dataset ########
DEFAULT_DATASET_NAME = "default"

DEFAULT_DATASET_CONFIG = "sft_dataset_config.json"

SFT_SUPPORT_DATA_TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


class SFTDataFormats(object):
    """sft data formats"""
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"


class DataRoles(object):
    """data roles"""
    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"
    SYSTEM = "system"


class Placeholder(object):
    """ Placeholders """
    IMAGE = "<image>"
    VIDEO = "<video>"


######## training args ########
class TrainingPhase(object):
    """Training phase"""
    PRETRAIN = "pretrain"
    SFT = "sft"


######## built-in models #######
# Using List[str] instead of list[str] to ensure compatibility with older versions of Python(<3.9)
class _BaseFamilies(object):
    @classmethod
    def names(cls) -> List[str]:
        """Return a list of all string names defined in the class and its subclasses"""
        string_names = [value for name, value in vars(cls).items()
                        if isinstance(value, str) and not name.startswith("__")]
        return string_names


class LanguageModelFamilies(_BaseFamilies):
    """Language model families"""
    QWEN3 = "qwen3"
    DEEPSEEK = "deepseek"
    GPT_FOR_WBL = "gpt_for_wbl"


class VideoLanguageModelFamilies(_BaseFamilies):
    """Video language model families"""
    STDIT = "stdit"
    STDIT3 = "stdit3"


class VisionLanguageModelFamilies(_BaseFamilies):
    """Vision language model families"""
    RICE_QWEN = "rice_qwen"
    RICE_GPT = "rice_gpt"

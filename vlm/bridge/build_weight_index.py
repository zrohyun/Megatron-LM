import os
import json
import argparse

from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils import SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from huggingface_hub import split_torch_state_dict_into_shards

parser = argparse.ArgumentParser()
parser.add_argument("--hf-model", type=str)
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    state_dict = model.state_dict()
    weights_name = SAFE_WEIGHTS_NAME
    filename_pattern = weights_name.replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size="5GB"
    )
    if state_dict_split.is_sharded:
        index = {
            "metadata": {"total_parameters": model.num_parameters(), **state_dict_split.metadata},
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(args.hf_model, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
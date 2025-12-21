# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
import torch
import sys
import time
import warnings
from argparse import Namespace

import torch

from megatron.core.inference.communication_utils import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)

import asyncio
import json
from typing import Any, Dict, Optional, OrderedDict, List
from dataclasses import dataclass

from examples.inference.gpt.utils import add_common_inference_args, build_requests
from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

from PIL import Image
from transformers import AutoProcessor

from vlm.models.rice_gpt.rice_gpt_provider import rice_gpt_model_provider
from vlm.train.arguments import extra_vlm_train_args_provider


# IMAGE_TOKEN = "<image>"
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]


def add_static_inference_args(parser):
    """Static inference arguments."""

    add_common_inference_args(parser)

    group = parser.add_argument_group(title="Static inference")
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        dest="max_batch_size",
        help="Deprecated, use `--inference-max-requests` instead",
    )
    group.add_argument(
        "--stream", action="store_true", default=False, help="Stream output tokens"
    )

    group.add_argument("--model-name", default="rice-gpt-7b-a1b")
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint",
        action="store_true",
        default=False,
    )
    (
        group.add_argument(
            "--trainable-modules",
            default=["all"],
            nargs="*",
            help="choices: all, language_model, vision_projection, vision_model, "
            "language_expert_linear, vision_expert_linear",
        ),
    )

    return parser


class VLMInferenceWrapper(GPTInferenceWrapper):
    """Inference wrapper for VLMs"""

    def prep_model_for_inference(self, prompts_tokens: Optional[torch.Tensor] = None):
        """A utility function for preparing model for inference

        The function gets called once before the auto regressive inference loop.
        It puts the model in eval mode.

        Args:
            prompts_tokens (torch.Tensor): Deprecated, will be removed in `megatron-core` 0.13
        """
        if prompts_tokens is not None:
            warnings.warn(
                "Passing `prompts_tokens` is deprecated and this argument will be ignored."
                "This parameter will be removed in `megatron-core` 0.13."
            )

        super().prep_model_for_inference()

        # For TP only model both is_pp_first_stage and _is_pp_last_stage returns True
        # set ignore_virtual=True since vpp is not used in inference
        self.model_is_pipeline_parallel = not (
            is_pipeline_first_stage(self.pp_group)
            and is_pipeline_last_stage(self.pp_group)
        )

        self._recv_only_vision_embeds = False
        pp_rank = self.pp_group.rank()
        # Checks if the previous stage only has a vision encoder, and that the current stage
        # has part of the LM decoder. In this case, the current stage should only receive
        # vision embeddings.
        if pp_rank > 0:
            self._recv_only_vision_embeds = (
                False  # TODO: Implement new logic for vision embeddings
            )

        # Checks if the current stage only has a vision encoder
        self._encoder_only = False  # TODO: Implement new logic for encoder-only stages

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        images: torch.Tensor,
        image_grid_thw: torch.Tensor,
        decoder_seq_length: int,
    ):
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
            num_img_embeddings_per_tile (int): The number of image embeddings per tile
            images (torch.Tensor): The image embeddings
            num_tiles (torch.Tensor): The number of tiles for each input image
            decoder_seq_length (int): The decoder sequence length
        """
        inference_input = super().prep_inference_input(prompts_tokens)

        # batch_size, max_sequence_length = prompts_tokens.shape
        # self.inference_context = StaticInferenceContext(
        #     batch_size, max_sequence_length
        # )

        inference_input["images"] = images
        inference_input["image_grid_thw"] = image_grid_thw
        inference_input["decoder_seq_length"] = decoder_seq_length

        return inference_input

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length.

        Returns:
            Dict[str, Any]: A dict of inputs that will be used by your model in the forward step
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        images = inference_input["images"]
        image_grid_thw = inference_input["image_grid_thw"]
        decoder_seq_length = inference_input["decoder_seq_length"]

        tokens2use = tokens[:, context_start_position:context_end_position]
        # positions2use = position_ids[:, context_start_position:context_end_position]

        return {
            "tokens": tokens2use,
            "position_ids": None,
            "images": images,
            "image_grid_thw": image_grid_thw,
            "decoder_seq_length": decoder_seq_length,
        }

    def _forward(self, inference_input: Dict[str, Any]):
        """Runs a forward pass of the model.

        Args:
            inference_input(Dict[str, Any]): The input data.

        Returns:
            The model output logits.
        """
        images = inference_input["images"]
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        image_grid_thw = inference_input["image_grid_thw"]

        output = self.model(
            images=images,
            image_grid_thw=image_grid_thw,
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,  # TODO: ????
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        return logits

    def run_one_forward_step(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """The forward pass of the model for inference

        Args:
            inference_input (Dict[str, Any]): A dict containing the inputs for the VLM model

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size].
            The logits are returned only in the last pipeline stage for PP models.
        """
        tokenizer = get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        tokens = inference_input["tokens"]
        num_image_tokens = (tokens == image_token_id).sum().item()
        image_grid_thw = inference_input["image_grid_thw"]
        decoder_seq_length = inference_input["decoder_seq_length"]
        num_tokens = tokens.size(1)

        num_img_embeddings = int((image_grid_thw.prod(dim=-1).sum() / 4).item())
        recv_buffer_seq_len = None
        if num_image_tokens > 0:
            # When there are image tokens and this stage only receives vision embeddings,
            # adjust the recv buffer seq length to match the image embeddings sequence length.
            # If there are image tokens and this stage receives full embeddings, make sure we
            # compensate for expansion of image tokens.
            # Note that this will set a recv_buffer_seq_len for the encoder stage,
            # this length is irrelevant since that recv buffer is never allocated.
            if self._recv_only_vision_embeds:
                recv_buffer_seq_len = num_img_embeddings
            else:
                # recv_buffer_seq_len = min(
                #     num_img_embeddings + num_tokens - num_image_tokens, decoder_seq_length
                # )
                recv_buffer_seq_len = min(  # leejh1230
                    num_tokens, decoder_seq_length
                )
        elif self._recv_only_vision_embeds:
            # If this stage only receives vision embeddings and there are no image tokens
            # we won't run the encoder and therefore shouldn't try to recv.
            recv_buffer_seq_len = 0

        # If the pipeline stage only has a vision encoder, then it only needs to
        # run when there are image tokens
        if not (self._encoder_only and num_image_tokens == 0):
            output = super().run_one_forward_step(
                inference_input, recv_buffer_seq_len=recv_buffer_seq_len
            )
        else:
            output = None
        logits = output

        # On the first inference iteration, we compute image tokens.
        # On every PP stage(although inference params should only matter for decoder),
        # update the sequence length offset by the number of image tokens.
        if num_tokens > 1 and num_image_tokens > 0:
            if "image_tokens_count" not in self.inference_context.key_value_memory_dict:
                self.inference_context.key_value_memory_dict["image_tokens_count"] = (
                    num_image_tokens
                )

            # # if num_img_embeddings + num_tokens - num_image_tokens > decoder_seq_length:
            # if num_tokens > decoder_seq_length:
            #     self.inference_context.sequence_len_offset += decoder_seq_length - num_tokens
            # else:
            #     self.inference_context.sequence_len_offset += (
            #         self.inference_context.key_value_memory_dict["image_tokens_count"]
            #         - num_image_tokens
            #     )

        return logits


def get_attention_mask(seq_length: int) -> torch.Tensor:
    """Constructs an attention mask given the input sequence length."""
    attention_mask = torch.tril(
        torch.ones((1, seq_length, seq_length), device=torch.cuda.current_device())
    ).view(1, 1, seq_length, seq_length)  # TODO: ????

    # Convert to boolean
    attention_mask = attention_mask < 0.5

    return attention_mask


@dataclass(kw_only=True)
class VLMInferenceRequest(InferenceRequest):
    """Class for a VLM inference request"""

    imgs: torch.Tensor
    image_grid_thw: torch.Tensor
    decoder_seq_length: int


class VLMTextGenerationController(TextGenerationController):
    """The text generation controller for VLMs"""

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
        **kwargs,
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Currently only supports batch size 1 inference.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.
        """
        assert len(active_requests) == 1, (
            f"VLM inference currently only supports batch size 1"
        )

        request = list(active_requests.values())[0]

        assert isinstance(request, VLMInferenceRequest), (
            f"Found inference request of type {type(request)}, expected VLMInferenceRequest"
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens,
            request.imgs,
            request.image_grid_thw,
            request.decoder_seq_length,
        )

        return inference_input


def get_inference_engine(
    args: Namespace, model: MegatronModule, processor
) -> StaticInferenceEngine:
    """Utility to get the relevant backend for running inference
    This function will automatically choose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.
    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model .
    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_requests=args.inference_max_batch_size,
        inference_max_seq_length=args.inference_max_seq_length,
        nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill,
        fp8=args.fp8,
    )

    inference_wrapped_model = VLMInferenceWrapper(model, inference_wrapper_config)
    controller = VLMTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(
        text_generation_controller=controller, max_batch_size=1
    )


async def generate(
    inference_engine: StaticInferenceEngine,
    sampling_params: SamplingParams,
    prompts: List[str],
) -> List[InferenceRequest]:
    async def collect_stream(prompt, request_id, stream_generator):
        print(f"Request {request_id}: {prompt}", end="", flush=True)
        prev_idx = 0
        async for output in stream_generator:
            print(output.generated_text[prev_idx:], end="", flush=True)
            prev_idx = len(output.generated_text)
        print()

    request_ids: List[int] = [
        inference_engine.add_request(
            prompt=prompt, sampling_params=sampling_params, streaming=True
        )
        for prompt in prompts
    ]
    stream_generators = [
        inference_engine.get_stream_generator(request_id) for request_id in request_ids
    ]

    tasks = [
        asyncio.create_task(collect_stream(prompt, request_id, stream_generator))
        for (prompt, request_id, stream_generator) in zip(
            prompts, request_ids, stream_generators
        )
    ]

    await inference_engine.run_engine_async()
    await asyncio.gather(*tasks)

    results: List[InferenceRequest] = [
        inference_engine.scheduler.completed_request_pool[request_id]
        for request_id in request_ids
    ]

    return results


def get_sample(user_prompt: str, processor):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{IMAGE_TOKEN}\n{user_prompt}"},
    ]

    img = Image.open("./tmp/image3.jpg").convert("RGB")
    # img = Image.new("RGB", (224, 224), (0, 0, 0))
    # img = np.array(img)

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_text = chat_text.replace("<image>", IMAGE_TOKEN_WITH_TAGS)
    if chat_text.endswith("\n"):
        chat_text = chat_text[:-1]

    print("&&" * 100)
    print(chat_text)
    print("&&" * 100)

    proc_out = processor(
        text=chat_text,
        images=[img],
        # return_tensors="pt",
    )
    pixel_values = proc_out["pixel_values"]
    image_grid_thw = proc_out["image_grid_thw"]
    num_tiles = [len(image_grid_thw)]

    input_ids = proc_out["input_ids"][0]
    # attention_mask = proc_out["attention_mask"][0].logical_not()

    return {
        "conv": chat_text,
        "input_ids": input_ids,
        # "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "num_tiles": num_tiles,
    }


@torch.inference_mode()
def main():
    """Main program."""
    # === DEBUGPY (rank0 전용 attach) ===========================
    import os, socket

    if os.getenv("DEBUGPY", "0") == "1":
        import debugpy

        port = int(os.getenv("DEBUGPY_PORT", "5678"))
        rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
        if rank == 0:
            try:
                debugpy.listen(
                    ("127.0.0.1", port)
                )  # ← dev container 면 loopback이 가장 안전
                print(
                    f"[debugpy] waiting on 127.0.0.1:{port} (rank={rank}) ...",
                    flush=True,
                )
                debugpy.wait_for_client()
                print("[debugpy] attached.", flush=True)
            except Exception as e:
                print(f"[debugpy] listen failed: {e}", flush=True)
    # ===========================================================

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(
        extra_args_provider=add_static_inference_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )

    args = get_args()

    if args.max_batch_size is not None:
        warnings.warn(
            f"`--max-batch-size` has been deprecated in favor of `--inference-max-requests`."
        )
        args.inference_max_batch_size = max(
            args.max_batch_size, args.inference_max_batch_size
        )

    # Set up model and load checkpoint
    def wrapped_model_provider(
        pre_process, post_process, add_encoder=True, add_decoder=True
    ):
        return rice_gpt_model_provider(
            pre_process, post_process, add_encoder=add_encoder, add_decoder=add_decoder
        )

    model = get_model(wrapped_model_provider, wrap_with_ddp=False)

    # .pt 파일 캐시 경로 설정
    cached_file_path_list = Path(args.load).resolve().glob("iter_*_model_cached.pt")
    if len(list(cached_file_path_list)) == 0:
        cached_pt_path = None
    else:
        cached_pt_path = list(Path(args.load).resolve().glob("iter_*_model_cached.pt"))[
            0
        ]
    # cached_pt_path = os.path.join(args.load, f"iter_{iteration:07d}_model_cached.pt") if args.load else None
    # cached_pt_path = os.path.join(args.load, "model_cached.pt") if args.load else None
    print(cached_pt_path)

    if cached_pt_path and os.path.exists(cached_pt_path):
        # 캐시된 .pt 파일이 있으면 빠르게 로드
        import datetime

        start_ts = datetime.datetime.now()
        print_rank_0(f"[{start_ts}] Loading cached checkpoint from {cached_pt_path}")
        state_dict = torch.load(cached_pt_path, map_location="cuda", weights_only=False)
        model[0].load_state_dict(state_dict, strict=False)
        end_ts = datetime.datetime.now()
        print_rank_0(
            f"[{end_ts}] Finished loading cached checkpoint from {cached_pt_path}"
        )
    else:
        # 기존 방식으로 로드
        import datetime

        start_ts = datetime.datetime.now()
        print_rank_0(f"[{start_ts}] Loading checkpoint from {args.load}")
        load_checkpoint(model, None, None, strict=True)
        end_ts = datetime.datetime.now()
        print_rank_0(f"[{end_ts}] Finished loading checkpoint from {args.load}")

        # rank 0에서만 .pt 파일로 캐시 저장
        if cached_pt_path and torch.distributed.get_rank() == 0:
            if not os.path.exists(
                Path(args.load) / "latest_checkpointed_iteration.txt"
            ):
                iteration = (
                    open(Path(args.load) / "latest_checkpointed_iteration.txt", "r")
                    .read()
                    .strip()
                )
            else:
                iteration = 0
            cached_pt_path = os.path.join(
                args.load, f"iter_{iteration:07d}_model_cached.pt"
            )
            print_rank_0(f"Saving cached checkpoint to {cached_pt_path}")
            # CPU로 이동하여 저장 (GPU 메모리 절약 및 호환성)
            start_ts = datetime.datetime.now()
            print_rank_0(f"[{start_ts}] Saving checkpoint to {cached_pt_path}")
            state_dict_cpu = {
                k: v.cpu() if v is not None else None
                for k, v in model[0].state_dict().items()
            }
            torch.save(state_dict_cpu, cached_pt_path)
            end_ts = datetime.datetime.now()
            print_rank_0(f"[{end_ts}] Finished saving checkpoint to {cached_pt_path}")
            del state_dict_cpu

        # 모든 rank가 저장 완료를 기다림
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    model = model[0]
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.tokenizer_model, trust_remote_code=True
    )

    inference_engine = get_inference_engine(args, model, processor)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        top_n_logprobs=args.top_n_logprobs,
    )

    user_prompt = "Describe this image."
    # user_prompt = "Hey, are you conscious? Can you talk to me?"
    sample = get_sample(user_prompt=user_prompt, processor=processor)

    inference_request = VLMInferenceRequest(
        request_id=inference_engine.get_new_request_id(),
        prompt=sample["conv"],
        prompt_tokens=sample["input_ids"],
        sampling_params=sampling_params,
        imgs=sample["pixel_values"].to("cuda"),
        image_grid_thw=sample["image_grid_thw"].to("cuda"),
        decoder_seq_length=args.decoder_seq_length,
    )

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=["warmup"],
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
    start_time = time.perf_counter()

    results: List[InferenceRequest] = inference_engine.generate(
        inference_requests=[inference_request]
    )

    end_time = time.perf_counter()
    latency = end_time - start_time

    if torch.distributed.get_rank() == 0 and args.output_path:
        results_output = {}
        for idx, result in enumerate(results):
            result_dict = {
                "input_prompt": result.prompt,
                "generated_text": result.generated_text,
                "generated_tokens": result.generated_tokens.tolist(),
                "tpot": result.tpot,
                "latency": latency,
            }
            if sampling_params.top_n_logprobs > 0:
                result_dict["generated_top_n_logprobs"] = (
                    result.generated_top_n_logprobs
                )
            if sampling_params.return_log_probs:
                response_logprobs = result.prompt_log_probs + result.generated_log_probs
                result_dict["logprobs"] = response_logprobs
            results_output[result.request_id] = result_dict

        with open(args.output_path, "w") as f:
            json.dump(results_output, f)

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:
        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map results by their prompt.
        from collections import defaultdict

        unique_prompt_map = defaultdict(list)
        for result_idx, result in enumerate(results):
            unique_prompt_map[result.prompt].append(result_idx)

        # Print unique prompts + outputs.
        for unique_idx, (prompt_text, result_idxs) in enumerate(
            unique_prompt_map.items()
        ):
            result_idx = result_idxs[0]
            result = results[result_idx]
            generated_text = result.generated_text.replace("\n", "\\n")
            print(
                f"{unique_idx}/{len(unique_prompt_map)} [{len(result_idxs)}]. {prompt_text} "
                f"... {generated_text}"
            )

    stats = torch.cuda.memory_stats()
    print_rank_0(
        "static | cg %d | %s | reqs %d [ batch %d ] ... mem %.1f/%.1f ... time %.3f."
        % (
            args.enable_cuda_graph,
            (
                f"<user prompts>"
                if args.prompts
                else "<auto prompts> %s, %d, %.1e, %.1e"
                % (
                    "(%s)" % " ".join(map(str, args.num_tokens_to_prompt)),
                    args.num_tokens_to_generate,
                    args.incoming_requests_duration,
                    args.incoming_requests_per_sec,
                )
            ),
            len(results),
            args.inference_max_batch_size,
            stats["allocated_bytes.all.peak"] / (1024**3),
            stats["reserved_bytes.all.peak"] / (1024**3),
            latency,
        )
    )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

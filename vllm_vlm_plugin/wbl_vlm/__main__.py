if __name__ == "__main__":
    import os

    from PIL import Image
    from vllm import LLM, SamplingParams

    ckpt_path = os.getenv("CHECKPOINT_PATH") or os.getenv(
        "CKPT_PATH",
        "/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_4/checkpoints/v1_rice_gpt_stage_4_7b_a1b_gbs_128_lr_3e6_warmup_0_02_18M_tk5_no_capacity__v9__v6/iter_0144531-HF"
        # "/mnt/checkpoint/mlx_ncai_backup/vlm_stage_2/checkpoints/"
        # "v3_rice_gpt_stage_4_7b_a1b_gbs_128_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v12__v11/"
        # "iter_0010000-HF",
        # "v12_rice_gpt_stage_2_7b_a1b_gbs_128_vv_and_stage1_data_lr_2e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v11/"
        # "iter_0037500-HF",
    )

    max_model_len_env = os.getenv("SEQ_LENGTH") or os.getenv("MAX_MODEL_LEN")
    max_model_len = int(max_model_len_env) if max_model_len_env else None
    tensor_parallel_size = int(
        os.getenv("TP_SIZE") or os.getenv("TENSOR_PARALLEL_SIZE", "1")
    )
    pipeline_parallel_size = int(os.getenv("PP_SIZE", "1"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

    llm = LLM(
        model=ckpt_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
        additional_config={"precision_level": 2},
        limit_mm_per_prompt={"image": 10},  # Support multimodal
        disable_custom_all_reduce=True,
        enforce_eager=True,  # Disable torch compilation to avoid MoE issues
    )

    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    top_k = int(os.getenv("TOP_K", "1"))
    max_tokens = int(os.getenv("NUM_TOKENS", "128"))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    prompt = os.getenv("USER_PROMPT") or os.getenv("PROMPT", "Describe the image.")
    image_path = os.getenv("IMAGE_PATH", "./test_images/mario.jpg")

    # Add image placeholder token to prompt
    # The model uses <|image_pad|> as the image token
    if "<|image_pad|>" not in prompt:
        prompt = f"<|image_pad|>{prompt}"

    image = Image.open(image_path).convert("RGB")
    requests = [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }]

    outputs = llm.generate(requests, sampling_params)
    print(f"Generated text: {outputs[0].outputs[0].text!r}")

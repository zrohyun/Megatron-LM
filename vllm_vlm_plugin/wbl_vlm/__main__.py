if __name__ == "__main__":
    import os
    import sys
    import base64
    from io import BytesIO

    from PIL import Image
    from vllm import LLM, SamplingParams

    # Force V0 engine for stability
    os.environ.setdefault("VLLM_USE_V1", "0")

    def pil_image_to_data_url(image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    ckpt_path = os.getenv("CHECKPOINT_PATH") or os.getenv(
        "CKPT_PATH",
        "/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_x/checkpoints/v1_rice_gpt_stage_x_7b_a1b_gbs_128_lr_1e5_vlrmult_0_2_warmup_0_1_decay_0_01_tk5__v1__v9__v6/iter_0027578-HF"
        # "/mnt/checkpoint/mlx_ncai_backup/multimodal/outputs/vlm_stage_4/checkpoints/v1_rice_gpt_stage_4_7b_a1b_gbs_128_lr_3e6_warmup_0_02_18M_tk5_no_capacity__v9__v6/iter_0144531-HF"
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
        enforce_eager=False,  # Disable torch compilation to avoid MoE issues
    )

    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    top_k = int(os.getenv("TOP_K", "1"))
    max_tokens = int(os.getenv("NUM_TOKENS", "128"))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    # Test 1: Single Image Inference
    print("=" * 80)
    print("Test 1: Single Image Inference")
    print("=" * 80)

    prompt_single = os.getenv("USER_PROMPT") or os.getenv("PROMPT", "Describe the image.")
    image_path_single = os.getenv("IMAGE_PATH", "../test_images/mario.jpg")

    # Add image placeholder token to prompt
    # The model uses <|image_pad|> as the image token
    if "<|image_pad|>" not in prompt_single:
        prompt_single = f"<|image_pad|>{prompt_single}"

    image_single = Image.open(image_path_single).convert("RGB")
    requests_single = [{
        "prompt": prompt_single,
        "multi_modal_data": {"image": image_single},
    }]

    print(f"Prompt: {prompt_single!r}")
    print(f"Image: {image_path_single}")

    outputs_single = llm.generate(requests_single, sampling_params)
    print(f"Generated text: {outputs_single[0].outputs[0].text!r}")
    print()

    # Test with chat template (using base64 image URL)
    print("--- Single Image with Chat Template ---")
    image_url_single = pil_image_to_data_url(image_single)
    messages_single = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url_single}},
                {"type": "text", "text": prompt_single}
            ]
        }
    ]

    outputs_single_chat = llm.chat(
        messages_single,
        sampling_params=sampling_params,
    )
    print(f"Generated text(w/ chat template): {outputs_single_chat[0].outputs[0].text!r}")
    print()

    # Test 2: Multi Image Inference
    print("=" * 80)
    print("Test 2: Multi Image Inference")
    print("=" * 80)

    # Get multiple image paths from environment or use defaults
    image_paths_multi = os.getenv("IMAGE_PATHS", "").split(",") if os.getenv("IMAGE_PATHS") else [
        "../test_images/mario.jpg",
        "../test_images/test_coco.jpg",
    ]
    image_paths_multi = [p.strip() for p in image_paths_multi if p.strip()]

    prompt_multi = os.getenv("MULTI_PROMPT", "Compare and describe these images.")

    # Add multiple image placeholder tokens
    num_images = len(image_paths_multi)
    if "<|image_pad|>" not in prompt_multi:
        image_tokens = "<|image_pad|>" * num_images
        prompt_multi = f"{image_tokens}{prompt_multi}"

    images_multi = [Image.open(path).convert("RGB") for path in image_paths_multi]
    requests_multi = [{
        "prompt": prompt_multi,
        "multi_modal_data": {"image": images_multi},
    }]

    print(f"Prompt: {prompt_multi!r}")
    print(f"Images: {image_paths_multi}")
    print(f"Number of images: {num_images}")

    outputs_multi = llm.generate(requests_multi, sampling_params)
    print(f"Generated text: {outputs_multi[0].outputs[0].text!r}")
    print()

    # Test with chat template (using base64 image URLs)
    print("--- Multi Image with Chat Template ---")
    image_urls_multi = [pil_image_to_data_url(img) for img in images_multi]

    # Build content list with multiple images
    content_multi = []
    for img_url in image_urls_multi:
        content_multi.append({"type": "image_url", "image_url": {"url": img_url}})
    content_multi.append({"type": "text", "text": prompt_multi})

    messages_multi = [
        {
            "role": "user",
            "content": content_multi
        }
    ]

    outputs_multi_chat = llm.chat(
        messages_multi,
        sampling_params=sampling_params,
    )
    print(f"Generated text(w/ chat template): {outputs_multi_chat[0].outputs[0].text!r}")
    print()
    print("=" * 80)

    # ERROR 01-05 20:03:30 [core_client.py:598] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
    # Known issue: https://github.com/vllm-project/vllm/issues/23517

    # Clean shutdown to avoid "Engine core proc died unexpectedly" error
    print("\nShutting down vLLM engine gracefully...")
    try:
        # Explicitly destroy the LLM instance
        del llm
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Warning during cleanup: {e}")

    # Exit cleanly
    sys.exit(0)

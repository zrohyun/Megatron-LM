#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Megatron 분산 체크포인트를 단일 .pt 파일로 변환

실행 방법:
    cd /workspace/Megatron-LM-MMM
    ./run_convert_checkpoint.sh
"""

import os
import sys
import time
import gc
import argparse

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)

from megatron.training import get_args, get_model, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

from vlm.models.rice_gpt.rice_gpt_provider import rice_gpt_model_provider


def add_convert_args(parser):
    """변환 관련 인자 추가"""
    group = parser.add_argument_group(title='Checkpoint Conversion')
    group.add_argument('--output-path', type=str, default=None,
                       help='Output .pt file path. Default: {checkpoint_dir}/model_cached.pt')
    group.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file')
    return parser


def main():
    """메인 함수"""
    # Megatron 초기화
    initialize_megatron(
        extra_args_provider=add_convert_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )
    
    args = get_args()
    
    # 모델 빌드 및 체크포인트 로드 (iteration 정보 필요)
    print_rank_0("\n" + "="*60)
    print_rank_0("Megatron Checkpoint to PT Converter")
    print_rank_0("="*60)
    print_rank_0(f"Input checkpoint: {args.load}")
    print_rank_0("="*60 + "\n")
    
    print_rank_0("Step 1: Building model and loading checkpoint...")
    start_time = time.time()
    
    def wrapped_model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
        return rice_gpt_model_provider(pre_process, post_process, add_encoder=add_encoder, add_decoder=add_decoder)
    
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model, None, None, strict=False)
    
    # 출력 경로 결정 (iteration 번호 포함)
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.load, f"iter_{iteration:07d}_model_cached.pt")
        # output_path = os.path.join(args.load, "model_cached.pt")
    
    load_time = time.time() - start_time
    print_rank_0(f"Checkpoint loaded in {load_time:.2f}s")
    print_rank_0(f"Loaded iteration: {iteration}")
    print_rank_0(f"Output path: {output_path}")
    
    # 기존 파일 확인
    if os.path.exists(output_path) and not args.overwrite:
        print_rank_0(f"\n[ERROR] Output file already exists: {output_path}")
        print_rank_0("Use --overwrite to overwrite existing file")
        return
    
    # state_dict 추출
    print_rank_0("\nStep 2: Extracting state_dict...")
    start_time = time.time()
    
    model[0].eval()
    state_dict = model[0].state_dict()
    
    # CPU로 이동 (이식성 향상)
    state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
    
    extract_time = time.time() - start_time
    print_rank_0(f"State dict extracted in {extract_time:.2f}s")
    print_rank_0(f"Total keys: {len(state_dict_cpu)}")
    
    # 파일 크기 추정
    total_params = sum(v.numel() for v in state_dict_cpu.values())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict_cpu.values())
    print_rank_0(f"Total parameters: {total_params:,}")
    print_rank_0(f"Estimated file size: {total_bytes / (1024**3):.2f} GB")
    
    # PT 파일로 저장
    print_rank_0(f"\nStep 3: Saving to {output_path}...")
    start_time = time.time()
    
    # rank 0에서만 저장
    if torch.distributed.get_rank() == 0:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        torch.save(state_dict_cpu, output_path)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    save_time = time.time() - start_time
    print_rank_0(f"Saved in {save_time:.2f}s")
    
    # 실제 파일 크기 확인
    if os.path.exists(output_path):
        actual_size = os.path.getsize(output_path)
        print_rank_0(f"Actual file size: {actual_size / (1024**3):.2f} GB")
    
    # 정리
    del state_dict, state_dict_cpu
    gc.collect()
    torch.cuda.empty_cache()
    
    print_rank_0(f"\n{'='*60}")
    print_rank_0("Conversion Complete!")
    print_rank_0(f"{'='*60}")
    print_rank_0(f"Output: {output_path}")
    print_rank_0(f"Total time: {load_time + extract_time + save_time:.2f}s")
    print_rank_0(f"{'='*60}\n")


if __name__ == "__main__":
    main()

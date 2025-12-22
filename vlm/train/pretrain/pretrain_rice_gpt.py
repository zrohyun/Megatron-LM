"""default pretrain for generative models like GPTS"""

import os
import torch
from typing import Tuple, Optional
from functools import partial

from megatron.training import get_timers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.enums import AttnMaskType

from transformers import DataCollatorForSeq2Seq

from vlm.utils import constants, get_args
from vlm.models.qwen_vl.utils import get_inputs_on_this_cp_rank
from vlm.models import get_model_provider, get_model_family
from vlm.train.megatron_trainer import MegatronTrainer
from vlm.train.trainer_builder import register_model_trainer
from vlm.train.sft.utils import build_sft_data_collator, make_attn_mask_4d
from vlm.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader,
    is_first_or_last_stage,
    is_dataloader_rank
)
from vlm.data.multimodal.qwen2vl_task_encoder import Qwen2VLTaskEncoder

stimer = StragglerDetector()


image_token_id = 137179     # <|image_pad|>
video_token_id = 137180     # <|video_pad|>
vision_start_token_id = 137176      # <|vision_start|>


def model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        MCoreModel: The returned model
    """
    args = get_args()
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
    assert model_provider is not None, f'model provider for {args.model_name} not found'
    return model_provider(pre_process, post_process, add_encoder, add_decoder)


def get_batch(data_iterator):
    """Generate a batch"""
    imgs = None
    thw = None
    pixel_values_videos = None
    video_grid_thw = None
    tokens = None
    labels = None
    loss_mask = None
    packed_seq_params = None
    attn_mask = None
    position_ids = None

    args = get_args()

    # Dataloader doesn't run on the middle stages in a pipeline parallel model.
    pp_size = get_pipeline_model_parallel_world_size()
    if not is_first_or_last_stage(pp_size):
        # Note these are all set to None above.
        return (
            imgs,
            thw,
            pixel_values_videos,
            video_grid_thw,
            tokens,
            position_ids,
            attn_mask,
            labels,
            loss_mask,
            attn_mask_type,
            packed_seq_params
        )

    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        if isinstance(data.get('tokens'), torch.Tensor):
            orig_dtype = data['tokens'].dtype
            if data['tokens'].dtype != torch.long:
                print(f"[WARN] tokens dtype {orig_dtype} -> force cast to torch.long; shape={tuple(data['tokens'].shape)}")
                data['tokens'] = data['tokens'].to(torch.long)
        if isinstance(data.get('labels'), torch.Tensor) and data['labels'].dtype != torch.long:
            data['labels'] = data['labels'].to(torch.long)

        assert isinstance(data['tokens'], torch.Tensor) and data['tokens'].dtype == torch.long, \
            f"Expected tokens torch.int64 but got {type(data['tokens'])} {getattr(data['tokens'],'dtype',None)}"
        assert isinstance(data['labels'], torch.Tensor) and data['labels'].dtype == torch.long, \
            f"Expected labels torch.int64 but got {type(data['labels'])} {getattr(data['labels'],'dtype',None)}"
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]

    has_video = video_token_id in tokens
    has_image = image_token_id in tokens
    
    # if has_image:
    imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
    thw = tensor_parallel.broadcast_data(["image_grid_thw"], data, torch.int32)["image_grid_thw"]
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(
            ["pixel_values_videos"],
            data,
            torch.float32)["pixel_values_videos"]
        video_grid_thw = tensor_parallel.broadcast_data(
            ["video_grid_thw"],
            data,
            torch.int32)["video_grid_thw"]

    packed_seq_params = None
    is_video = video_token_id in tokens

    labels = torch.roll(labels, shifts=-1, dims=1)
    loss_mask = (labels != -100).long()

    if cu_lengths.shape == torch.Size([1, 1]):
        for i in range(attn_mask.shape[0]):
            loss_mask[i, (attn_mask[i] == False).sum() - 1] = 0
            
        attn_mask = make_attn_mask_4d(attn_mask)
    else:
        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        # for i in range(cu_lengths.shape[0]):
        #     for j in range(1, cu_lengths[i].shape[0]):
        #         loss_mask[i, cu_lengths[i][j] - 1] = 0

        attn_mask = None
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )

    if args.context_parallel_size > 1:
        labels = get_inputs_on_this_cp_rank(labels.transpose(0, 1)).transpose(0, 1)
        loss_mask = get_inputs_on_this_cp_rank(loss_mask.transpose(0, 1)).transpose(0, 1)

    position_ids = None
    
    return (
        imgs,
        thw,
        pixel_values_videos,
        video_grid_thw,
        tokens,
        position_ids,
        attn_mask,
        labels,
        loss_mask,
        packed_seq_params
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """    
    args = get_args()
    
    # if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
    #     return loss_func_modelopt(loss_mask, output_tensor, model=model)
    
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    
        # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    global stimer
    with stimer(bdata=True):
        images, image_grid_thw, pixel_values_videos, video_grid_thw, \
        input_ids, position_ids, attention_mask, \
        labels, loss_mask, packed_seq_params \
            = get_batch(data_iterator)
            
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            images=images,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw
        )
 
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_dataset_provider(train_val_test_num_samples):
    """ Provides the datasets used by the trainer """

    args = get_args()
    task_encoder = Qwen2VLTaskEncoder(args)
    train_dataset = get_train_dataset(task_encoder)
    collator = build_sft_data_collator(DataCollatorForSeq2Seq)

    # iter 를 돌려보면, <class 'dict'> 형태로 나온다. 
    # 각 data 의 키는 (dict_keys(['__key__', '__restore_key__', '__subflavor__', 'tokens', 'labels', 'num_tiles', 'max_lengths', 'cu_lengths', 'attn_mask', 'imgs', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']))
    train_dataloader = get_train_loader(train_dataset, collator)
    return train_dataloader, None, None


@register_model_trainer(
    model_family=[
        constants.VisionLanguageModelFamilies.RICE_GPT],
        training_phase=constants.TrainingPhase.PRETRAIN)
def default_pretrain_trainer(train_args):
    """build trainer"""
    model_type = ModelType.encoder_or_decoder
    
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        model_type=model_type,
        forward_step_func=forward_step,
    )

    return trainer
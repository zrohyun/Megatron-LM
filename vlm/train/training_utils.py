"""
Pretrain utilities.
Modified from Megatron-LM/megatron/training.py, https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training.py
"""
import logging
from typing import Optional

import torch.distributed

from megatron.training.log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)

import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

try:
    from nvidia_resiliency_ext.inprocess import CallWrapper
except ImportError:
    CallWrapper = type(None)

from megatron.core.utils import (
    get_model_config,
)

from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import set_jit_fusion_options

from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.utils import (
    append_to_progress_log,
    print_rank_0,
)
from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_wandb_writer,
    get_one_logger,
)
from megatron.training import one_logger_utils, ft_integration
from megatron.training.training import (
    print_datetime, 
    preprocess_common_state_dict, 
    setup_model_and_optimizer,
    build_train_valid_test_data_iterators,
    evaluate_and_print_results,
    train,
)

from vlm.utils import get_args, initialize_megatron


def vision_scale_lr_cond(name, param):
    if name is None:
        return False
    
    if "vision_model." in name:
        print_rank_0("[VISION_LR] using vision lr for:", name)

    return "vision_model." in name


def pretrain(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
        store: an optional instance of torch.distributed.Store, to be used by
            torch.distributed.init_process_group
        inprocess_call_wrapper: an optional instance of inprocess.CallWrapper,
            it is automatically injected when in-process restart is in use
    """

    if inprocess_call_wrapper is not None:
        iteration = inprocess_call_wrapper.iteration
        store = torch.distributed.PrefixStore(str(iteration), store)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        args=train_args,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.double, device='cuda')
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0(
        'time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME)
    )
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
                LocalCheckpointManager,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import (
                parse_group_sequence,
                GroupWrapper,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
                CliqueReplicationStrategy,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvidia_resiliency_ext' module is required for local "
                "checkpointing but was not found. Please ensure it is installed."
            )

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump, args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            'local_checkpoint_manager': LocalCheckpointManager(
                args.non_persistent_local_ckpt_dir, repl_strategy=repl_strategy
            )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, 
        model_type, 
        no_wd_decay_cond=None,
        scale_lr_cond=vision_scale_lr_cond,
        lr_mult=args.vision_lr_mult,
        checkpointing_context=checkpointing_context
    )

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate ' 'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            iterators = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(
        args.train_iters,
        args.skip_train,
        args.do_train,
        args.do_valid,
        args.do_test,
        args.dataloader_type,
        args.retro_project_dir,
        args.retro_cyclic_train_iters,
    )

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)
    
    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
            )

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
            )

        one_logger and one_logger.log_metrics(
            {'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()}
        )

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    one_logger and one_logger.log_metrics(
        {'app_finish_time': one_logger_utils.get_timestamp_in_ms()}
    )

    ft_integration.shutdown()
    one_logger_utils.finish()

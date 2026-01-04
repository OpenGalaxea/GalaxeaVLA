import copy
import math
import os
import json
import shutil
import signal
import logging

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.distributed as dist
import tqdm

# TODO: fix bnb version
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset
from galaxea_fm.processors.base_processor import BaseProcessor
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.models.fdp.unet_policy import DiffusionUnetImagePolicy
from galaxea_fm.utils.get_scheduler import get_scheduler
from galaxea_fm.utils.logging_config import setup_logging
from galaxea_fm.utils.pytorch_utils import dict_apply, set_global_seed
from galaxea_fm.utils.train_utils import MFUTracker, init_experiment_tracker
from galaxea_fm.utils.edp import edp_uploader
from galaxea_fm.utils.edp.train_utils import override_cfg_with_edp, EDP_STORE_DATASET_ROOT
from galaxea_fm.utils.onnx_exporter import ONNXExporter
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

# Initialize Accelerator Logger
logger = get_logger(__name__)

require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        logger.info(msg)


def handle_resize(signum, frame):
    for instance in list(tqdm.tqdm._instances):
        if hasattr(instance, 'refresh'):
            # 
            instance.refresh()

signal.signal(signal.SIGWINCH, handle_resize)


def save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    batch_idx: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema_model: EMA,
):
    assert path.suffix == ".pt"
    path.parent.mkdir(exist_ok = True)
    state = {
        "step": step, 
        "epoch": epoch, 
        "batch_idx": batch_idx, 
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_model_state_dict": ema_model.ema_model.state_dict() if ema_model is not None else None,
    }
    torch.save(state, path)


def load_state_dict_safely(model, state_dict, extra_prefixes=None):
    """
    Safely load state dict with support for extra keys like normalizer parameters.
    
    Args:
        model: The model to load weights into
        state_dict: State dict from checkpoint
        extra_prefixes: List of key prefixes to force load even if not in model.state_dict()
                       Default: ['normalizer.']
    """
    if extra_prefixes is None:
        extra_prefixes = ['normalizer.']
    
    model_dict = model.state_dict()

    mismatched_keys = []
    missing_keys = []
    unexpected_keys = []
    extra_loaded_keys = []

    # First pass: load keys that exist in model_dict
    for key, param in model_dict.items():
        if key in state_dict:
            ckpt_param = state_dict[key]
            if param.shape == ckpt_param.shape:
                model_dict[key] = ckpt_param
            else:
                mismatched_keys.append((key, param.shape, ckpt_param.shape))
                logger.warning(f"Shape mismatch for {key}: model={param.shape}, checkpoint={ckpt_param.shape} - keeping random initialization")
        else:
            missing_keys.append(key)
            logger.warning(f"Missing key in checkpoint: {key} - keeping random initialization")

    # Second pass: handle extra keys (like normalizer parameters)
    for key in state_dict.keys():
        if key not in model_dict:
            # Check if this key should be force-loaded based on prefix
            should_load = any(key.startswith(prefix) for prefix in extra_prefixes)
            
            if should_load:
                model_dict[key] = state_dict[key]
                extra_loaded_keys.append(key)
                logger.info(f"Loading extra key: {key}")
            else:
                unexpected_keys.append(key)

    model.load_state_dict(model_dict, strict=False)

    # Log summary
    if missing_keys:
        logger.warning(f"Missing keys (keeping random init): {len(missing_keys)} parameters")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
    if mismatched_keys:
        logger.info(f"Shape mismatched keys (keeping random init): {len(mismatched_keys)} parameters")
    if extra_loaded_keys:
        logger.info(f"Extra keys loaded (e.g., normalizer): {len(extra_loaded_keys)} parameters")

    total_params = len(model_dict)
    loaded_params = total_params - len(missing_keys) - len(mismatched_keys)
    logger.info(f"Successfully loaded {loaded_params}/{total_params} parameters from checkpoint")

    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def finetune(cfg: DictConfig):
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    amp_dtype = torch.bfloat16 if cfg.model.enable_bf16_training else torch.float32
    mixed_precision = "bf16" if cfg.model.enable_bf16_training else "no"
    
    # Initialize Accelerator with mixed precision support
    project_config = ProjectConfiguration(project_dir=str(Path(cfg.output_dir)))
    # Initialize distributed context
    from datetime import timedelta
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_config=project_config,
        kwargs_handlers=[init_process_group_kwargs],
        log_with=cfg.logger.type,
    )
    torch.cuda.set_device(device_id := accelerator.local_process_index)
    torch.cuda.empty_cache()
    
    # Log AMP configuration for verification
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("AMP Configuration:")
        logger.info(f"  enable_bf16_training: {cfg.model.enable_bf16_training}")
        logger.info(f"  model_weights_to_bf16: {cfg.model.model_weights_to_bf16}")
        logger.info(f"  mixed_precision: {accelerator.mixed_precision}")
        logger.info(f"  amp_dtype: {amp_dtype}")
        logger.info(f"  native_amp: {accelerator.native_amp}")
        logger.info("=" * 60)

    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)

    is_use_edp = False
    if accelerator.is_main_process:
        is_use_edp = override_cfg_with_edp(cfg, output_dir)

    # Configure unified logging system (applies to all modules in the codebase)
    # Hydra's FileHandler (configured in hydra.yaml) will be preserved automatically
    setup_logging(
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    cfg_json = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_json, indent=2)
    logger.info(f"Output directory: {output_dir}")

    model: BasePolicy = instantiate(cfg.model.model_arch)
    # assert not cfg.resume_ckpt and cfg.pretrained_ckpt
    if cfg.resume_ckpt or cfg.model.pretrained_ckpt:
        checkpoint = cfg.resume_ckpt if cfg.resume_ckpt else cfg.model.pretrained_ckpt
        logger.info(f"Loading checkpoint from {checkpoint}")
        state_dict = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model = load_state_dict_safely(model, state_dict["model_state_dict"], extra_prefixes=["normalizer."])

    if cfg.model.model_weights_to_bf16:
        model = model.to(torch.bfloat16)

    use_ema = cfg.model.use_ema
    if use_ema:
        ema_model = EMA(
            model, 
            update_after_step=cfg.model.ema.update_after_step, 
            beta=cfg.model.ema.power,
        ).to(device_id) 

    if cfg.model.use_sync_bn and accelerator.num_processes > 1:
        logger.info("Use sync batch norm.")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.model.use_torch_compile:  # model being compiled in the first batch which takes some time
        # torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default")
    
    model = model.to(device_id)

    if accelerator.is_main_process:
        log_allocated_gpu_memory(stage="loading model", device=0)
        
    # Initialize experiment tracker
    tracker_type = init_experiment_tracker(cfg, accelerator, output_dir)

    train_dataset: BaseLerobotDataset = instantiate(cfg.data, is_training_set=True)
    eval_dataset: BaseLerobotDataset = instantiate(cfg.data, is_training_set=False)
    train_processor: BaseProcessor = instantiate(cfg.model.processor)
    eval_processor: BaseProcessor = instantiate(cfg.model.processor)
    if not (cfg.resume_ckpt or (cfg.model.pretrained_ckpt and cfg.model.use_pretrained_norm_stats)):
        logger.info("Calculating norm stats in the main process.")
        if accelerator.is_main_process:
            dataset_stats = train_dataset.get_dataset_stats(train_processor)
            save_dataset_stats_to_json(dataset_stats, output_dir / "dataset_stats.json")
        else:
            dataset_stats = None

        container = [dataset_stats]
        dist.broadcast_object_list(container, src=0)
        dataset_stats = container[0]
    else:
        checkpoint_path = Path(cfg.resume_ckpt if cfg.resume_ckpt else cfg.model.pretrained_ckpt)
        dataset_stats = load_dataset_stats_from_json(checkpoint_path.parent.parent / "dataset_stats.json")
        save_dataset_stats_to_json(dataset_stats, output_dir / "dataset_stats.json")

    train_processor.set_normalizer_from_stats(dataset_stats)
    eval_processor.set_normalizer_from_stats(dataset_stats)
    train_dataset.set_processor(train_processor)
    eval_dataset.set_processor(eval_processor)
    
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.model.batch_size, 
        sampler=train_sampler,
        shuffle=False,
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=worker_init_fn, 
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=cfg.batch_size_val, 
        sampler=eval_sampler, 
        shuffle=False, 
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=worker_init_fn, 
    )

    if cfg.model.max_epochs:
        assert not cfg.model.max_steps, "Cannot set both `max_epochs` and `max_steps`!"
        steps_per_epoch = len(train_dataloader) // cfg.model.grad_accumulation_steps
        max_steps = steps_per_epoch * cfg.model.max_epochs
    else:
        max_steps = cfg.model.max_steps
    
    # Determine whether MFU tracking is supported before wrapping with DDP
    skip_mfu_tracker = isinstance(model, DiffusionUnetImagePolicy)

    # Wrap model in PyTorch DDP Wrapper for Multi-GPU Training
    model = DDP(model, device_ids=[device_id], find_unused_parameters=cfg.model.find_unused_parameters, gradient_as_bucket_view=True)

    # Create Optimizer
    param_groups = model.module.get_optim_param_groups(cfg.model.learning_rate, cfg.model.weight_decay)
    # Convert OmegaConf objects to plain Python objects to avoid serialization issues
    betas = tuple(cfg.model.betas)
    if cfg.model.use_8bit_optimizer:
        assert bnb is not None, "bitsandbytes is not installed, cannot use 8bit optimizer"
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=betas)
    else:
        optimizer = AdamW(param_groups, betas=betas)
    
    if cfg.model.lr_scheduler_type == "OneCycleLR":
        # from galaxea_dp lr scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=cfg.model.learning_rate,
            total_steps=max_steps,
            pct_start=cfg.model.pct_start,
            anneal_strategy=cfg.model.anneal_strategy,
            div_factor=cfg.model.div_factor,
            final_div_factor=cfg.model.final_div_factor,
        )
    else:
        scheduler = get_scheduler(
            name=cfg.model.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.model.warmup_steps,
            num_training_steps=max_steps,
        )

    # Resume Training
    if cfg.resume_ckpt:
        resume_dataloader = True
        checkpoint = torch.load(cfg.resume_ckpt, weights_only=True, map_location=lambda storage, loc: storage.cuda(device_id))
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        batch_idx = checkpoint["batch_idx"]
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if use_ema:
            try:
                ema_model.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            except KeyError:
                logger.warning("EMA model not found in checkpoint, skipping EMA update")
        del checkpoint  # Clean up checkpoint to avoid OOM
        torch.cuda.empty_cache()
        logger.info(f"Resuming training from step {step}")
    else:
        resume_dataloader = False
        step = 0
        epoch = 0
        batch_idx = 0
    
    # Initialize MFU Tracker
    mfu_tracker = None
    if accelerator.is_main_process:
        effective_batch_size = cfg.model.batch_size * cfg.model.grad_accumulation_steps * dist.get_world_size()
        if skip_mfu_tracker:
            logger.info("Skipping MFU tracker for DiffusionUnetImagePolicy (FDP) since MFU estimation is not supported.")
        else:
            mfu_tracker = MFUTracker(
                model=model.module,
                batch_size=effective_batch_size,
                device_id=device_id,
                update_interval=cfg.logger.log_steps,
                world_size=dist.get_world_size(),
                dtype=amp_dtype,  # Pass the training dtype
            )
            mfu_tracker.reset(step)
    
    accelerator.wait_for_everyone()
    # Train!
    training_done = False
    with tqdm.tqdm(initial=step, total=max_steps, leave=False, dynamic_ncols=True) as progress:
        while not training_done:
            train_sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            if resume_dataloader:
                logger.info(f"Resume dataloader state from batch_idx {batch_idx}")
                for _ in range(batch_idx):
                    next(data_iter)
                resume_dataloader = False
            else:
                batch_idx = 0

            model.train()
            optimizer.zero_grad()
            while batch_idx < len(train_dataloader):
                batch = next(data_iter)
                # Turn off sync when is not optimizer step
                is_optimizer_step = (batch_idx + 1) % cfg.model.grad_accumulation_steps == 0
                sync_ctx = model.no_sync() if not is_optimizer_step else nullcontext()
                with sync_ctx:
                    with accelerator.autocast():
                        # AMP Best Practice: Keep input in FP32, let autocast handle the conversion
                        # No manual dtype conversion needed here - autocast will automatically
                        # cast operations to the appropriate precision
                        loss, loss_value_dict = model(batch)
                    # Normalize loss to account for gradient accumulation
                    normalized_loss = loss / cfg.model.grad_accumulation_steps
                    normalized_loss.backward()
                    
                batch_idx += 1

                if is_optimizer_step:
                    # TODO : rename it into grad clip norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    progress.set_description(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                    progress.update()
                    progress.refresh()

                    if use_ema:
                        ema_model.update()
                    
                    step += 1

                    # Log metrics on optimizer steps
                    if step % cfg.logger.log_steps == 0 and tracker_type != "none":
                        # Ensure values are plain Python numbers
                        log_dict = {k: (v.item() if hasattr(v, "item") else float(v)) for k, v in loss_value_dict.items()}
                        log_dict.update({
                            "lr/encoder": optimizer.param_groups[0]["lr"],
                            "lr/model": optimizer.param_groups[1]["lr"],
                            "grad_norm": grad_norm.item(),
                        })
                        
                        # Add MFU metrics if tracker is available
                        if mfu_tracker is not None:
                            mfu_metrics = mfu_tracker.compute_metrics(step)
                            log_dict.update(mfu_metrics)
                        
                        accelerator.log(log_dict, step=step)

                # Save checkpoint in the main process
                if step > 0 and (step % cfg.checkpointing_steps) == 0:
                    if accelerator.is_main_process:
                        logger.info(f"Saving model checkpoint for step {step} ...")
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            path=output_dir / "checkpoints" / f"step_{step}.pt", 
                            step=step, 
                            epoch=epoch, 
                            batch_idx=batch_idx, 
                            model=unwrapped_model, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            ema_model=ema_model if use_ema else None,
                        )

                    # Block on main process checkpointing
                    accelerator.wait_for_everyone()

                # Stop training when max_steps is reached
                if step >= max_steps:
                    logger.info(f"Max step {max_steps} reached, stop training ...")
                    training_done = True
                    break

            epoch += 1
        
    if accelerator.is_main_process:
        logger.info(f"Saving model checkpoint for step {step} ...")
        checkpoint_path = output_dir / "checkpoints" / f"step_{step}.pt"
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint(
            path=checkpoint_path,
            step=step, 
            epoch=epoch, 
            batch_idx=batch_idx, 
            model=unwrapped_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            ema_model=ema_model if use_ema else None
        )
        
        last_pt_path = output_dir / "last.pt"
        if last_pt_path.exists():
            last_pt_path.unlink()
        last_pt_path.symlink_to(checkpoint_path.name)

        # Removed dataset tar files after training
        if is_use_edp:
            shutil.rmtree(EDP_STORE_DATASET_ROOT)
            logger.info(f"Removed dataset tar files from {EDP_STORE_DATASET_ROOT}")

    edp = edp_uploader.EDPCardCreator(cfg['edp'])
    if accelerator.is_main_process and edp.has_card:
        try:
            logger.info(f"Uploading EDP card: {edp.has_card}")
            edp.upload()

            logger.info("Exporting ONNX model...")
            onnx_exporter = ONNXExporter(cfg=cfg)
            onnx_exporter.export_onnx()
            logger.info("ONNX model exported successfully")
        except Exception as e:
            logger.error(f"EDPONNX: {e}")
            import traceback; traceback.print_exc()
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    finetune()

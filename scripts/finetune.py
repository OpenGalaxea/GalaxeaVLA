import copy
import math
import os
import json
import shutil
import signal
import logging
import time

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.distributed as dist
import tqdm
from functools import partial

# TODO: fix bnb version
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from accelerate import Accelerator, InitProcessGroupKwargs
from g05.utils.logging.logging_config import get_logger
from accelerate.utils import ProjectConfiguration
from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

from g05.data.base_lerobot_dataset import BaseLerobotDataset
from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.models.base_policy import BasePolicy
from g05.utils.training.get_scheduler import get_scheduler
from g05.utils.logging.logging_config import setup_logging
from g05.utils.common.pytorch_utils import set_global_seed
from g05.utils.common.dist import (
    DistributedSampler,
    ResumableDistributedSampler,
)
from g05.utils.training.train_utils import (
    MFUTracker,
    init_experiment_tracker,
    eval_tokenizer_first_batch,
)
from g05.utils.training.train_utils import set_global_monitor, get_global_monitor
from g05.utils.data.data_utils import collate_fn_pad_sequences
from g05.utils.data.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json
from g05.utils.data.data_utils import to_json_serializable
from g05.utils.logging.banner import print_banner
from g05.utils.logging.log_box import log_box
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.checkpoint.ckpt_utils import copy_hf_processor_files
from g05.utils.checkpoint.checkpoint_utils import (
    fix_optimizer_state_after_resume,
    save_training_checkpoint,
)

from utils.metric import resolve_parts_meta
from utils.preflight import build_filter_str, run_preflight_checks
from utils.eval_snapshot import save_train_snapshot
from utils.train_eval import PeriodicEvaluator

register_default_resolvers()

# Flag to enable first batch tokenizer evaluation for debugging
EVAL_TOKENIZER_FIRST_BATCH = True

# Initialize Accelerator Logger
logger = get_logger(__name__)

require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_data_config_path(config_path: str) -> Path:
    """Resolve dataset/mixture config path from CLI input."""
    raw_path = Path(config_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        project_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            [
                project_root / raw_path,
                project_root / "configs" / "data" / raw_path,
                project_root / "configs" / "data" / f"{config_path}.yaml",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Dataset config not found: {config_path} (tried: {[str(p) for p in candidates]})"
    )


def _load_dataset_override_config(dataset_path: str) -> dict:
    """Load dataset config and extract mixture-compatible fields."""
    dataset_yaml_path = _resolve_data_config_path(dataset_path)
    raw = OmegaConf.to_container(OmegaConf.load(dataset_yaml_path), resolve=False)

    if not isinstance(raw, dict):
        raise ValueError(f"Dataset config must be a dict: {dataset_yaml_path}")

    if raw.get("_target_") == "g05.data.mixture_lerobot_dataset.MixtureLerobotDataset":
        return {
            "embodiment_datasets": raw.get("embodiment_datasets", {}),
            "processors": raw.get("processors", {}),
        }

    reserved_keys = {"_target_", "processor", "processors"}
    dataset_names = [key for key in raw.keys() if key not in reserved_keys]
    if not dataset_names:
        raise ValueError(f"Could not find dataset config in {dataset_yaml_path}")

    processor_config = raw.get("processor")
    processors_config = raw.get("processors") if isinstance(raw.get("processors"), dict) else None
    embodiment_datasets = {dataset_name: raw[dataset_name] for dataset_name in dataset_names}

    if processor_config is not None:
        processors = {dataset_name: processor_config for dataset_name in dataset_names}
    elif processors_config is not None:
        processors = {
            dataset_name: processors_config.get(dataset_name) for dataset_name in dataset_names
        }
    else:
        processors = {}

    return {
        "embodiment_datasets": embodiment_datasets,
        "processors": processors,
    }


def _apply_dataset_override_if_needed(cfg: DictConfig) -> None:
    """Apply single-dataset override while preserving mixture-level parameters."""
    dataset_path = os.environ.get("OVERRIDE_DATASET")
    if not dataset_path:
        return

    dataset_config = _load_dataset_override_config(dataset_path)
    data_resolved = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_resolved, dict):
        raise ValueError("cfg.data must resolve to a mapping before applying dataset override")

    data_resolved["embodiment_datasets"] = dataset_config.get("embodiment_datasets", {})
    data_resolved["processors"] = dataset_config.get("processors", {})

    OmegaConf.set_struct(cfg, False)
    cfg.data = OmegaConf.create(data_resolved)
    OmegaConf.set_struct(cfg, True)

    logger.info(f"[Data Override] Using dataset: {dataset_path}")
    logger.info(
        "[Data Override] embodiment_datasets: "
        f"{list(dataset_config.get('embodiment_datasets', {}).keys())}"
    )
    logger.info(f"[Data Override] processors: {list(dataset_config.get('processors', {}).keys())}")


def unwrap_model(model):
    """Unwrap model from the DDP wrapper to access the underlying module."""
    if hasattr(model, "module"):
        return model.module  # DDP wrapper
    return model


def _vlm_chat_verification(model, device) -> None:
    """Send a text prompt to the VLM and log its autoregressive response.

    This verifies that VLM weights were loaded correctly — if the weights
    are intact, the model should produce coherent text.  Only runs on the
    main process and is wrapped in a try/except so it never blocks training.
    """
    # --- Locate VLM and text tokenizer ---
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        vlm = getattr(getattr(model, "model", None), "vlm", None)
    if vlm is None:
        logger.info("[VLM Chat] Skipped — no VLM found on model")
        return

    if not hasattr(vlm, "generate") or not callable(getattr(vlm, "generate", None)):
        log_box(
            logger,
            "🗣  VLM Weight Verification (Chat)",
            [
                ("Status", "Skipped"),
                None,
                (
                    "Reason",
                    f"VLM module ({type(vlm).__name__}) is a custom nn.Module "
                    "without a generate() method — cannot perform AR text "
                    "generation. Weight integrity is verified by the training "
                    "loss instead.",
                ),
            ],
            inner_width=80,
        )
        return

    tokenizer = None
    if hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
        tokenizer = model.processor.tokenizer
    elif hasattr(model, "action_tokenizer") and hasattr(model.action_tokenizer, "tokenizer"):
        tokenizer = model.action_tokenizer.tokenizer
    if tokenizer is None:
        logger.info("[VLM Chat] Skipped — no text tokenizer found")
        return

    prompt = "Are you ready for training!"
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = vlm.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                do_sample=False,
            )
        new_tokens = output_ids[0, input_ids.shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        log_box(
            logger,
            "🗣  VLM Weight Verification (Chat)",
            [
                ("Prompt", prompt),
                None,
                ("Response", response[:200] if response else "(empty)"),
            ],
            inner_width=80,
        )
    except Exception as e:
        log_box(
            logger,
            "🗣  VLM Weight Verification (Chat)",
            [
                ("Prompt", prompt),
                None,
                ("Error", f"{type(e).__name__}: {e}"),
            ],
            inner_width=80,
        )


def handle_resize(signum, frame):
    for instance in list(tqdm.tqdm._instances):
        if hasattr(instance, "refresh"):
            #
            instance.refresh()


signal.signal(signal.SIGWINCH, handle_resize)


def _fmt_n(n: int) -> str:
    """Format a sample count as compact K/M string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _build_data_pipeline_summary(
    train_dataset,
    eval_dataset,
    train_processor,
    train_dataloader,
    dl_num_workers: int,
    dl_prefetch_factor: int,
    batch_size: int,
    batch_size_val: int,
    grad_accumulation_steps: int,
    norm_stats_source: str = "unknown",
) -> list:
    """Build rows for the Data Pipeline Summary log_box."""
    from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset

    rows = []

    # --- Top-level summary ---
    if isinstance(train_dataset, MixtureLerobotDataset):
        # Aggregate per embodiment (multiple groups per emb are summed)
        emb_info: dict = {}
        for i, (emb, ds) in enumerate(zip(train_dataset.embodiments, train_dataset.datasets)):
            if emb not in emb_info:
                emb_info[emb] = {"dirs": 0, "samples": 0, "weight": 0.0}
            emb_info[emb]["dirs"] += len(ds.dataset_dirs)
            emb_info[emb]["samples"] += train_dataset.actual_lengths[i]
            emb_info[emb]["weight"] += train_dataset.weights[i]

        total_embs = len(emb_info)
        total_dirs = sum(v["dirs"] for v in emb_info.values())
        total_samples = sum(v["samples"] for v in emb_info.values())
        rows.append(
            f"{total_embs} embodiments · {total_dirs} dirs · {total_samples:,} train samples"
        )
    else:
        rows.append(f"train samples: {len(train_dataset):,}")
        emb_info = {}

    rows.append(("norm_stats", norm_stats_source))

    # --- DataLoader params ---
    action_steps = len(train_dataloader)
    rows.append(
        (
            "train",
            f"{len(train_dataset):,} samples  bs={batch_size}  "
            f"workers={dl_num_workers}  prefetch={dl_prefetch_factor}  "
            f"steps/epoch={action_steps}",
        )
    )
    rows.append(
        (
            "eval",
            f"{len(eval_dataset):,} samples  bs={batch_size_val}",
        )
    )

    # --- Per-embodiment table ---
    if emb_info:
        rows.append(None)  # separator
        header = f"  {'embodiment':<20}{'dirs':>4}  {'samples':>7}  {'wt':>5}  {'norm':<10}  {'merger':<10}  filter"
        rows.append(header)
        rows.append("  " + "-" * 68)
        for emb, info in emb_info.items():
            p = (
                train_processor.processors.get(emb)
                if hasattr(train_processor, "processors")
                else None
            )
            norm_mode = str(getattr(p, "norm_default_mode", "?")) if p else "?"
            merger_cls = type(getattr(p, "action_state_merger", None)).__name__ if p else "?"
            # Shorten verbose class names: PaddingActionStateMerger → padding
            merger = (
                merger_cls.replace("ActionStateMerger", "")
                .replace("ActionMerger", "")
                .replace("Merger", "")
                .lower()
                or merger_cls
            )
            filter_str = build_filter_str(getattr(p, "action_filter", None)) if p else "-"
            emb_short = emb[:20]
            row = (
                f"  {emb_short:<20}{info['dirs']:>4}  {_fmt_n(info['samples']):>7}  "
                f"{info['weight']:>5.2f}  {norm_mode:<10}  {merger:<10}  {filter_str}"
            )
            rows.append(row)

    # --- Per-embodiment image sizes (resolved) ---
    if hasattr(train_processor, "processors"):
        rows.append(None)  # separator
        rows.append("  [Image Sizes — after _resolve_image_shapes]")
        img_header = f"  {'embodiment':<20}{'camera_key':<25}{'H×W':<12}{'camera_type'}"
        rows.append(img_header)
        rows.append("  " + "-" * 70)
        for emb, p in train_processor.processors.items():
            shape_meta = getattr(p, "shape_meta", None)
            if shape_meta is None:
                continue
            images = shape_meta.get("images", [])
            if not images:
                rows.append(f"  {emb[:20]:<20}(no images)")
                continue
            for img_meta in images:
                key = img_meta.get("key", "?")
                shape = img_meta.get("shape", None)
                cam_type = img_meta.get("camera_type", "?")
                if shape and len(shape) == 3:
                    hw_str = f"{shape[1]}×{shape[2]}"
                else:
                    hw_str = str(shape)
                rows.append(f"  {emb[:20]:<20}{key:<25}{hw_str:<12}{cam_type}")

    return rows


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

    # Configure unified logging system early so ALL subsequent logs (including
    # truncate_embodiments / truncate_datasets boxes) are colored by RichHandler.
    # Hydra's FileHandler (configured in hydra.yaml) is preserved automatically.
    setup_logging(
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    if accelerator.is_main_process:
        print_banner(subtitle=f"Post-Training · {accelerator.num_processes}× GPU")

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

    _apply_dataset_override_if_needed(cfg)
    OmegaConf.resolve(cfg)

    from g05.utils.config.config_validator import validate_train_config

    validate_train_config(cfg)

    # --test mode: truncate embodiments, dataset_dirs, and VLM datasets for fast validation
    # (must be after resolve so ${oc.load:...} interpolations are expanded)
    max_embodiments = os.environ.get("MAX_EMBODIMENTS")
    if max_embodiments is not None:
        from g05.utils.eval.eval_utils import truncate_embodiments

        truncate_embodiments(cfg, int(max_embodiments))
    max_datasets = os.environ.get("MAX_DATASETS")
    if max_datasets is not None:
        from g05.utils.eval.eval_utils import truncate_datasets

        truncate_datasets(cfg, int(max_datasets))
    # --dry-run: override to exactly 1 training step, no local file saves
    _dry_run = os.environ.get("DRY_RUN", "0") == "1"
    if _dry_run:
        logger.info(
            "[Dry Run] Training 1 step + 1 eval step, only log files will be saved (no checkpoints/tokenizer)"
        )
    output_dir = Path(cfg.output_dir)
    # Two-channel file logging strategy (main process only):
    #   1. log_box output  → written directly by log_box.py via set_log_file,
    #      using inner_width=200 (no ellipsis truncation).
    #   2. Regular logger.info/warning messages → FileHandler on root logger,
    #      filtered to skip box lines (║/╔) so there's no duplicate/truncated copy.
    # Note: In dry-run mode, we still save log files so users can inspect the detailed
    #       log boxes and configuration without running a full training.
    if accelerator.is_main_process:
        from g05.utils.logging.log_box import set_log_file

        _log_file = output_dir / "logs" / "pretrain_log.txt"
        set_log_file(_log_file)

        class _SkipBoxFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                return "║" not in msg and "╔" not in msg

        _fh = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _fh.addFilter(_SkipBoxFilter())
        logging.getLogger().addHandler(_fh)

    _tokenizer_src_path = (
        cfg.model.tokenizer.vq_config.get("ckpt_dir", "N/A")
        if cfg.model.get("tokenizer", None) and cfg.model.tokenizer.get("vq_config", None)
        else "N/A"
    )

    # we copy the tokenizer checkpoint to output_dir making the inference easier.
    if (
        not _dry_run
        and cfg.model.get("tokenizer", None)
        and cfg.model.tokenizer.get("vq_config", None)
    ):
        if cfg.model.tokenizer.vq_config.get("ckpt_dir", None):
            src_ckpt_path = Path(cfg.model.tokenizer.vq_config.ckpt_dir)
            dst_ckpt_path = output_dir / "action_tokenizer.pt"
            if accelerator.is_main_process:
                dst_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_ckpt_path, dst_ckpt_path)
                logger.info(f"Copied tokenizer checkpoint from {src_ckpt_path} to {dst_ckpt_path}")
            accelerator.wait_for_everyone()
            cfg.model.tokenizer.vq_config.ckpt_dir = str(dst_ckpt_path)
            logger.info(
                f"Updated tokenizer checkpoint path to {cfg.model.tokenizer.vq_config.ckpt_dir}"
            )

    # Copy HF processor/tokenizer config files to output_dir/hf_processor/ so eval/serve
    # can run without access to the original pretrained_model_path.
    if not _dry_run and accelerator.is_main_process:
        _pretrained_path = cfg.model.model_arch.get("pretrained_model_path", None)
        if _pretrained_path:
            copy_hf_processor_files(_pretrained_path, output_dir / "hf_processor")
    accelerator.wait_for_everyone()

    cfg_json = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_json, indent=2)
    logger.info(f"Output directory: {output_dir}")

    # ── Box 1: Run Configuration ─────────────────────────────────────
    if accelerator.is_main_process:
        _tokenizer_path = (
            cfg.model.tokenizer.vq_config.get("ckpt_dir", "N/A")
            if cfg.model.get("tokenizer", None) and cfg.model.tokenizer.get("vq_config", None)
            else "N/A"
        )
        _ckpt_path_str = cfg.resume_ckpt or cfg.model.get("pretrained_ckpt", "N/A") or "N/A"
        log_box(
            logger,
            "⚙  Run Configuration",
            [
                ("Output dir", str(output_dir)),
                ("VLA checkpoint", _ckpt_path_str),
                ("Tokenizer src", _tokenizer_src_path),
                ("Tokenizer dst", _tokenizer_path),
                None,
                ("World size", f"{accelerator.num_processes}"),
                ("AMP dtype", f"{amp_dtype}  (native={accelerator.native_amp})"),
                (
                    "BF16 training",
                    f"enable={cfg.model.enable_bf16_training}   weights_cast={cfg.model.model_weights_to_bf16}   mixed_precision={accelerator.mixed_precision}",
                ),
            ],
        )

    # assert not cfg.resume_ckpt and cfg.pretrained_ckpt
    checkpoint = None
    if cfg.resume_ckpt or cfg.model.pretrained_ckpt:
        ckpt_path = cfg.resume_ckpt if cfg.resume_ckpt else cfg.model.pretrained_ckpt
        logger.info(f"Loading checkpoint from {ckpt_path}")
        from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint

        model, checkpoint = load_model_from_checkpoint(
            cfg.model.model_arch,
            ckpt_path,
            extra_prefixes=["normalizer."],
            eval_mode=False,
            return_full_checkpoint=True,
        )
    else:
        model: BasePolicy = instantiate(cfg.model.model_arch)
    if cfg.model.get("force_reinit_extra_token_embedding", False):
        at = model.action_tokenizer
        if at.use_extra_tokens:
            model.model.resize_embedding(
                new_vocab_size=len(at.tokenizer),
                base_vocab_size=model._base_vocab_size,
                pad_token_id=model.model_config.pad_token_id,
                force=True,
            )

    if cfg.model.model_weights_to_bf16:
        model = model.to(torch.bfloat16)

    # Force critical layers to float32 (patterns defined per-model in fp32_param_patterns)
    model.apply_fp32_params()

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

    if cfg.model.use_torch_compile:
        model = torch.compile(model, mode="default")

    model = model.to(device_id)
    if hasattr(model, "action_tokenizer"):
        model.action_tokenizer.to(device_id)

    with accelerator.main_process_first():
        logger.info(f"[Process {accelerator.process_index}] Loading dataset...")
        train_dataset = instantiate_dataset(cfg, is_training_set=True)
        eval_dataset = instantiate_dataset(cfg, is_training_set=False)

    # Emit a single summary for all eval datasets that were reused from the train cache.
    from g05.data.base_lerobot_dataset import BaseLerobotDataset

    if accelerator.is_main_process and BaseLerobotDataset._cache_hit_count > 0:
        logger.info(
            f"MultiLeRobotDataset: {BaseLerobotDataset._cache_hit_count} dataset(s) "
            f"reused from train cache (eval avoided duplicate loading)"
        )
        BaseLerobotDataset._cache_hit_count = 0

    # NOTE: wandb/tracker init moved to just before training loop (see below).

    train_processor = build_processors(cfg)
    eval_processor = build_processors(cfg)
    eval_processor.eval()
    eval_parts_meta = resolve_parts_meta(processor=eval_processor)

    def _compute_partial_stats(
        train_dataset_obj,
        train_processor_obj,
        missing_keys: set,
    ) -> Dict[str, Any]:
        from collections import defaultdict as DefaultDict

        stats_by_type = DefaultDict(list)
        weights_by_type = DefaultDict(list)

        for emb, w, ds in zip(
            train_dataset_obj.embodiments,
            train_dataset_obj.weights,
            train_dataset_obj.datasets,
        ):
            emb_type = train_dataset_obj.embodiments2types[emb]
            if emb_type not in missing_keys:
                continue
            logger.info(f"Computing stats for missing embodiment_type: {emb_type}")
            stats = ds.get_dataset_stats(train_processor_obj[emb_type])
            stats_by_type[emb_type].append(stats)
            weights_by_type[emb_type].append(w)

        aggregated_stats_by_type = {}
        for emb_type in stats_by_type:
            aggregated_stats_by_type[emb_type] = MixtureLerobotDataset._aggregate_weighted_stats(
                weights_by_type[emb_type], stats_by_type[emb_type]
            )

        return aggregated_stats_by_type

    norm_stats_source = "unknown"
    stats_path = Path(cfg.datastatics_path) if cfg.get("datastatics_path", None) else None

    dataset_stats = None

    if stats_path is not None:
        # All ranks follow the exact same control flow to avoid collective desync.
        # train_dataset.all_embodiment_types is rank-local due to directory-level
        # bin-packing, so each rank must compute its own missing stats and then
        # all_gather them into a global union.
        loaded_stats: Dict[str, Any] = {}
        if stats_path.exists() and stats_path.stat().st_size > 0:
            try:
                loaded_stats = load_dataset_stats_from_json(stats_path)
            except (json.JSONDecodeError, OSError) as e:
                if accelerator.is_main_process:
                    logger.warning(f"Stats file {stats_path} is corrupted ({e}), will recompute...")
                loaded_stats = {}
        elif accelerator.is_main_process:
            logger.info(f"Stats file {stats_path} not found or empty, will compute...")

        needed_keys = set(train_dataset.all_embodiment_types)
        missing_keys = needed_keys - set(loaded_stats.keys())

        if missing_keys:
            logger.info(
                f"[rank {accelerator.process_index}] computing {len(missing_keys)} "
                f"missing embodiment_type stats: {sorted(missing_keys)}"
            )
            local_computed = _compute_partial_stats(train_dataset, train_processor, missing_keys)
        else:
            local_computed = {}

        # All ranks participate in all_gather unconditionally, merging locally
        # computed stats into a global union.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
            dist.all_gather_object(gathered, local_computed)
        else:
            gathered = [local_computed]

        for g in gathered:
            if g:
                loaded_stats.update(g)

        # loaded_stats is now the global union; filter it to this rank's needed subset.
        try:
            dataset_stats = {k: loaded_stats[k] for k in needed_keys}
        except KeyError as e:
            raise RuntimeError(
                f"[rank {accelerator.process_index}] embodiment_type {e} stats missing "
                f"even after all-rank compute. Loaded keys: {sorted(loaded_stats.keys())}"
            ) from e

        any_missing_globally = any(g for g in gathered)
        if any_missing_globally:
            norm_stats_source = f"pre-computed + per-rank compute ({stats_path})"
        else:
            norm_stats_source = f"pre-computed ({stats_path})"

        # Only the main process writes the global union to disk.
        if accelerator.is_main_process:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(to_json_serializable(loaded_stats), f, indent=2)
            save_dataset_stats_to_json(loaded_stats, output_dir / "dataset_stats.json")
            logger.info(
                f"Stats union persisted to {stats_path} ({len(loaded_stats)} embodiment_types)"
            )

    elif not (
        cfg.resume_ckpt or (cfg.model.pretrained_ckpt and cfg.model.use_pretrained_norm_stats)
    ):
        norm_stats_source = "fresh (computed from training data)"
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
        src_stats_path = checkpoint_path.parent.parent / "dataset_stats.json"
        norm_stats_source = f"checkpoint ({src_stats_path})"
        dataset_stats = load_dataset_stats_from_json(src_stats_path)
        if accelerator.is_main_process:
            dst_stats_path = output_dir / "dataset_stats.json"
            if dst_stats_path.resolve() != src_stats_path.resolve():
                save_dataset_stats_to_json(dataset_stats, dst_stats_path)
        accelerator.wait_for_everyone()

    train_processor.set_normalizer_from_stats(dataset_stats)
    eval_processor.set_normalizer_from_stats(dataset_stats)
    train_dataset.set_processor(train_processor)
    eval_dataset.set_processor(eval_processor)

    overfit_batch = cfg.get("overfit_batch", None)
    if overfit_batch is not None:
        overfit_mode = cfg.get("overfit_mode", "per_dataset")
        overfit_samples = int(overfit_batch) * int(cfg.model.batch_size)
        if isinstance(train_dataset, MixtureLerobotDataset):
            train_dataset.enable_overfit(overfit_samples, mode=overfit_mode)
        else:
            train_dataset.enable_overfit(overfit_samples)
        train_processor.eval()
        logger.info(
            f"Overfit mode ({overfit_mode}): requested {overfit_samples} samples "
            f"({overfit_batch} batches x batch_size={cfg.model.batch_size}), "
            f"selected {len(train_dataset)} stable samples, "
            f"processor set to eval mode"
        )

    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    sampler_num_replicas = accelerator.num_processes
    sampler_rank = accelerator.process_index
    overfit_shuffle = overfit_batch is None
    train_sampler = ResumableDistributedSampler(
        train_dataset,
        num_replicas=sampler_num_replicas,
        rank=sampler_rank,
        seed=cfg.seed,
        shuffle=overfit_shuffle,
        batch_size=cfg.model.batch_size,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=sampler_num_replicas,
        rank=sampler_rank,
        seed=cfg.seed,
        shuffle=False,
    )  # no need to resume eval sampler

    # TODO: use keyword to select collate_fn
    if cfg.model.get("collate_fn"):
        action_collate_fn = partial(
            collate_fn_pad_sequences, padding_input_id=train_processor.pad_token_id
        )
        logger.info(f"Using collate_fn: {action_collate_fn}")
    else:
        action_collate_fn = None

    dl_num_workers = cfg.model.num_workers
    dl_prefetch_factor = cfg.model.get("prefetch_factor", 2)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=cfg.model.pin_memory,
        persistent_workers=cfg.model.persistent_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=action_collate_fn,
        prefetch_factor=dl_prefetch_factor,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size_val,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=cfg.model.pin_memory,
        persistent_workers=cfg.model.persistent_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=action_collate_fn,
        prefetch_factor=dl_prefetch_factor,
    )

    evaluator = PeriodicEvaluator(
        eval_dataloader=eval_dataloader,
        eval_sampler=eval_sampler,
        eval_processor=eval_processor,
        parts_meta=eval_parts_meta,
        output_dir=output_dir,
    )

    if overfit_batch is not None:
        logger.info(
            "Overfit mode: periodic eval will reuse the current overfit training batch "
            "instead of sampling from the validation split."
        )

    log_box(
        logger,
        "📊  Data Pipeline Summary",
        _build_data_pipeline_summary(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_processor=train_processor,
            train_dataloader=train_dataloader,
            dl_num_workers=dl_num_workers,
            dl_prefetch_factor=dl_prefetch_factor,
            batch_size=cfg.model.batch_size,
            batch_size_val=cfg.batch_size_val,
            grad_accumulation_steps=cfg.model.grad_accumulation_steps,
            norm_stats_source=norm_stats_source,
        ),
        inner_width=100,
    )
    # Clear dataset cache — DataLoader workers have their own copies
    from g05.data.base_lerobot_dataset import BaseLerobotDataset

    BaseLerobotDataset.clear_cache()

    if _dry_run:
        max_steps = 1
        cfg.eval_steps = 1
        cfg.logger.log_steps = 1
    elif cfg.model.max_epochs:
        assert not cfg.model.max_steps, "Cannot set both `max_epochs` and `max_steps`!"
        action_steps_per_epoch = len(train_dataloader) // cfg.model.grad_accumulation_steps
        max_steps = int(math.ceil(action_steps_per_epoch * cfg.model.max_epochs))
    else:
        max_steps = cfg.model.max_steps

    # ---- DDP wrapping ----
    logger.info("Wrapping model with DDP...")
    model = DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=cfg.model.find_unused_parameters,
        gradient_as_bucket_view=True,
        bucket_cap_mb=128,
    )

    # Create Optimizer
    logger.info("Creating optimizer and LR scheduler...")
    betas = tuple(cfg.model.betas)
    _optim_model = model.module
    if hasattr(_optim_model, "get_optim_param_groups"):
        trainable_params = _optim_model.get_optim_param_groups(
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            apply_decay_on_norm_and_bias=cfg.model.get("apply_decay_on_norm_and_bias", False),
            backbone_lr_multiplier=cfg.model.get("backbone_lr_multiplier", 1.0),
            vision_lr_multiplier=cfg.model.get("vision_lr_multiplier", 1.0),
        )
    else:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
    if cfg.model.use_8bit_optimizer:
        assert bnb is not None, "bitsandbytes is not installed, cannot use 8bit optimizer"
        if cfg.get("use_fused_optimizer", False):
            raise ValueError(
                "use_fused_optimizer is incompatible with 8bit optimizer, ignoring fused option"
            )
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=cfg.model.learning_rate,
            betas=betas,
            weight_decay=cfg.model.weight_decay,
        )
    else:
        use_fused = cfg.get("use_fused_optimizer", False)
        optimizer = AdamW(
            trainable_params,
            lr=cfg.model.learning_rate,
            betas=betas,
            weight_decay=cfg.model.weight_decay,
            fused=use_fused,
        )
        if use_fused:
            pass  # optimizer info shown in Training Configuration box

    # Log per-group learning rates (vision / backbone / action)
    if (
        isinstance(trainable_params, list)
        and trainable_params
        and isinstance(trainable_params[0], dict)
    ):
        for g in trainable_params:
            logger.info(
                f"  param_group '{g.get('name', '?')}': lr={g['lr']:.2e}, #params={len(g['params'])}, wd={g.get('weight_decay', 0)}"
            )

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
        warmup_ratio = cfg.model.get("warmup_ratio", None)
        if warmup_ratio is not None:
            warmup_steps = int(max_steps * warmup_ratio)
            pass  # LR schedule info shown in Training Configuration box
        else:
            warmup_steps = cfg.model.warmup_steps
        scheduler = get_scheduler(
            name=cfg.model.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_min_ratio=cfg.model.get("lr_min_ratio", 0.0),
            constant_end_ratio=cfg.model.get("constant_end_ratio", 0.5),
        )

    # Resume Training
    if cfg.resume_ckpt:
        resume_dataloader = True
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        batch_idx = checkpoint["batch_idx"]
        action_batch_idx = checkpoint.get("action_batch_idx", 0)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _reset_count = fix_optimizer_state_after_resume(optimizer)
        if _reset_count > 0:
            logger.warning(
                f"Reset optimizer state for {_reset_count} params due to shape mismatch "
                f"between checkpoint and current model (e.g., vocab size changed)."
            )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if use_ema:
            try:
                ema_model.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            except KeyError:
                logger.warning("EMA model not found in checkpoint, skipping EMA update")
        del checkpoint  # Clean up checkpoint to avoid OOM
        torch.cuda.empty_cache()
        logger.info(f"Resuming training from step {step}")
    else:
        resume_dataloader = False
        step = 0
        epoch = 0
        action_batch_idx = 0
        batch_idx = 0

    # Initialize MFU Tracker
    mfu_tracker = None
    if accelerator.is_main_process:
        effective_batch_size = (
            cfg.model.batch_size * cfg.model.grad_accumulation_steps * dist.get_world_size()
        )
        mfu_tracker = MFUTracker(
            model=model.module,
            batch_size=effective_batch_size,
            device_id=device_id,
            update_interval=cfg.logger.log_steps,
            world_size=dist.get_world_size(),
            dtype=amp_dtype,  # Pass the training dtype
        )
        mfu_tracker.reset(step)

    # Hack for inner model layers to pass some monitoring info to the trainer.
    set_global_monitor()

    _unwrapped_for_metrics = unwrap_model(model)

    # ---- File-based logging setup ----
    # 1) Re-add Hydra's FileHandler if torch.compile/DDP reset the root logger
    _root = logging.getLogger()
    if (
        not any(isinstance(h, logging.FileHandler) for h in _root.handlers)
        and accelerator.is_main_process
    ):
        _hydra_fh = logging.FileHandler(output_dir / "train.log", mode="a")
        _hydra_fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        _root.addHandler(_hydra_fh)
    # 2) Model debug log: direct fh.write(), immune to compile / logger config
    _inner_model = unwrap_model(model)
    if _dry_run:
        _model_log_fh = None
    else:
        _model_log_path = output_dir / "logs" / f"model_debug_rank{accelerator.process_index}.log"
        _model_log_path.parent.mkdir(parents=True, exist_ok=True)
        _model_log_fh = open(_model_log_path, "a")
    _inner_model._model_log_fh = _model_log_fh
    if hasattr(_inner_model, "model") and hasattr(_inner_model.model, "_model_log_fh"):
        _inner_model.model._model_log_fh = _model_log_fh
    if not _dry_run:
        logger.info(f"Model debug log: {_model_log_path}")

    # ── Box 5: Training Configuration ────────────────────────────────
    if accelerator.is_main_process and mfu_tracker is not None:
        _optim_type = (
            "AdamW8bit"
            if cfg.model.use_8bit_optimizer
            else ("AdamW (fused)" if cfg.get("use_fused_optimizer", False) else "AdamW")
        )
        _parallel = f"DDP ({accelerator.num_processes} GPU)"
        _lr_sched = cfg.model.lr_scheduler_type
        if cfg.model.lr_scheduler_type == "OneCycleLR":
            _warmup_info = f"pct_start={cfg.model.get('pct_start', '?')} / {max_steps} steps"
        else:
            _warmup_steps = locals().get("warmup_steps", 0)
            _warmup_info = (
                f"warmup={_warmup_steps}/{max_steps} steps"
                if _warmup_steps
                else f"no warmup / {max_steps} steps"
            )
        _gpu_label = getattr(mfu_tracker, "_gpu_name", "unknown")
        _peak_t = mfu_tracker.total_peak_flops / 1e12
        _flops_step = getattr(mfu_tracker, "_flops_per_step_T", 0.0)
        _num_params = getattr(mfu_tracker, "_num_params_M", 0.0)
        log_box(
            logger,
            "🚀  Training Configuration",
            [
                ("Parallelism", _parallel),
                ("Optimizer", f"{_optim_type}   lr={cfg.model.learning_rate}"),
                ("LR schedule", f"{_lr_sched}   {_warmup_info}"),
                None,
                ("GPU", f"{_gpu_label}   {_peak_t:.1f} TFLOPS ({mfu_tracker._training_mode})"),
                ("Model params", f"{_num_params:.2f} M"),
                ("FLOPs/step", f"{_flops_step:.2f} TFLOPs"),
            ],
        )

    # ── Pre-flight checks ─────────────────────────────────────────────────
    if accelerator.is_main_process:
        run_preflight_checks(
            logger=logger,
            model=_unwrapped_for_metrics,
            optimizer=optimizer,
            train_processor=train_processor,
            log_dir=output_dir / "logs",
        )

    # ── Token decode diagnostic (verify AR input/target text) ───────────
    if accelerator.is_main_process:
        from g05.utils.training.train_utils import log_sample_text_diagnostic

        log_sample_text_diagnostic(
            model=_unwrapped_for_metrics,
            train_dataset=train_dataset,
            train_processor=train_processor,
            device=device_id,
        )

    # ── VLM weight verification (text chat) ─────────────────────────────
    if accelerator.is_main_process:
        _vlm_chat_verification(model=_unwrapped_for_metrics, device=device_id)

    # Initialize experiment tracker (moved here so all setup logs precede wandb init)
    tracker_type = init_experiment_tracker(cfg, accelerator, output_dir)

    logger.info("Waiting for all processes to synchronize before training...")
    accelerator.wait_for_everyone()
    last_action_loss = None
    # Train!
    logger.info("Starting training...")
    training_done = False
    with tqdm.tqdm(initial=step, total=max_steps, leave=False, dynamic_ncols=True) as progress:
        latest_action_eval_batch = None
        _period_train_start = time.time()
        while not training_done:
            train_sampler.set_epoch(epoch)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)

            if resume_dataloader:
                logger.info(
                    f"Resume action dataloader state from batch_idx {action_batch_idx} of epoch {epoch}"
                )
                train_sampler.set_start_batch(action_batch_idx)
                resume_dataloader = False
            else:
                action_batch_idx = 0
                batch_idx = 0
                train_sampler.set_start_batch(0)

            data_iter = iter(train_dataloader)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            while batch_idx < len(train_dataloader):
                batch = next(data_iter)

                # First batch tokenizer evaluation (only once, controlled by flag).
                # Restricted to VQ-style tokenizers whose backend.encode returns a dict of
                # per-part codes used by the diagnostic.
                if EVAL_TOKENIZER_FIRST_BATCH and batch_idx == 0 and step == 0:
                    _inner_model = unwrap_model(model)
                    if hasattr(_inner_model, "action_tokenizer"):
                        from g05.tokenizer.interface.vq_base import (
                            VQActionTokenizer,
                        )

                        if isinstance(_inner_model.action_tokenizer, VQActionTokenizer):
                            _hf_tok = getattr(
                                getattr(_inner_model, "processor", None), "tokenizer", None
                            )
                            eval_tokenizer_first_batch(
                                _inner_model.action_tokenizer,
                                batch,
                                device_id,
                                hf_tokenizer=_hf_tok,
                            )

                if batch_idx == 0 and step == 0:
                    # Log actual pixel_values tensor shapes from first batch
                    _pv = batch.get("pixel_values")
                    if _pv is not None:
                        if isinstance(_pv, dict):
                            _shapes = {k: list(v.shape) for k, v in _pv.items()}
                        else:
                            _shapes = list(_pv.shape)
                        logger.info(f"[First Batch] pixel_values shape: {_shapes}")
                    save_train_snapshot(
                        batch,
                        output_dir,
                        parts_meta=eval_parts_meta,
                    )

                action_batch_idx += 1
                if overfit_batch is not None:
                    latest_action_eval_batch = batch

                # Turn off sync when is not optimizer step
                is_optimizer_step = (batch_idx + 1) % cfg.model.grad_accumulation_steps == 0
                sync_ctx = model.no_sync() if not is_optimizer_step else nullcontext()
                with sync_ctx:
                    with accelerator.autocast():
                        _monitor = get_global_monitor()
                        if _monitor is not None:
                            _monitor.reset()
                            _monitor.set_step(step + 1)
                        loss, loss_value_dict = model(batch)
                        last_action_loss = loss.item()
                    # Normalize loss to account for gradient accumulation
                    normalized_loss = loss / cfg.model.grad_accumulation_steps
                    normalized_loss.backward()

                batch_idx += 1

                if is_optimizer_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.model.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    action_loss_value = last_action_loss if last_action_loss is not None else 0.0
                    # Only all_reduce loss on log steps — loss is for logging only,
                    # no need to sync every optimizer step
                    is_log_step = (step + 1) % cfg.logger.log_steps == 0
                    if dist.is_initialized() and is_log_step:
                        loss_tensor = torch.tensor(
                            [action_loss_value], device=device_id, dtype=torch.float32
                        )
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        action_loss_value = (loss_tensor / dist.get_world_size()).item()

                    description = f"Epoch {epoch}, Step {step}"
                    if last_action_loss is not None:
                        description += f", Loss: {action_loss_value:.4f}"
                    progress.set_description(description)
                    progress.update()
                    progress.refresh()

                    if use_ema:
                        ema_model.update()

                    step += 1

                    # Log metrics on optimizer steps
                    if step % cfg.logger.log_steps == 0 and tracker_type != "none":
                        # Ensure values are plain Python numbers
                        loss_log_dict = {
                            k: (v.item() if hasattr(v, "item") else float(v))
                            for k, v in loss_value_dict.items()
                        }

                        log_dict = {
                            "lr": optimizer.param_groups[0]["lr"],
                            "grad_norm": grad_norm.item(),
                        }
                        log_dict.update(loss_log_dict)

                        _unwrapped = unwrap_model(model)
                        # Grad-accum-aware micro-batch means, logged separately for action/cot/caption.
                        _acc = _unwrapped.flush_train_accuracy()
                        if "overall" in _acc:
                            log_dict["train/overall_accuracy"] = _acc["overall"]
                        if "action_token" in _acc:
                            log_dict["train/action_token_accuracy"] = _acc["action_token"]
                        if "cot" in _acc:
                            log_dict["train/cot_accuracy"] = _acc["cot"]

                        if step % cfg.eval_steps == 0:
                            _train_time_in_period = time.time() - _period_train_start
                            if overfit_batch is not None and latest_action_eval_batch is not None:
                                logger.info(
                                    "Eval step %s: using current overfit training batch for periodic eval.",
                                    step,
                                )
                                eval_batch = latest_action_eval_batch
                            else:
                                logger.info(
                                    "Eval step %s: using validation dataloader batch for periodic eval.",
                                    step,
                                )
                                eval_batch = None
                            eval_log_dict = evaluator.evaluate(
                                unwrap_model(model),
                                accelerator,
                                step,
                                eval_batch=eval_batch,
                            )
                            _eval_time = eval_log_dict.pop("_eval_time_sec", 0.0)
                            log_dict.update(eval_log_dict)
                            _total_time = _train_time_in_period + _eval_time
                            if _total_time > 0:
                                log_dict["performance/train_eval_time_ratio"] = (
                                    _train_time_in_period / _total_time
                                )
                            _period_train_start = time.time()

                        # Add MFU metrics if tracker is available
                        if mfu_tracker is not None:
                            mfu_metrics = mfu_tracker.compute_metrics(step)
                            log_dict.update(mfu_metrics)

                        global_monitor = get_global_monitor()
                        if global_monitor is not None:
                            log_dict.update(global_monitor.get_metrics())

                        accelerator.log(log_dict, step=step)

                # Periodic sync to catch CUDA errors early
                if step > 0 and (step % 1000) == 0:
                    torch.cuda.synchronize()

                # Save checkpoint in the main process
                if step > 0 and (step % cfg.checkpointing_steps) == 0 and not _dry_run:
                    if accelerator.is_main_process:
                        logger.info(f"Saving model checkpoint for step {step} ...")
                        save_training_checkpoint(
                            output_dir,
                            step=step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            action_batch_idx=action_batch_idx,
                            model=unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema_model=ema_model if use_ema else None,
                        )
                        logger.info(f"step {step} checkpoint saved")
                    # All ranks wait for rank 0 to finish saving before resuming training,
                    # otherwise non-rank-0 may start next forward's all-gather while rank 0
                    # is still doing IO, causing NCCL timeout.
                    accelerator.wait_for_everyone()

                # Stop training when max_steps is reached
                if step >= max_steps:
                    logger.info(f"Max step {max_steps} reached, stop training ...")
                    training_done = True
                    break

            epoch += 1

    # Final checkpoint (inference-only: no optimizer/scheduler state)
    if not _dry_run and accelerator.is_main_process:
        logger.info(f"Saving model checkpoint for step {step} ...")
        save_training_checkpoint(
            output_dir,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            model=unwrap_model(model),
            optimizer=None,
            scheduler=None,
            ema_model=ema_model if use_ema else None,
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    finetune()

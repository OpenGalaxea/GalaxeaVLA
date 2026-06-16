#!/usr/bin/env python3
"""
Unit test for data pipeline smoke validation.

Validates that a mixture YAML (with per-embodiment processor configs) produces
a working dataset, processor, dataloader, and batch.

Supports three modes:
  1. Single dataset (primary): --dataset path/to/dataset.yaml. Automatically
     wrapped into mixture format, no manual mixture file needed.
  2. Mixture YAML: --mixture path/to/mixture.yaml. Per-embodiment configs +
     PROCESSOR_DEFAULTS provide the processor config used for instantiation.
  3. Legacy (backward compat): --task-config provides model.processor as the
     processor base, overriding per-embodiment defaults.

Usage:
    # Data YAML mode:
    python tests/test_dataloader_batch.py \
        --mixture configs/data/r1lite.yaml \
        --stats data/stats/r1lite_stats.json

    # Single dataset mode:
    python tests/test_dataloader_batch.py \
        --dataset path/to/single_dataset.yaml \
        --stats data/stats/r1lite_stats.json

    # Legacy mode (with task config):
    python tests/test_dataloader_batch.py \
        --task-config r1lite \
        --stats data/stats/r1lite_stats.json

    # Override mixture in legacy mode:
    python tests/test_dataloader_batch.py \
        --task-config r1lite \
        --mixture configs/data/r1lite.yaml \
        --stats data/stats/r1lite_stats.json
"""

import argparse
import math
import os
import sys
import termios
import tty
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from termcolor import colored, cprint
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset  # noqa: E402
from g05.data_processor.processor.mixture_processor import MixtureProcessor  # noqa: E402
from g05.utils.config.config_resolvers import register_default_resolvers  # noqa: E402
from g05.utils.data.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json  # noqa: E402

register_default_resolvers()


# ── Tree drawing characters ─────────────────────────────────────────────────
PIPE = "│   "
TEE = "├── "
ELBOW = "└── "
BLANK = "    "

# ── Processor defaults (model-dependent fields) ───────────────────────────
# These are the default fields used when a mixture-only test does not specify
# them. Per-embodiment configs may still override `_target_` and any other key.
# In training, task.processor overrides these via OmegaConf.merge.
PROCESSOR_DEFAULTS = {
    "_target_": "g05.data_processor.processor.galaxea_cot_processor.GalaxeaCoTProcessor",
    "cot_steps": None,
    "discrete_action": True,
    "num_output_cameras": 3,
    "pad_token_id": 0,
    "image_token_index": 257152,
    "tokenizer_params": {
        "pretrained_model_name_or_path": "google/paligemma-3b-pt-224",
        "local_files_only": True,
        "token": None,
    },
    "max_text_tokens": 160,
    "num_input_cameras": 3,
    "num_obs_steps": 1,
    "use_stepwise_action_norm": True,
}


def load_processor_defaults() -> DictConfig:
    """Return PROCESSOR_DEFAULTS as an OmegaConf DictConfig."""
    return OmegaConf.create(PROCESSOR_DEFAULTS)


# ── Collate function (from scripts/train_vq.py pattern) ────────────────────
def custom_collate_fn(batch):
    """Robust collate for nested dicts — handles gt_action, shape mismatches."""
    if not batch:
        return {}
    elem = batch[0]
    collated = {}
    for key in elem.keys():
        if key == "gt_action":
            collated[key] = [d[key] for d in batch]
        elif isinstance(elem[key], dict):
            collated[key] = custom_collate_fn([d[key] for d in batch])
        else:
            try:
                collated[key] = default_collate([d[key] for d in batch])
            except Exception:
                collated[key] = [d[key] for d in batch]
    return collated


# ── Pretty-print helpers ────────────────────────────────────────────────────
def _format_value(v: Any) -> str:
    """Format a single value for display."""
    if isinstance(v, torch.Tensor):
        type_s = colored("Tensor", "cyan")
        device_s = colored(str(v.device), "dark_grey")
        dtype_s = colored(str(v.dtype).replace("torch.", ""), "dark_grey")
        shape_s = colored(str(list(v.shape)), "yellow")
        parts = [type_s, device_s, dtype_s, shape_s]
        if v.numel() > 0 and v.is_floating_point():
            parts.append(colored(f"range=[{v.min().item():.4f}, {v.max().item():.4f}]", "green"))
        elif v.numel() > 0 and not v.is_floating_point() and v.dtype != torch.bool:
            parts.append(colored(f"range=[{v.min().item()}, {v.max().item()}]", "green"))
        return " | ".join(parts)

    if isinstance(v, np.ndarray):
        type_s = colored("ndarray", "cyan")
        dtype_s = colored(str(v.dtype), "dark_grey")
        shape_s = colored(str(list(v.shape)), "yellow")
        parts = [type_s, dtype_s, shape_s]
        if v.size > 0 and np.issubdtype(v.dtype, np.number):
            parts.append(colored(f"range=[{v.min():.4f}, {v.max():.4f}]", "green"))
        return " | ".join(parts)

    if isinstance(v, str):
        rep = repr(v)
        return (
            colored("str", "cyan")
            + " = "
            + colored(f"{rep[:80]}{'...' if len(rep) > 80 else ''}", "dark_grey")
        )

    if isinstance(v, (int, float, bool)):
        return colored(type(v).__name__, "cyan") + " = " + colored(str(v), "white")

    return colored(type(v).__name__, "cyan")


def _is_homogeneous_list(lst: list) -> Optional[str]:
    """Return the element type name if all elements share the same basic type."""
    if not lst:
        return None
    first_type = type(lst[0])
    if all(type(x) is first_type for x in lst):
        return first_type.__name__
    return None


def _print_node(prefix: str, connector: str, label: str, desc: str):
    """Print a single tree node: dim connector + bold key + description."""
    dim = colored(f"{prefix}{connector}", "dark_grey")
    key_s = colored(label, "white", attrs=["bold"])
    print(f"{dim}{key_s} \u2500 {desc}")


def print_batch_tree(data: Any, prefix: str = "", label: str = "Batch", is_last: bool = True):
    """Recursively print a nested batch structure as a tree."""
    connector = ELBOW if is_last else TEE
    child_prefix = prefix + (BLANK if is_last else PIPE)

    if isinstance(data, dict):
        _print_node(prefix, connector, label, colored(f"dict ({len(data)} keys)", "magenta"))
        keys = list(data.keys())
        for i, k in enumerate(keys):
            print_batch_tree(data[k], child_prefix, str(k), is_last=(i == len(keys) - 1))

    elif isinstance(data, list):
        elem_type = _is_homogeneous_list(data)
        if elem_type and elem_type in ("str", "int", "float", "bool"):
            _print_node(
                prefix,
                connector,
                label,
                colored(f"list[{elem_type}]", "cyan")
                + " | "
                + colored(f"len={len(data)}", "yellow"),
            )
        elif data and isinstance(data[0], dict):
            _print_node(prefix, connector, label, colored(f"list (len={len(data)})", "magenta"))
            print_batch_tree(data[0], child_prefix, "[0]", is_last=True)
        elif data and isinstance(data[0], torch.Tensor):
            shapes = [list(t.shape) for t in data[:3]]
            _print_node(
                prefix,
                connector,
                label,
                colored("list[Tensor]", "cyan")
                + " | "
                + colored(f"len={len(data)}", "yellow")
                + " | "
                + colored(f"first shapes={shapes}", "dark_grey"),
            )
        else:
            _print_node(prefix, connector, label, colored(f"list (len={len(data)})", "cyan"))
    else:
        _print_node(prefix, connector, label, _format_value(data))


def print_batch_tree_root(batch: dict, batch_size: int):
    """Print the batch tree starting from the root."""
    print()
    cprint(f"{'=' * 80}", "blue")
    cprint(f"  \U0001f4e6 BATCH CONTENTS ({batch_size} samples)", "blue", attrs=["bold"])
    cprint(f"{'=' * 80}", "blue")
    keys = list(batch.keys())
    for i, k in enumerate(keys):
        is_last = i == len(keys) - 1
        connector = ELBOW if is_last else TEE
        child_prefix = BLANK if is_last else PIPE
        v = batch[k]

        if isinstance(v, dict):
            _print_node("", connector, k, colored(f"dict ({len(v)} keys)", "magenta"))
            sub_keys = list(v.keys())
            for j, sk in enumerate(sub_keys):
                print_batch_tree(v[sk], child_prefix, str(sk), is_last=(j == len(sub_keys) - 1))
        elif isinstance(v, list):
            elem_type = _is_homogeneous_list(v)
            if elem_type and elem_type in ("str", "int", "float", "bool"):
                _print_node(
                    "",
                    connector,
                    k,
                    colored(f"list[{elem_type}]", "cyan")
                    + " | "
                    + colored(f"len={len(v)}", "yellow"),
                )
            elif v and isinstance(v[0], dict):
                _print_node("", connector, k, colored(f"list (len={len(v)})", "magenta"))
                print_batch_tree(v[0], child_prefix, "[0]", is_last=True)
            elif v and isinstance(v[0], torch.Tensor):
                shapes = [list(t.shape) for t in v[:3]]
                _print_node(
                    "",
                    connector,
                    k,
                    colored("list[Tensor]", "cyan")
                    + " | "
                    + colored(f"len={len(v)}", "yellow")
                    + " | "
                    + colored(f"first shapes={shapes}", "dark_grey"),
                )
            else:
                _print_node("", connector, k, colored(f"list (len={len(v)})", "cyan"))
        else:
            _print_node("", connector, k, _format_value(v))
    print()


# ── Processor structure printing ────────────────────────────────────────────
def _get_transform_desc(transform) -> str:
    """Get a short description of a transform object."""
    cls_name = colored(type(transform).__name__, "cyan")
    if hasattr(transform, "keys"):
        return f"{cls_name} (keys: {colored(str(transform.keys), 'dark_grey')})"
    if hasattr(transform, "category_keys"):
        return f"{cls_name} (category_keys: {colored(str(transform.category_keys), 'dark_grey')})"
    return cls_name


def print_processor_structure(mixture_processor: MixtureProcessor):
    """Print the internal structure of each per-embodiment processor."""
    print()
    cprint(f"{'=' * 80}", "blue")
    cprint("  \U0001f9e9 PROCESSOR STRUCTURE", "blue", attrs=["bold"])
    cprint(f"{'=' * 80}", "blue")

    processors = mixture_processor.processors
    emb_names = list(processors.keys())

    for idx, emb_name in enumerate(emb_names):
        proc = processors[emb_name]
        is_last_emb = idx == len(emb_names) - 1
        emb_connector = ELBOW if is_last_emb else TEE
        emb_prefix = BLANK if is_last_emb else PIPE

        cls_name = colored(type(proc).__name__, "cyan")
        dim = colored(emb_connector, "dark_grey")
        print(f"{dim}{colored(emb_name, 'white', attrs=['bold'])} \u2500 {cls_name}")

        # Collect attributes to print
        attrs: List[tuple] = []

        # normalizer
        if proc._normalizer is not None:
            attrs.append(
                (
                    "normalizer",
                    colored(type(proc._normalizer).__name__, "cyan")
                    + f" (mode: {colored(proc.norm_default_mode, 'yellow')})",
                )
            )
        else:
            attrs.append(("normalizer", colored("NOT SET", "red", attrs=["bold"])))

        # action_filter
        if hasattr(proc, "action_filter") and proc.action_filter is not None:
            af = proc.action_filter
            af_cls = colored(type(af).__name__, "cyan")
            extra = []
            for attr in (
                "joint_threshold",
                "gripper_threshold",
                "velocity_threshold",
                "eef_threshold",
            ):
                if hasattr(af, attr):
                    extra.append(f"{attr}={getattr(af, attr)}")
            if extra:
                af_cls += f" ({colored(', '.join(extra), 'dark_grey')})"
            attrs.append(("action_filter", af_cls))

        # action_state_transforms
        if hasattr(proc, "action_state_transforms") and proc.action_state_transforms:
            transforms_desc = [_get_transform_desc(t) for t in proc.action_state_transforms]
            attrs.append(("action_state_transforms", transforms_desc))
        else:
            attrs.append(("action_state_transforms", colored("None", "dark_grey")))

        # action_state_merger
        if hasattr(proc, "action_state_merger") and proc.action_state_merger is not None:
            merger = proc.action_state_merger
            m_cls = colored(type(merger).__name__, "cyan")
            merge_flag = getattr(merger, "merge", "?")
            attrs.append(
                ("action_state_merger", f"{m_cls} (merge={colored(str(merge_flag), 'yellow')})")
            )

        # norm info
        attrs.append(("norm_default_mode", colored(repr(proc.norm_default_mode), "yellow")))
        if hasattr(proc, "norm_exception_mode") and proc.norm_exception_mode:
            raw = (
                OmegaConf.to_container(proc.norm_exception_mode)
                if isinstance(proc.norm_exception_mode, DictConfig)
                else proc.norm_exception_mode
            )
            attrs.append(("norm_exception_mode", colored(str(raw), "dark_grey")))

        # shape_meta summary
        sm = proc.shape_meta
        action_n = len(sm.get("action", []))
        state_n = len(sm.get("state", []))
        images_n = len(sm.get("images", [])) if "images" in sm else 0
        attrs.append(
            (
                "shape_meta",
                f"action={colored(str(action_n), 'yellow')} keys, "
                f"state={colored(str(state_n), 'yellow')} keys, "
                f"images={colored(str(images_n), 'yellow')} keys",
            )
        )

        # train_transforms
        if hasattr(proc, "train_transforms") and proc.train_transforms:
            tt = proc.train_transforms
            if hasattr(tt, "keys"):
                cameras = list(tt.keys())
                attrs.append(
                    (
                        "train_transforms",
                        f"{colored(str(len(cameras)), 'yellow')} cameras: {colored(str(cameras), 'dark_grey')}",
                    )
                )
            else:
                attrs.append(("train_transforms", colored(repr(type(tt)), "dark_grey")))

        # Print attributes
        for j, (attr_name, attr_val) in enumerate(attrs):
            is_last_attr = j == len(attrs) - 1
            attr_connector = ELBOW if is_last_attr else TEE
            attr_child_prefix = BLANK if is_last_attr else PIPE
            dim = colored(f"{emb_prefix}{attr_connector}", "dark_grey")
            attr_label = colored(attr_name, "white")

            if isinstance(attr_val, list):
                print(f"{dim}{attr_label} ({colored(str(len(attr_val)), 'yellow')})")
                for k, item in enumerate(attr_val):
                    item_connector = ELBOW if k == len(attr_val) - 1 else TEE
                    dim2 = colored(f"{emb_prefix}{attr_child_prefix}{item_connector}", "dark_grey")
                    print(f"{dim2}[{k}] {item}")
            else:
                print(f"{dim}{attr_label}: {attr_val}")

    print()


# ── Decoupling gap analysis ─────────────────────────────────────────────────
def load_task_config(task_name: str) -> DictConfig:
    """Load a task config via Hydra compose — same resolution as finetune.sh."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(PROJECT_ROOT / "configs")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"task={task_name}"],
        )
    return cfg


def load_processor_base(task_cfg: DictConfig) -> DictConfig:
    """
    Extract the processor base from a resolved task config (model.processor).

    No hardcoded fallbacks — the test validates the REAL config.
    """
    if "model" not in task_cfg or "processor" not in task_cfg.model:
        cprint(
            "  \u274c  Task config has no model.processor — cannot build processors.",
            "red",
            attrs=["bold"],
        )
        sys.exit(1)

    base_cfg = task_cfg.model.processor.copy()
    OmegaConf.set_struct(base_cfg, False)
    return base_cfg


def load_dataset_base(mixture_cfg: DictConfig) -> DictConfig:
    """
    Load the dataset base from the mixture YAML's `dataset_base` key (legacy).
    New mixture files have these params inlined directly.
    """
    if "dataset_base" not in mixture_cfg:
        cprint(
            "  \u26a0\ufe0f  mixture YAML has no 'dataset_base' key \u2014 "
            "assuming new structure with inline dataset params.",
            "yellow",
        )
        return OmegaConf.create(
            {
                "_target_": "g05.data.mixture_lerobot_dataset.MixtureLerobotDataset",
                "use_weight_normalization": True,
                "use_weight_for_sampling": False,
                "action_size": 32,
                "past_action_size": 0,
                "obs_size": 1,
                "val_set_proportion": 0.05,
                "is_training_set": True,
            }
        )

    base_cfg = mixture_cfg["dataset_base"]
    base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    return base_cfg


def apply_dataset_runtime_overrides(
    datasets_cfg: DictConfig, stats_downsample_rate: Optional[int]
) -> int:
    """Apply CLI runtime overrides to per-embodiment dataset configs."""
    if stats_downsample_rate is None:
        return 0

    overridden = 0
    for emb_name in datasets_cfg:
        emb_cfg = datasets_cfg[emb_name]
        if emb_cfg is None:
            continue
        dataset_type = str(emb_cfg.get("type", ""))
        if "LerobotDatasetV3" not in dataset_type and "galaxea_lerobot_dataset" not in dataset_type:
            continue
        emb_cfg["stats_downsample_rate"] = int(stats_downsample_rate)
        overridden += 1
    return overridden


# ── Visualization helpers ──────────────────────────────────────────────────
def load_parts_meta(parts_meta_path: str) -> OrderedDict:
    """Load body-part dimensions from an explicit parts_meta yaml file."""
    meta_path = Path(parts_meta_path)
    if not meta_path.is_absolute():
        meta_path = PROJECT_ROOT / meta_path
    with open(meta_path) as f:
        raw = yaml.safe_load(f)
    return OrderedDict(raw["parts_meta"])


def _first_dim_value(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return _first_dim_value(value[0])
    return int(value)


def infer_parts_meta(batch: dict, parts_meta_path: Optional[str] = None) -> OrderedDict:
    """Infer visualization parts from the batch unless an explicit yaml is provided."""
    if parts_meta_path:
        return load_parts_meta(parts_meta_path)

    raw = batch.get("action_parts_meta")
    if isinstance(raw, dict):
        return OrderedDict((name, _first_dim_value(dim)) for name, dim in raw.items())
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return OrderedDict((name, int(dim)) for name, dim in raw[0].items())

    action_dim = int(batch["action"].shape[-1])
    return OrderedDict({"action": action_dim})


def split_by_parts(tensor: torch.Tensor, parts_meta: OrderedDict) -> Dict[str, torch.Tensor]:
    """Split a merged (..., D_total) tensor into per-part dict along the last dim."""
    result = {}
    offset = 0
    for part_name, dim in parts_meta.items():
        result[part_name] = tensor[..., offset : offset + dim]
        offset += dim
    return result


def compute_merged_mask(
    action_op_mask: torch.Tensor,
    action_is_pad: torch.Tensor,
    action_dim_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Compute per-element validity mask (T, D) for one sample.

    Args:
        action_op_mask: (D,) — True = dim is active
        action_is_pad:  (T,)  — True = timestep is padding
        action_dim_is_pad: (D,) — True = dim is padding (cross-embodiment)

    Returns:
        is_valid: (T, D) bool — True where data is meaningful
    """
    is_valid = (
        (~action_is_pad.unsqueeze(-1)) & (~action_dim_is_pad.unsqueeze(0)) & action_op_mask.bool()
    )
    return is_valid


def visualize_sample(
    sample_idx: int,
    batch: dict,
    parts_meta: OrderedDict,
    output_path: Path,
):
    """Create and save a matplotlib figure for one batch sample."""
    batch_size = batch["action"].shape[0]
    action = batch["action"][sample_idx]  # (T, D)
    proprio = batch["proprio"][sample_idx]  # (T_s, D)

    # Masks
    action_op_mask = batch["action_op_mask"][sample_idx]  # (D,)
    action_is_pad = batch["action_is_pad"][sample_idx]  # (T,)
    action_dim_is_pad = batch["action_dim_is_pad"][sample_idx]  # (D,)
    is_valid = compute_merged_mask(action_op_mask, action_is_pad, action_dim_is_pad)

    # Split into per-part
    action_parts = split_by_parts(action, parts_meta)
    proprio_parts = split_by_parts(proprio, parts_meta)
    mask_parts = split_by_parts(is_valid, parts_meta)

    # Embodiment & language
    emb_name = batch["embodiment"][sample_idx] if "embodiment" in batch else "?"
    language = batch["language"][sample_idx] if "language" in batch else "?"
    if len(language) > 60:
        language = language[:57] + "..."

    # Grid layout
    n_parts = len(parts_meta)
    cols = math.ceil(math.sqrt(n_parts))
    rows = math.ceil(n_parts / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat = np.array(axes).flatten()

    T = action.shape[0]
    t_steps = np.arange(T)

    for ax_idx, (part_name, dim) in enumerate(parts_meta.items()):
        ax = axes_flat[ax_idx]
        act_data = action_parts[part_name].numpy()  # (T, dim)
        prop_data = proprio_parts[part_name].numpy()  # (T_s, dim)
        mask_data = mask_parts[part_name].numpy()  # (T, dim)

        n_solid, n_dashed, n_mixed = 0, 0, 0
        cmap = plt.cm.tab10 if dim <= 10 else plt.cm.tab20
        for d in range(dim):
            color = cmap(d % cmap.N)
            dim_valid = mask_data[:, d]  # (T,) bool

            if dim_valid.all():
                n_solid += 1
                ax.plot(t_steps, act_data[:, d], "-", color=color, linewidth=1.0, label=f"d{d}")
            elif not dim_valid.any():
                n_dashed += 1
                ax.plot(
                    t_steps,
                    act_data[:, d],
                    "--",
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                    label=f"d{d} (pad)",
                )
            else:
                n_mixed += 1
                # Mixed: draw solid for valid, dashed for masked
                valid_y = np.where(dim_valid, act_data[:, d], np.nan)
                masked_y = np.where(~dim_valid, act_data[:, d], np.nan)
                ax.plot(t_steps, valid_y, "-", color=color, linewidth=1.0, label=f"d{d}")
                ax.plot(t_steps, masked_y, "--", color=color, linewidth=0.8, alpha=0.35)

            # State marker at t=-0.5 (use last obs step if multiple)
            state_val = prop_data[-1, d]
            ax.scatter([-0.5], [state_val], marker="*", color=color, s=30, zorder=5)

        print(
            f"    [VIS] {part_name:25s} solid={n_solid} dashed={n_dashed} mixed={n_mixed} y_range=[{act_data.min():.4f}, {act_data.max():.4f}]"
        )
        ax.set_title(f"{part_name} ({dim}d) s={n_solid}/d={n_dashed}", fontsize=9)
        ax.set_xlabel("t", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if dim <= 8:
            ax.legend(fontsize=5, ncol=max(1, dim // 2), loc="upper right")

    # Hide unused axes
    for ax_idx in range(n_parts, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        f"Sample {sample_idx + 1}/{batch_size}  |  {emb_name}  |  {language}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def read_key() -> Optional[str]:
    """Read a single keypress from stdin (raw terminal mode).

    Uses os.read() on the raw fd to bypass Python's TextIOWrapper buffering,
    which can swallow or reorder bytes from escape sequences.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            ch2 = os.read(fd, 1)
            ch3 = os.read(fd, 1)
            if ch2 == b"[":
                if ch3 == b"C":
                    return "right"
                if ch3 == b"D":
                    return "left"
        if ch in (b"q", b"Q"):
            return "quit"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def parse_missing_keys_from_error(error_msg: str) -> Dict[str, Set[str]]:
    """
    Parse missing keys from AssertionError message.

    Input format:
      "Embodiment 'galaxea_r1lite': Action keys in shape_meta are not a subset of stats keys. Missing: {'left_ee_pose', 'right_ee_pose'}"

    Returns:
      {"action": {"left_ee_pose", "right_ee_pose"}, "state": {...}}
    """
    import re

    result = {"action": set(), "state": set()}

    action_match = re.search(r"Action keys.*Missing: (\{[^}]+\})", error_msg)
    if action_match:
        keys_str = action_match.group(1)
        try:
            result["action"] = set(eval(keys_str))
        except Exception:
            pass

    state_match = re.search(r"State keys.*Missing: (\{[^}]+\})", error_msg)
    if state_match:
        keys_str = state_match.group(1)
        try:
            result["state"] = set(eval(keys_str))
        except Exception:
            pass

    return result


# ── Single dataset wrapper ──────────────────────────────────────────────────
DEFAULT_MIXTURE_PARAMS = {
    "_target_": "g05.data.mixture_lerobot_dataset.MixtureLerobotDataset",
    "use_weight_normalization": True,
    "use_weight_for_sampling": False,
    "action_size": 32,
    "past_action_size": 0,
    "obs_size": 1,
    "val_set_proportion": 0.05,
    "is_training_set": True,
}


def wrap_single_dataset_as_mixture(dataset_cfg: DictConfig) -> DictConfig:
    """
    Wrap a single dataset config into mixture format.

    Input (path/to/single_dataset.yaml):
        galaxea_r1lite: {...}
        processor: {...}

    Output:
        {
            _target_: MixtureLerobotDataset,
            embodiment_datasets: {galaxea_r1lite: {...}},
            processors: {galaxea_r1lite: {...}},
            use_weight_normalization: True,
            ...
        }
    """
    cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Dataset config must be a dict")

    reserved_keys = {"processor", "processors"}
    emb_names = [k for k in cfg.keys() if k not in reserved_keys]

    if not emb_names:
        raise ValueError(
            "No embodiment dataset found in config (expected top-level keys other than 'processor')"
        )

    mixture_cfg = OmegaConf.create(DEFAULT_MIXTURE_PARAMS)

    embodiment_datasets = {}
    processors = {}

    for emb_name in emb_names:
        emb_cfg = cfg[emb_name]
        if emb_cfg is None:
            continue
        embodiment_datasets[emb_name] = emb_cfg

    if "processor" in cfg:
        for emb_name in emb_names:
            processors[emb_name] = cfg["processor"]
    elif "processors" in cfg:
        processors_raw = cfg["processors"]
        if isinstance(processors_raw, dict):
            for emb_name in emb_names:
                if emb_name in processors_raw:
                    processors[emb_name] = processors_raw[emb_name]
                else:
                    processors[emb_name] = None

    mixture_cfg.embodiment_datasets = OmegaConf.create(embodiment_datasets)
    mixture_cfg.processors = OmegaConf.create(processors)

    return mixture_cfg


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    from accelerate import PartialState

    PartialState()

    parser = argparse.ArgumentParser(
        description="Unit test: build dataloader from mixture YAML and print batch tree"
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="(Optional, legacy) Hydra task config name (e.g. 'pretrain/6k_pretrain_debug'). "
        "When provided, model.processor overrides per-embodiment defaults.",
    )
    parser.add_argument(
        "--mixture",
        default=None,
        help="Mixture YAML path. If omitted, uses the mixture from --task-config or --dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset YAML path (e.g. path/to/single_dataset.yaml). "
        "Automatically wrapped into mixture format. Mutually exclusive with --mixture.",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Path to pre-computed dataset stats JSON",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--stats-downsample-rate",
        type=int,
        default=None,
        help="Override stats_downsample_rate for V3 datasets during online stats computation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enter interactive visualization mode after printing",
    )
    parser.add_argument(
        "--parts-meta",
        default=None,
        help=(
            "Optional parts_meta yaml for visualization. If omitted, part dimensions are "
            "read from batch['action_parts_meta']."
        ),
    )
    args = parser.parse_args()

    provided_sources = sum(
        [
            args.task_config is not None,
            args.mixture is not None,
            args.dataset is not None,
        ]
    )
    if provided_sources == 0:
        cprint(
            "\n  Error: at least one of --dataset, --mixture, or --task-config must be provided.",
            "red",
            attrs=["bold"],
        )
        parser.print_help()
        sys.exit(1)

    if args.mixture is not None and args.dataset is not None:
        cprint(
            "\n  Error: --mixture and --dataset are mutually exclusive. Please specify only one.",
            "red",
            attrs=["bold"],
        )
        parser.print_help()
        sys.exit(1)

    if args.stats_downsample_rate is not None and args.stats_downsample_rate <= 0:
        cprint(
            "\n  Error: --stats-downsample-rate must be a positive integer.",
            "red",
            attrs=["bold"],
        )
        parser.print_help()
        sys.exit(1)

    # ── 0. Determine mode: dataset / mixture-only / legacy ──────────────────────
    task_cfg = None
    if args.task_config:
        cprint(
            f"\n\U0001f527 [0/7] Loading task config: {args.task_config}", "blue", attrs=["bold"]
        )
        task_cfg = load_task_config(args.task_config)

    # Stats: CLI > task config's datastatics_path
    stats_path = args.stats
    if stats_path is None and task_cfg is not None:
        stats_path = getattr(task_cfg, "datastatics_path", None)

    # Mixture config loading (priority: --dataset > --mixture > task config)
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        assert dataset_path.exists(), f"Dataset YAML not found: {dataset_path}"
        cprint(
            f"\U0001f4c2 [1/7] Loading single dataset YAML: {dataset_path}", "blue", attrs=["bold"]
        )
        dataset_cfg = OmegaConf.load(dataset_path)
        OmegaConf.resolve(dataset_cfg)
        mixture_cfg = wrap_single_dataset_as_mixture(dataset_cfg)
        cprint("       Auto-wrapped into mixture format", "dark_grey")
    elif args.mixture:
        mixture_path = Path(args.mixture)
        if not mixture_path.is_absolute():
            mixture_path = PROJECT_ROOT / mixture_path
        assert mixture_path.exists(), f"Mixture YAML not found: {mixture_path}"
        cprint(f"\U0001f4c2 [1/7] Loading mixture YAML: {mixture_path}", "blue", attrs=["bold"])
        mixture_cfg = OmegaConf.load(mixture_path)
        OmegaConf.resolve(mixture_cfg)
    else:
        cprint(
            f"\U0001f4c2 [1/7] Using mixture from task config: {args.task_config}",
            "blue",
            attrs=["bold"],
        )
        mixture_cfg = task_cfg.data

    # Support both new structure (embodiment_datasets) and legacy structure (datasets)
    if "embodiment_datasets" in mixture_cfg:
        datasets_cfg = mixture_cfg["embodiment_datasets"]
        # New structure: dataset params are inlined in mixture yaml
        use_weight_normalization = mixture_cfg.get("use_weight_normalization", True)
        use_weight_for_sampling = mixture_cfg.get("use_weight_for_sampling", False)
        action_size = mixture_cfg.get("action_size", 32)
        past_action_size = mixture_cfg.get("past_action_size", 0)
        obs_size = mixture_cfg.get("obs_size", 1)
        obs_stride_second = mixture_cfg.get("obs_stride_second", 0.0)
        val_set_proportion = mixture_cfg.get("val_set_proportion", 0.05)
        is_training_set = mixture_cfg.get("is_training_set", True)
    else:
        # Legacy structure: dataset params from dataset_base
        datasets_cfg = mixture_cfg["datasets"]
        dataset_base = load_dataset_base(mixture_cfg)
        ds_base = OmegaConf.to_container(dataset_base, resolve=True)
        use_weight_normalization = ds_base.get("use_weight_normalization", True)
        use_weight_for_sampling = ds_base.get("use_weight_for_sampling", False)
        action_size = ds_base.get("action_size", 32)
        past_action_size = ds_base.get("past_action_size", 0)
        obs_size = ds_base.get("obs_size", 1)
        obs_stride_second = ds_base.get("obs_stride_second", 0.0)
        val_set_proportion = ds_base.get("val_set_proportion", 0.05)
        is_training_set = ds_base.get("is_training_set", True)

    processors_cfg = mixture_cfg.get("processors", {})

    emb_names = list(datasets_cfg.keys())
    print(f"       Found {colored(str(len(emb_names)), 'yellow')} embodiments: {emb_names}")

    # ── 2. Load base processor defaults ───────────────────────────────────
    if task_cfg is not None:
        # Legacy path: task.processor overrides per-embodiment defaults
        base_defaults = load_processor_base(task_cfg)
        cprint("       Mode: legacy (task.processor as base)", "dark_grey")
    else:
        # Mixture-only path: PROCESSOR_DEFAULTS provides model-dependent fields
        base_defaults = load_processor_defaults()
        cprint("       Mode: mixture-only (PROCESSOR_DEFAULTS as base)", "dark_grey")

    # ── 3. Create MixtureLerobotDataset ─────────────────────────────────
    cprint("\U0001f4be [3/7] Creating MixtureLerobotDataset...", "blue", attrs=["bold"])
    datasets_container = OmegaConf.to_container(datasets_cfg, resolve=True)
    datasets_cfg_mutable = OmegaConf.create(datasets_container)
    OmegaConf.set_struct(datasets_cfg_mutable, False)
    overridden_stats_rate = apply_dataset_runtime_overrides(
        datasets_cfg_mutable,
        args.stats_downsample_rate,
    )
    if overridden_stats_rate > 0:
        cprint(
            f"       🎯 Override stats_downsample_rate={args.stats_downsample_rate} for {overridden_stats_rate} V3 dataset(s)",
            "yellow",
        )

    try:
        train_dataset = MixtureLerobotDataset(
            embodiment_datasets=datasets_cfg_mutable,
            use_weight_normalization=use_weight_normalization,
            use_weight_for_sampling=use_weight_for_sampling,
            action_size=action_size,
            past_action_size=past_action_size,
            obs_size=obs_size,
            obs_stride_second=obs_stride_second,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            n_datasets=mixture_cfg.get("n_datasets", None),
        )
        cprint(
            f"       ✅ Dataset created: {len(train_dataset)} samples, "
            f"{len(train_dataset.datasets)} sub-datasets",
            "green",
        )
        all_types = train_dataset.all_embodiment_types
        cprint(
            f"       ✅ All embodiment types: {all_types}",
            "green",
        )
        cprint(
            f"       \u2705 Dataset created: {len(train_dataset)} samples, "
            f"{len(train_dataset.datasets)} sub-datasets",
            "green",
        )
        cprint(
            f"       \u2705 All embodiment types: {train_dataset.all_embodiment_types}",
            "green",
        )
    except Exception as e:
        import traceback

        cprint(f"       ❌ Dataset creation FAILED: {e}", "red", attrs=["bold"])
        traceback.print_exc()
        sys.exit(1)

    # ── 4. Build processors ─────────────────────────────────────────────
    cprint("\n\u2699\ufe0f  [4/7] Building processors...", "blue", attrs=["bold"])
    if task_cfg is not None:
        merge_note = "merge order: per_emb + task.processor (task wins)"
    else:
        merge_note = "merge order: PROCESSOR_DEFAULTS + per_emb (per_emb wins)"
    print(colored(f"       ({merge_note})", "dark_grey"))

    train_processors = {}
    failed_processors = []
    for emb_name in processors_cfg:
        emb_proc_cfg = processors_cfg[emb_name]
        if emb_proc_cfg is None:
            emb_proc_cfg = OmegaConf.create({})
        else:
            emb_proc_cfg = emb_proc_cfg.copy()
        OmegaConf.set_struct(emb_proc_cfg, False)

        if task_cfg is not None:
            # Legacy: task.processor has higher priority (overrides per-emb)
            merged_cfg = OmegaConf.merge(emb_proc_cfg, base_defaults)
        else:
            # Mixture-only: per-emb wins over PROCESSOR_DEFAULTS
            merged_cfg = OmegaConf.merge(base_defaults, emb_proc_cfg)
        OmegaConf.set_struct(merged_cfg, False)

        try:
            proc = instantiate(merged_cfg)
            train_processors[emb_name] = proc
            cprint(f"       \u2705 {emb_name} \u2192 {type(proc).__name__}", "green")
        except Exception as e:
            cprint(f"       \u274c {emb_name} \u2192 FAILED: {e}", "red")
            failed_processors.append(emb_name)

    if failed_processors:
        cprint(
            f"\n       \u274c {len(failed_processors)} processor(s) failed to instantiate. "
            f"Cannot continue.",
            "red",
            attrs=["bold"],
        )
        sys.exit(1)

    mixture_processor = MixtureProcessor(train_processors)
    cprint(
        f"       \u2705 MixtureProcessor created with {len(train_processors)} processors", "green"
    )

    # ── 5. Load / compute normalizer stats ──────────────────────────────
    required_embodiments = set(mixture_processor.processors.keys())
    total_dataset_samples = len(train_dataset)
    total_frames = sum(
        sum(int(getattr(inner_ds, "num_frames", 0)) for inner_ds in ds.multi_dataset._datasets)
        for ds in train_dataset.datasets
    )
    total_episodes = sum(
        sum(int(getattr(inner_ds, "num_episodes", 0)) for inner_ds in ds.multi_dataset._datasets)
        for ds in train_dataset.datasets
    )

    if stats_path:
        stats_path = Path(stats_path)
        cprint("\n\U0001f4ca [5/7] Loading normalizer stats...", "blue", attrs=["bold"])
        print(f"       {colored(str(stats_path), 'dark_grey')}")
        cprint(
            "       📏 Dataset samples={} | raw frames={} | episodes={} | groups={}".format(
                f"{total_dataset_samples:,}",
                f"{total_frames:,}",
                f"{total_episodes:,}",
                len(train_dataset.datasets),
            ),
            "cyan",
        )
        if not stats_path.exists():
            cprint(
                "       \u26a0\ufe0f  Stats file not found. Computing stats online and creating it...",
                "yellow",
            )
            dataset_stats = train_dataset.get_dataset_stats(mixture_processor)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            cprint("       \U0001f4be  Saving computed stats...", "yellow")
            save_dataset_stats_to_json(dataset_stats, stats_path)
            cprint(
                f"       \u2705 Stats file created with {len(dataset_stats)} embodiment(s)",
                "green",
            )
        else:
            loaded_stats = load_dataset_stats_from_json(stats_path)

            loaded_embodiments = set(loaded_stats.keys())
            missing_embodiments = required_embodiments - loaded_embodiments

            if missing_embodiments:
                cprint(
                    f"       \u26a0\ufe0f  Missing stats for {len(missing_embodiments)} embodiment(s): "
                    f"{sorted(missing_embodiments)}",
                    "yellow",
                )
                cprint(
                    "       \U0001f504  Computing missing stats online for missing embodiments...",
                    "yellow",
                )
                computed_stats = train_dataset.get_dataset_stats(
                    mixture_processor,
                    embodiments=missing_embodiments,
                )

                dataset_stats = dict(loaded_stats)
                for emb_name in missing_embodiments:
                    if emb_name in computed_stats:
                        dataset_stats[emb_name] = computed_stats[emb_name]

                cprint("       \U0001f4be  Updating stats file...", "yellow")
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                save_dataset_stats_to_json(dataset_stats, stats_path)
                cprint(
                    f"       \u2705 Stats file updated with {len(missing_embodiments)} new embodiment(s)",
                    "green",
                )
            else:
                dataset_stats = loaded_stats
                cprint("       \u2705 All embodiment stats found in file", "green")
    else:
        cprint(
            "\n\U0001f4ca [5/7] Computing normalizer stats (this may take a while)...",
            "blue",
            attrs=["bold"],
        )
        dataset_stats = train_dataset.get_dataset_stats(mixture_processor)
    cprint(f"       \u2705 Stats loaded for {len(dataset_stats)} embodiment types", "green")

    # ── 6. Set normalizer and assign processor to dataset ───────────────
    cprint(
        "\n\U0001f517 [6/7] Setting normalizer & assigning processors to dataset...",
        "blue",
        attrs=["bold"],
    )

    import re

    max_iterations = 20
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        try:
            mixture_processor.set_normalizer_from_stats(dataset_stats)
            train_dataset.set_processor(mixture_processor)
            if iteration > 1:
                cprint("       \u2705 All missing keys resolved.", "green")
            else:
                cprint("       \u2705 Done.", "green")
            break
        except AssertionError as e:
            error_msg = str(e)
            if "Missing:" not in error_msg:
                cprint(f"       \u274c Failed: {e}", "red", attrs=["bold"])
                sys.exit(1)

            if iteration == 1:
                cprint(
                    f"       \u26a0\ufe0f  Some keys missing from stats, computing incrementally...",
                    "yellow",
                )

            emb_match = re.search(r"Embodiment '([^']+)'", error_msg)
            if not emb_match:
                cprint(
                    f"       \u274c Failed to parse embodiment from error: {error_msg}",
                    "red",
                    attrs=["bold"],
                )
                sys.exit(1)

            emb_name = emb_match.group(1)
            missing_keys = parse_missing_keys_from_error(error_msg)

            missing_action = missing_keys["action"]
            missing_state = missing_keys["state"]

            if not missing_action and not missing_state:
                cprint(
                    f"       \u274c Failed to parse missing keys from error: {error_msg}",
                    "red",
                    attrs=["bold"],
                )
                sys.exit(1)

            if missing_action:
                cprint(
                    f"       \U0001f504  [{iteration}] Computing missing action keys for '{emb_name}': {missing_action}",
                    "yellow",
                )
            if missing_state:
                cprint(
                    f"       \U0001f504  [{iteration}] Computing missing state keys for '{emb_name}': {missing_state}",
                    "yellow",
                )

            only_keys_for_emb = {}
            if missing_action:
                only_keys_for_emb["action"] = missing_action
            if missing_state:
                only_keys_for_emb["state"] = missing_state

            computed_stats = train_dataset.get_dataset_stats(
                mixture_processor, only_keys={emb_name: only_keys_for_emb}
            )

            if emb_name in computed_stats:
                for category in ["action", "state"]:
                    if category in computed_stats[emb_name]:
                        if category not in dataset_stats[emb_name]:
                            dataset_stats[emb_name][category] = {}
                        for key, key_stats in computed_stats[emb_name][category].items():
                            dataset_stats[emb_name][category][key] = key_stats

            cprint(f"       \U0001f4be  Updating stats file...", "yellow")
            save_dataset_stats_to_json(dataset_stats, stats_path)
            cprint(
                f"       \u2705 [{iteration}] Stats file updated for '{emb_name}'",
                "green",
            )
    else:
        cprint(
            f"       \u274c Max iterations ({max_iterations}) reached, still have missing keys",
            "red",
            attrs=["bold"],
        )
        sys.exit(1)

    # ── 7. Create DataLoader and fetch one batch ────────────────────────
    cprint(
        f"\n\U0001f680 [7/7] Creating DataLoader (batch_size={args.batch_size}, "
        f"num_workers={args.num_workers})...",
        "blue",
        attrs=["bold"],
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
    )

    try:
        batch = next(iter(dataloader))
        cprint(f"       \u2705 Batch fetched! Keys: {list(batch.keys())}", "green")
    except Exception as e:
        cprint(f"       \u274c Batch fetch FAILED: {e}", "red", attrs=["bold"])
        sys.exit(1)

    # ── Print processor structure ───────────────────────────────────────
    print_processor_structure(mixture_processor)

    # ── Print batch tree ────────────────────────────────────────────────
    print_batch_tree_root(batch, args.batch_size)

    # ── Final verdict ───────────────────────────────────────────────────
    cprint(f"{'=' * 80}", "blue")
    cprint("  \u2705 TEST PASSED \u2014 dataloader works.", "green", attrs=["bold"])
    cprint(f"{'=' * 80}", "blue")
    print()

    # ── Interactive visualization ──────────────────────────────────────
    if args.visualize:
        parts_meta = infer_parts_meta(batch, getattr(args, "parts_meta", None))
        output_dir = PROJECT_ROOT / "tmp" / "unit_tests" / "test_dataloader_batch"
        # Clean stale images from previous runs to avoid network filesystem cache confusion.
        if output_dir.exists():
            for old_img in output_dir.glob("vis_sample*.png"):
                old_img.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        B = batch["action"].shape[0]
        sample_idx = 0
        prev_path: Optional[Path] = None

        while True:
            # ── Debug: per-part mask diagnosis ──
            _si = sample_idx
            _op = batch["action_op_mask"][_si]  # (D,)
            _dp = batch["action_dim_is_pad"][_si]  # (D,)
            _ap = batch["action_is_pad"][_si]  # (T,)
            _offset = 0
            _emb = batch["embodiment"][_si] if "embodiment" in batch else "?"
            _lang = batch["language"][_si] if "language" in batch else "?"
            print(f"\n{'─' * 60}")
            print(f"  [DEBUG] Sample {_si}  emb={_emb}  action_is_pad any={_ap.any().item()}")
            print(f"  [DEBUG] language={_lang[:80]}")
            for _pname, _pdim in parts_meta.items():
                _op_slice = _op[0, _offset : _offset + _pdim]
                _dp_slice = _dp[_offset : _offset + _pdim]
                _act_slice = batch["action"][_si, :, _offset : _offset + _pdim]
                print(
                    f"  {_pname:25s} op_mask={_op_slice.tolist()}  "
                    f"dim_is_pad={_dp_slice.tolist()}  "
                    f"act_abs_max={_act_slice.abs().max().item():.6f}"
                )
                _offset += _pdim
            print(f"{'─' * 60}")

            # Use a unique filename per sample to bypass network filesystem client cache.
            image_path = output_dir / f"vis_sample{sample_idx}_{_emb}.png"
            visualize_sample(sample_idx, batch, parts_meta, image_path)
            # Remove previous file after new one is saved
            if prev_path is not None and prev_path != image_path and prev_path.exists():
                prev_path.unlink()
            prev_path = image_path
            print(
                f"Saved: {image_path}  |  "
                f"Sample {sample_idx + 1}/{B}  |  "
                f"Press \u2192 next, \u2190 prev, q quit"
            )
            # Wait until a recognized key is pressed (ignore unknown keys)
            key = None
            while key is None:
                key = read_key()
            if key == "right":
                sample_idx = (sample_idx + 1) % B
            elif key == "left":
                sample_idx = (sample_idx - 1) % B
            elif key == "quit":
                break

        if prev_path is not None and prev_path.exists():
            prev_path.unlink()
        print("Visualization exited.")


# ═══════════════════════════════════════════════════════════════════════════
# pytest wrappers (dual-mode: pytest + CLI)
# ═══════════════════════════════════════════════════════════════════════════

import pytest  # noqa: E402

_has_real_data = Path("configs/data/debug.yaml").exists()
requires_real_data = pytest.mark.skipif(not _has_real_data, reason="Real data not available")


@requires_real_data
class TestDataloaderBatch:
    """pytest wrapper using debug mixture to verify the full pipeline."""

    def test_load_mixture_and_build_batch(self):
        """Verify mixture -> dataset -> processor -> dataloader -> batch pipeline."""
        from omegaconf import OmegaConf  # noqa: F811

        mixture_path = PROJECT_ROOT / "configs/data/debug.yaml"
        mixture_cfg = OmegaConf.load(mixture_path)
        OmegaConf.resolve(mixture_cfg)

        assert "embodiment_datasets" in mixture_cfg or "datasets" in mixture_cfg

        if "embodiment_datasets" in mixture_cfg:
            datasets_cfg = mixture_cfg["embodiment_datasets"]
        else:
            datasets_cfg = mixture_cfg["datasets"]

        processors_cfg = mixture_cfg.get("processors", {})
        emb_names = list(datasets_cfg.keys())
        assert len(emb_names) > 0, "No embodiments found in mixture"

        # Build dataset
        datasets_container = OmegaConf.to_container(datasets_cfg, resolve=True)
        datasets_cfg_mutable = OmegaConf.create(datasets_container)
        OmegaConf.set_struct(datasets_cfg_mutable, False)
        apply_dataset_runtime_overrides(datasets_cfg_mutable, args.stats_downsample_rate)

        train_dataset = MixtureLerobotDataset(
            embodiment_datasets=datasets_cfg_mutable,
            use_weight_normalization=mixture_cfg.get("use_weight_normalization", True),
            use_weight_for_sampling=mixture_cfg.get("use_weight_for_sampling", False),
            action_size=mixture_cfg.get("action_size", 32),
            past_action_size=mixture_cfg.get("past_action_size", 0),
            obs_size=mixture_cfg.get("obs_size", 1),
            obs_stride_second=mixture_cfg.get("obs_stride_second", 0.0),
            val_set_proportion=mixture_cfg.get("val_set_proportion", 0.05),
            is_training_set=True,
            n_datasets=mixture_cfg.get("n_datasets", None),
        )
        assert len(train_dataset) > 0, "Dataset is empty"

        # Build processors
        base_defaults = load_processor_defaults()
        train_processors = {}
        for emb_name in processors_cfg:
            emb_proc_cfg = processors_cfg[emb_name]
            if emb_proc_cfg is None:
                emb_proc_cfg = OmegaConf.create({})
            else:
                emb_proc_cfg = emb_proc_cfg.copy()
            OmegaConf.set_struct(emb_proc_cfg, False)
            merged_cfg = OmegaConf.merge(base_defaults, emb_proc_cfg)
            OmegaConf.set_struct(merged_cfg, False)
            proc = instantiate(merged_cfg)
            train_processors[emb_name] = proc

        mixture_processor = MixtureProcessor(train_processors)
        assert len(train_processors) > 0, "No processors created"

        # Set normalizer (compute on-the-fly for test)
        dataset_stats = train_dataset.get_dataset_stats(mixture_processor)
        mixture_processor.set_normalizer_from_stats(dataset_stats)
        train_dataset.set_processor(mixture_processor)

        # Create DataLoader and fetch one batch
        dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
        batch = next(iter(dataloader))
        assert "action" in batch, "Batch missing 'action' key"
        assert batch["action"].ndim == 3, "Action should be 3D (B, T, D)"


if __name__ == "__main__":
    main()

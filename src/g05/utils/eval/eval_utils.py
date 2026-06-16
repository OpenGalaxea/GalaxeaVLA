"""eval_utils.py

Shared utilities for eval scripts (eval_tokenizer.py, eval_open_loop.py).
Extracted to eliminate duplicated code across evaluation pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from g05.utils.logging.logging_config import get_logger
from g05.utils.logging.log_box import log_box

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scalar / dict helpers
# ---------------------------------------------------------------------------


def to_scalar(value: Any) -> float:
    """Convert tensor or scalar to float."""
    if torch.is_tensor(value):
        return float(value.item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def concat_dict_list(dict_list: list[dict]) -> dict:
    """Concatenate a list of identically-keyed dicts along axis=0.

    Example: [{k: (B1,T,D)}, {k: (B2,T,D)}] -> {k: (B1+B2, T, D)}
    """
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def truncate_datasets(
    cfg: DictConfig,
    max_datasets: int,
) -> None:
    """Truncate dataset_dirs to at most *max_datasets* entries (debug mode).

    Supports both mixture configs (embodiment_datasets → dataset_groups → dataset_dirs)
    and flat configs (data.dataset_dirs).
    """
    OmegaConf.set_struct(cfg, False)
    try:
        if "embodiment_datasets" not in cfg.data:
            # Flat config: dataset_dirs directly under cfg.data
            dirs = cfg.data.get("dataset_dirs")
            if dirs and len(dirs) > max_datasets:
                cfg.data.dataset_dirs = dirs[:max_datasets]
            return
        truncated = {}
        for emb_name, emb_cfg in cfg.data.embodiment_datasets.items():
            for group in emb_cfg.get("dataset_groups") or []:
                dirs = group.get("dataset_dirs")
                if dirs and len(dirs) > max_datasets:
                    truncated[emb_name] = (len(dirs), max_datasets)
                    group.dataset_dirs = dirs[:max_datasets]
        if truncated:
            rows = [f"  max_datasets={max_datasets}   ({len(truncated)} embodiments truncated)"]
            for emb, (orig, kept) in truncated.items():
                rows.append(f"  {emb:<36}  {orig:>5} → {kept}")
            log_box(logger, "✂  Dataset dirs truncated (--max_datasets)", rows, inner_width=62)
    finally:
        OmegaConf.set_struct(cfg, True)


def truncate_embodiments(cfg: DictConfig, max_embodiments: int) -> None:
    """Keep only the first *max_embodiments* embodiments (debug mode).

    Cleans up cfg.data.processors and cfg.model.mixture_processors accordingly.
    """
    if "embodiment_datasets" not in cfg.data:
        return
    all_keys = list(cfg.data.embodiment_datasets.keys())
    if len(all_keys) <= max_embodiments:
        return
    remove_keys = all_keys[max_embodiments:]
    kept_keys = all_keys[:max_embodiments]
    OmegaConf.set_struct(cfg, False)
    for k in remove_keys:
        del cfg.data.embodiment_datasets[k]
        if cfg.data.get("processors") and k in cfg.data.processors:
            del cfg.data.processors[k]
        if cfg.model.get("mixture_processors") and k in cfg.model.mixture_processors:
            del cfg.model.mixture_processors[k]
    OmegaConf.set_struct(cfg, True)
    rows = [f"  {len(all_keys)} → {max_embodiments} embodiments kept"]
    rows.append("  Kept: " + ", ".join(kept_keys))
    rows.append("  Removed: " + ", ".join(remove_keys))
    log_box(logger, "✂  Embodiments truncated (--max_embodiments)", rows, inner_width=72)


def filter_embodiment(cfg: DictConfig, eval_embodiment: str) -> None:
    """Keep only datasets/processors matching the specified embodiment_type."""
    if "embodiment_datasets" not in cfg.data:
        return
    all_emb_keys = list(cfg.data.embodiment_datasets.keys())
    keep_emb_keys = [
        k
        for k in all_emb_keys
        if str(cfg.data.embodiment_datasets[k].embodiment_type) == eval_embodiment
    ]
    if not keep_emb_keys:
        available_types = sorted(
            {str(v.embodiment_type) for v in cfg.data.embodiment_datasets.values()}
        )
        raise ValueError(
            f"eval_embodiment={eval_embodiment!r} not found as an embodiment_type. "
            f"Available embodiment_types: {available_types}"
        )
    OmegaConf.set_struct(cfg, False)
    for k in all_emb_keys:
        if k not in keep_emb_keys:
            del cfg.data.embodiment_datasets[k]
    if cfg.data.get("processors"):
        for k in list(cfg.data.processors.keys()):
            if str(k) != eval_embodiment:
                del cfg.data.processors[k]
    if cfg.model.get("mixture_processors"):
        for k in list(cfg.model.mixture_processors.keys()):
            if str(k) != eval_embodiment:
                del cfg.model.mixture_processors[k]
    OmegaConf.set_struct(cfg, True)
    logger.info(
        f"  Filtered to embodiment_type: {eval_embodiment} ({len(keep_emb_keys)} dataset source(s))"
    )


# ---------------------------------------------------------------------------
# Episode boundary helpers
# ---------------------------------------------------------------------------


def compute_valid_episode_indices(
    dataset, frame_offset: int | None = None, frame_end: int | None = None
) -> tuple[int, int, list[int]]:
    """Compute episode indices that fall within the dataset's active frame range.

    Returns:
        (frame_offset, frame_end, valid_episode_indices)
    """
    if frame_offset is None:
        frame_offset = getattr(dataset, "_start_idx", 0)
    if frame_end is None:
        frame_end = getattr(dataset, "_end_idx", int(dataset.episode_data_index["to"][-1]))

    all_ep_from = dataset.episode_data_index["from"]
    all_ep_to = dataset.episode_data_index["to"]

    valid_episode_indices = [
        i
        for i in range(len(all_ep_from))
        if int(all_ep_from[i]) >= frame_offset and int(all_ep_to[i]) <= frame_end
    ]
    return frame_offset, frame_end, valid_episode_indices

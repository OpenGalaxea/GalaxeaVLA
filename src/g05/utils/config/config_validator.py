"""Config validation utilities for G0.5 training entry points.

Called after OmegaConf.resolve(cfg) to catch common misconfiguration errors
early — before GPU allocation or dataset loading.

Checks performed:
  1. model.tokenizer.vq_config.ckpt_dir — path must exist on disk (if set)
  2. model.model_arch.action_dim vs sum(tokenizer.vq_config.max_action_shape_meta.values())
     — must match when both are concrete integers / dicts
"""

import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def validate_train_config(cfg: DictConfig) -> None:
    """Validate a fully-resolved Hydra training config.

    Must be called **after** ``OmegaConf.resolve(cfg)`` so all interpolations
    are expanded and concrete values are available.

    Raises:
        ValueError: if any hard-error checks fail (collects all errors first,
                    then raises once with all messages).
    """
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Check 1: tokenizer ckpt_dir must exist on disk (if provided)
    # ------------------------------------------------------------------
    ckpt_dir = OmegaConf.select(cfg, "model.tokenizer.vq_config.ckpt_dir", default=None)
    if ckpt_dir is not None and isinstance(ckpt_dir, str):
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.exists():
            errors.append(
                f"[ckpt_dir] model.tokenizer.vq_config.ckpt_dir does not exist: "
                f"{ckpt_dir!r}\n"
                f"  Hint: update the tokenizer config or copy the checkpoint first "
                f"(see Bug #7 in MEMORY.md)."
            )

    # ------------------------------------------------------------------
    # Check 2: action_dim must equal sum(max_action_shape_meta.values()) when both present.
    #
    # NOTE: We compare action_dim against max_action_shape_meta (the task-specific
    # embodiment's actual action space), NOT against parts_meta.
    # parts_meta describes the tokenizer's training distribution (all possible
    # representations across embodiments) and is intentionally larger than any
    # single robot's action_dim — comparing the two would always fail.
    # ------------------------------------------------------------------
    action_dim = OmegaConf.select(cfg, "model.model_arch.action_dim", default=None)
    max_action_shape_meta = OmegaConf.select(
        cfg, "model.tokenizer.vq_config.max_action_shape_meta", default=None
    )

    if action_dim is not None and max_action_shape_meta is not None:
        if isinstance(action_dim, int) and isinstance(max_action_shape_meta, (dict, DictConfig)):
            meta_sum = sum(int(v) for v in max_action_shape_meta.values())
            if action_dim != meta_sum:
                errors.append(
                    f"[action_dim] model.model_arch.action_dim={action_dim} != "
                    f"sum(tokenizer.vq_config.max_action_shape_meta.values())={meta_sum}\n"
                    f"  max_action_shape_meta keys: {list(max_action_shape_meta.keys())}\n"
                    f"  Hint: action_dim and max_action_shape_meta must refer to the same "
                    f"embodiment action space (see PaddingActionMerger in CLAUDE.md)."
                )

    # ------------------------------------------------------------------
    # Raise if any hard errors accumulated
    # ------------------------------------------------------------------
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))

    logger.debug("validate_train_config: all checks passed.")

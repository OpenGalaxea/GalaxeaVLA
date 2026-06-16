"""
Pre-flight checks for training startup.

Call run_preflight_checks() after optimizer creation, before training loop.
Surfaces "config → constructed object" correctness using log_box style.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer

from g05.utils.logging.log_box import log_box


def run_preflight_checks(
    logger: logging.Logger,
    model: "nn.Module",
    optimizer: "Optimizer",
    train_processor,  # reserved for per-embodiment filter checks
    log_dir: "Path | None" = None,
) -> None:
    """Run all pre-flight checks and log results. All checks are informational (soft).

    Args:
        log_dir: If provided, writes the full (non-truncated) no-decay parameter
            list to ``log_dir/no_decay_params.txt``.
    """
    _print_no_decay_groups(logger, model, optimizer, log_dir=log_dir)
    _print_model_training_state(logger, model)


def _is_positive_scalar(val) -> bool:
    """Check if value is a positive scalar (int or float)."""
    return isinstance(val, (int, float)) and val > 0


def _is_nonempty_sequence(val) -> bool:
    """Check if value is a non-empty sequence (list/tuple) but not string."""
    return hasattr(val, "__len__") and not isinstance(val, str) and len(val) > 0


def _fmt_threshold(val: float) -> str:
    """Format threshold value for display.

    Uses scientific notation for values < 0.1, otherwise decimal.
    """
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val < 0.1:
        return f"{val:.1e}"
    if abs_val >= 1000:
        return f"{val:.1e}"
    return f"{val:.4g}"


def build_filter_str(action_filter) -> str:
    """Return abbreviated threshold string for an action filter object.

    Returns '-' if filter is None, DummyActionFilter, or has no active thresholds.
    Otherwise returns e.g. 'J=0.01 G=20.0' using the uppercased first letter of
    each threshold attribute name — generalises to any future threshold keys.

    For dim_thresholds with list values (per-dim thresholds), shows abbreviated
    form like 'L=[min~max]' to convey the threshold range without verbose output.
    """
    from g05.data_processor.transforms.action_filter import DummyActionFilter

    if action_filter is None or isinstance(action_filter, DummyActionFilter):
        return "-"

    # Named scalar threshold attributes on BaseActionFilter (and subclasses)
    named_attrs = [
        "joint_threshold",
        "gripper_threshold",
        "velocity_threshold",
        "eef_threshold",
    ]
    parts: list[str] = []
    for attr in named_attrs:
        val = getattr(action_filter, attr, None)
        if _is_positive_scalar(val):
            abbrev = attr[0].upper()
            parts.append(f"{abbrev}={_fmt_threshold(val)}")

    # dim_thresholds: arbitrary dict keyed by dimension name.
    # Values can be scalars or lists (per-dim thresholds).
    # Note: first-letter abbreviation may collide with named attrs (J/G/V/E)
    # if a dim_thresholds key starts with the same letter. Acceptable for now
    # since no current embodiment triggers this.
    for key, val in (getattr(action_filter, "dim_thresholds", None) or {}).items():
        if val is None:
            continue
        abbrev = key[0].upper()

        # Handle different value types
        if _is_positive_scalar(val):
            parts.append(f"{abbrev}={_fmt_threshold(val)}")
        elif _is_nonempty_sequence(val):
            # Per-dim thresholds: show range [min~max]
            val_list = list(val)
            min_v, max_v = min(val_list), max(val_list)
            parts.append(f"{abbrev}=[{_fmt_threshold(min_v)}~{_fmt_threshold(max_v)}]")

    return " ".join(parts) if parts else "-"


# ---------------------------------------------------------------------------
# Section 1: No-Decay Parameter Groups
# ---------------------------------------------------------------------------


def _print_no_decay_groups(
    logger: logging.Logger,
    model: "nn.Module",
    optimizer: "Optimizer",
    log_dir: "Path | None" = None,
) -> None:
    """Print only the param groups that have significantly less weight_decay than the max.

    Uses relative comparison so configs with tiny global wd (e.g. 1e-10) still show
    the genuinely no-decay groups (wd=0). Skipped if all groups have the same wd.

    Args:
        log_dir: If provided, writes the complete (non-truncated) parameter name list
            to ``log_dir/no_decay_params.txt``.
    """
    all_groups = optimizer.param_groups
    if not all_groups:
        return

    max_wd = max(g.get("weight_decay", 0.0) for g in all_groups)
    # Threshold: either absolute 1e-9 or 0.1% of the max wd, whichever is larger.
    # Groups below this threshold are shown as "no-decay" special cases.
    threshold = max(1e-9, max_wd * 0.001)
    no_decay_groups = [g for g in all_groups if g.get("weight_decay", 0.0) < threshold]

    if not no_decay_groups:
        return

    # Map tensor id → param name for informative display
    param_id_to_name: dict[int, str] = {id(p): name for name, p in model.named_parameters()}

    rows: list = []  # terminal: 5 patterns per group, truncated
    file_rows: list = []  # file: all patterns, one per line, no truncation

    for g in no_decay_groups:
        group_name = g.get("name", "no_decay")
        wd = g.get("weight_decay", 0.0)
        total_params_m = sum(p.numel() for p in g["params"]) / 1e6

        # Collect all param names for this group
        all_names: list[str] = sorted(param_id_to_name.get(id(p), "?") for p in g["params"])
        patterns_set: set[str] = set()
        for pname in all_names:
            parts = pname.rsplit(".", 2)
            pattern = ".".join(parts[-2:]) if len(parts) >= 2 else pname
            patterns_set.add(f"*.{pattern}")
        sorted_patterns = sorted(patterns_set)

        # Terminal: first 5 patterns + "+N more" hint
        terminal_patterns = "  ".join(sorted_patterns[:5])
        if len(sorted_patterns) > 5:
            terminal_patterns += f"  ... (+{len(sorted_patterns) - 5} more)"
        rows.append(
            f"  {group_name:<18}  wd={wd:.2e}  {total_params_m:>7.1f}M  {terminal_patterns}"
        )

        # File: header row + one pattern per line
        file_rows.append(
            f"  {group_name:<18}  wd={wd:.2e}  {total_params_m:>7.1f}M  ({len(sorted_patterns)} patterns)"
        )
        for pat in sorted_patterns:
            file_rows.append(f"      {pat}")
        file_rows.append(None)  # separator between groups

    log_box(logger, "⚙  No-Decay Parameter Groups", rows, file_rows=file_rows)

    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            out_path = log_dir / "no_decay_params.txt"
            # Also write the raw param name list (full paths, not just *.suffix patterns)
            raw_lines: list[str] = []
            for g in no_decay_groups:
                group_name = g.get("name", "no_decay")
                wd = g.get("weight_decay", 0.0)
                total_params_m = sum(p.numel() for p in g["params"]) / 1e6
                all_names = sorted(param_id_to_name.get(id(p), "?") for p in g["params"])
                raw_lines.append(f"[{group_name}]  wd={wd:.2e}  {total_params_m:.1f}M params")
                for name in all_names:
                    raw_lines.append(f"  {name}")
                raw_lines.append("")
            out_path.write_text("\n".join(raw_lines), encoding="utf-8")
            logger.debug(f"Full no-decay parameter list written to {out_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to write no_decay_params.txt: {exc}")


# ---------------------------------------------------------------------------
# Section 2: Model Training State
# ---------------------------------------------------------------------------


def _print_model_training_state(
    logger: logging.Logger,
    model: "nn.Module",
) -> None:
    """Print frozen parameters and eval-mode submodules. Skipped if all trainable+train."""
    frozen_prefixes = _summarize_frozen_modules(model)
    eval_roots = _summarize_eval_modules(model)

    if not frozen_prefixes and not eval_roots:
        return

    rows: list = []
    if frozen_prefixes:
        rows.append("  Frozen (no grad):")
        for prefix, param_m in frozen_prefixes:
            rows.append(f"    {prefix:<40}  {param_m:>7.1f}M params")
    if eval_roots:
        if rows:
            rows.append(None)  # separator between sections
        rows.append("  Eval mode:")
        for name in eval_roots:
            rows.append(f"    {name}")

    log_box(logger, "⚠  Model Training State", rows)


def _summarize_frozen_modules(model: "nn.Module") -> list[tuple[str, float]]:
    """Return list of (top_level_module_name, param_count_M) for frozen modules."""
    from collections import defaultdict

    top_level: dict[str, float] = defaultdict(float)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            top = name.split(".")[0]
            top_level[top] += param.numel() / 1e6

    return list(top_level.items())


def _summarize_eval_modules(model: "nn.Module") -> list[str]:
    """Return names of submodules in eval mode, showing only roots of eval subtrees.

    Note: if model.eval() is called on the root before this check, all children's
    parent resolves to the root (training=False) and are suppressed — call this
    only when the root module is in train mode.
    """
    module_map = dict(model.named_modules())
    eval_roots: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if not module.training:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = module_map.get(parent_name)
            # Only show this module if its parent is in train mode (root of eval subtree)
            if parent is None or parent.training:
                eval_roots.append(name)
    return eval_roots

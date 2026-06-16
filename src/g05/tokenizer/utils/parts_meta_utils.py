"""Utilities for computing derived values from parts_meta.

Dynamically computes offsets, total dimensions, and merged dimensions from the
parts_meta dict, removing hardcoded magic numbers from the code.

Usage (three-level merge_spec, WBCActionPreprocessor):
    from g05.tokenizer.utils.parts_meta_utils import (
        compute_part_offsets, compute_total_action_dim, compute_merged_dims,
        compute_merge_layout, build_default_repr_info, SlotLayout, GroupLayout,
    )
    offsets = compute_part_offsets(parts_meta)  # {'left_arm': (0, 8), ...}
    total   = compute_total_action_dim(parts_meta)  # sum of configured part dims
    merged  = compute_merged_dims(parts_meta)  # {'left_limb': 21, ...}
    layout  = compute_merge_layout(parts_meta, merge_spec)  # GroupLayout per group
    default = build_default_repr_info(merge_spec)  # {'left_limb/control': True, ...}

Usage (two-level merge_spec, GroupedPaddingMerger):
    from g05.tokenizer.utils.parts_meta_utils import (
        GroupedSlotLayout, compute_grouped_dims, compute_grouped_layout,
    )
    grouped_dims = compute_grouped_dims(parts_meta, merge_spec)  # {'left_control': 9, ...}
    grouped_layout = compute_grouped_layout(parts_meta, merge_spec)  # [GroupedSlotLayout, ...]

    merge_spec format: {alternative_name: [raw_key, ...]} — list of mutually exclusive keys
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SlotLayout:
    """Layout information for one slot."""

    slot_name: str  # e.g. 'control'
    alt_names: List[str]  # e.g. ['arm', 'ee_pose']
    part_names: List[str]  # e.g. ['left_arm', 'left_ee_pose']
    part_dims: List[int]  # e.g. [8, 9]
    max_dim: int  # e.g. 9


@dataclass
class GroupLayout:
    """Layout information for one group."""

    group_name: str  # e.g. 'left_limb'
    slots: List[SlotLayout] = field(default_factory=list)
    merged_dim: int = 0


def compute_part_offsets(parts_meta: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """Compute the (start, end) offsets for each part in the flat action.

    Example: {'left_arm': (0, 8), 'right_arm': (8, 16), 'left_gripper': (16, 17), ...}
    """
    offsets = {}
    cursor = 0
    for name, dim in parts_meta.items():
        offsets[name] = (cursor, cursor + dim)
        cursor += dim
    return offsets


def compute_total_action_dim(parts_meta: Dict[str, int]) -> int:
    """Return the total action dimension implied by parts_meta."""
    return sum(parts_meta.values())


# WBC 3-group merge spec: mapping from 12 parts to 3 merged groups.
# Each group consists of several slots, and each slot has multiple mutually
# exclusive optional parts. merged dim = sum(max(alternative_dims) for each slot).
MERGE_SPEC = {
    "left_limb": {
        "control": {"arm": "left_arm", "ee_pose": "left_ee_pose"},
        "end_effector": {"gripper": "left_gripper", "hand": "left_hand"},
    },
    "right_limb": {
        "control": {"arm": "right_arm", "ee_pose": "right_ee_pose"},
        "end_effector": {"gripper": "right_gripper", "hand": "right_hand"},
    },
    "others": {
        "torso": {"position": "torso", "velocity": "torso.velocities"},
        "chassis": {"position": "chassis", "velocity": "chassis.velocities"},
    },
}


def compute_merged_dims(parts_meta: Dict[str, int], merge_spec: Dict = None) -> Dict[str, int]:
    """Derive merged_parts_meta from parts_meta + merge_spec.

    Each group dim = sum(max(alternative_dims) for each slot).
    Use max(dim) instead of first_existing_dim because the merged tensor must fit
    any alternative; different samples in a cross-embodiment batch may use different
    alternatives.

    Returns: {'left_limb': N, 'right_limb': M, 'others': K}

    Args:
        parts_meta: mapping from part name to dim
        merge_spec: optional; falls back to MERGE_SPEC when None

    Note:
        At least one alternative only needs to exist in parts_meta.
        If multiple exist, use max(dim); if none exist, skip that slot.
    """
    spec = merge_spec if merge_spec is not None else MERGE_SPEC
    merged = {}
    for group_name, slots in spec.items():
        total = 0
        for slot_name, alternatives in slots.items():
            existing = [
                (alt_name, part_name)
                for alt_name, part_name in alternatives.items()
                if part_name in parts_meta
            ]
            if not existing:
                logger.warning(
                    f"Slot '{group_name}/{slot_name}': no alternatives found in parts_meta "
                    f"(looked for {list(alternatives.values())}), skipping"
                )
                continue
            max_dim = max(parts_meta[p] for _, p in existing)
            total += max_dim
        if total > 0:
            merged[group_name] = total
    return merged


def compute_merge_layout(
    parts_meta: Dict[str, int],
    merge_spec: Dict = None,
) -> Dict[str, GroupLayout]:
    """Precompute all group/slot layouts, called once during __init__.

    Only uses alternatives that actually exist in parts_meta.
    If multiple exist, choose the first; if none exist, skip that slot.

    Returns:
        {'left_limb': GroupLayout(...), 'right_limb': ..., 'others': ...}
    """
    spec = merge_spec if merge_spec is not None else MERGE_SPEC
    layout = {}
    for group_name, slots in spec.items():
        gl = GroupLayout(group_name=group_name)
        for slot_name, alternatives in slots.items():
            existing = [
                (alt_name, part_name)
                for alt_name, part_name in alternatives.items()
                if part_name in parts_meta
            ]
            if not existing:
                continue
            alt_names = [e[0] for e in existing]
            part_names = [e[1] for e in existing]
            part_dims = [parts_meta[p] for p in part_names]
            sl = SlotLayout(
                slot_name=slot_name,
                alt_names=alt_names,
                part_names=part_names,
                part_dims=part_dims,
                # others group always uses second alternatives (velocities), so need max dim
                max_dim=max(part_dims),
            )
            gl.slots.append(sl)
        gl.merged_dim = sum(s.max_dim for s in gl.slots)
        layout[group_name] = gl
    return layout


def build_default_repr_info(merge_spec: Dict = None) -> Dict[str, bool]:
    """Mark all slots as selecting the first alternative -> True.

    Returns:
        {'left_limb/control': True, 'left_limb/end_effector': True, ...}
    """
    spec = merge_spec if merge_spec is not None else MERGE_SPEC
    info: Dict[str, bool] = {}
    for group_name, slots in spec.items():
        for slot_name in slots:
            info[f"{group_name}/{slot_name}"] = True
    return info


@dataclass
class GroupedSlotLayout:
    """Layout information for one alternative name in a two-level merge_spec."""

    name: str  # alternative name, e.g. 'left_control'
    part_names: List[str]  # e.g. ['left_arm', 'left_ee_pose']
    part_dims: List[int]  # e.g. [8, 9]
    max_dim: int  # e.g. 9 — max(part_dims)


def compute_grouped_dims(parts_meta: Dict[str, int], merge_spec: Dict) -> Dict[str, int]:
    """Derive grouped_parts_meta from parts_meta + two-level merge_spec.

    merge_spec format: {alternative_name: [raw_key, ...]}
    Example: {'left_control': ['left_arm', 'left_ee_pose']}

    Each alternative-name dim = max(alternative_dims).
    At least one alternative only needs to exist in parts_meta.
    """
    grouped = {}
    for name, alternatives in merge_spec.items():
        existing = [p for p in alternatives if p in parts_meta]
        if not existing:
            logger.warning(
                f"Group '{name}': no alternatives found in parts_meta "
                f"(looked for {list(alternatives)}), skipping"
            )
            continue
        max_dim = max(parts_meta[p] for p in existing)
        grouped[name] = max_dim
    return grouped


def compute_grouped_layout(
    parts_meta: Dict[str, int],
    merge_spec: Dict,
) -> List[GroupedSlotLayout]:
    """Precompute two-level merge_spec layout, called once in GroupedPaddingMerger.__init__.

    Returns:
        List[GroupedSlotLayout], ordered by merge_spec keys.
    """
    layout = []
    for name, alternatives in merge_spec.items():
        existing = [p for p in alternatives if p in parts_meta]
        if not existing:
            continue
        part_dims = [parts_meta[p] for p in existing]
        layout.append(
            GroupedSlotLayout(
                name=name,
                part_names=existing,
                part_dims=part_dims,
                max_dim=max(part_dims),
            )
        )
    return layout


def compute_grouped_keys(parts_meta: Dict[str, int], merge_spec: Dict) -> Dict[str, int]:
    """Derive grouped key names and dims from merge_spec.

    If merge_spec is provided, each group name becomes a key with dim = max(alternative_dims).
    Keys not covered by any group are appended as residual keys in parts_meta order.
    If merge_spec is empty, returns dict(parts_meta) unchanged.
    """
    if not merge_spec:
        return dict(parts_meta)

    grouped = {}
    covered_keys = set()
    for group_name, raw_keys in merge_spec.items():
        max_dim = 0
        for rk in raw_keys:
            if rk in parts_meta:
                max_dim = max(max_dim, parts_meta[rk])
                covered_keys.add(rk)
        if max_dim > 0:
            grouped[group_name] = max_dim

    for k, v in parts_meta.items():
        if k not in covered_keys:
            grouped[k] = v

    return grouped

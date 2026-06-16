"""Parts meta padding utilities for action tokenizer.

This module provides functions to handle parts_meta mismatch between
dataset and model:
- check_parts_meta_subset: Check if dataset parts_meta is subset of model
- pad_action_to_model_dim: Pad action from dataset dim to model dim
- unpad_action_to_input_dim: Unpad action back to original input dim
"""

from collections import OrderedDict
from typing import List, Tuple

import torch


def check_parts_meta_subset(
    input_parts_meta: OrderedDict,
    model_parts_meta: OrderedDict,
) -> Tuple[bool, List[str], List[str]]:
    """Check if input_parts_meta is a subset of model_parts_meta.

    Subset means:
    1. All keys in input_parts_meta exist in model_parts_meta
    2. All dimensions match for those keys

    Args:
        input_parts_meta: Dataset's parts_meta (for example a compact gripper layout)
        model_parts_meta: Model's parts_meta (for example a larger shared layout)

    Returns:
        is_subset: True if input is subset of model
        missing_keys: Keys in model but not in input (need padding)
        mismatch_keys: Keys with dimension mismatch
    """
    input_keys = list(input_parts_meta.keys())
    model_keys = list(model_parts_meta.keys())

    missing_keys = []
    mismatch_keys = []

    for key in input_keys:
        if key not in model_keys:
            mismatch_keys.append(f"{key}: not in model")
        elif input_parts_meta[key] != model_parts_meta[key]:
            mismatch_keys.append(
                f"{key}: input={input_parts_meta[key]}, model={model_parts_meta[key]}"
            )

    for key in model_keys:
        if key not in input_keys:
            missing_keys.append(key)

    is_subset = len(mismatch_keys) == 0
    return is_subset, missing_keys, mismatch_keys


def pad_action_to_model_dim(
    action: torch.Tensor,
    input_parts_meta: OrderedDict | List[OrderedDict],
    model_parts_meta: OrderedDict,
    action_dim_is_pad: torch.Tensor | None = None,
    action_op_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Pad action from input dimension to model dimension.

    Args:
        action: (b, h, input_dim) action tensor
        input_parts_meta: Dataset's parts_meta
        model_parts_meta: Model's parts_meta
        action_dim_is_pad: (b, input_dim) or None, True = padding dimension
        action_op_mask: (b, 1, input_dim) or (b, h, input_dim) or None

    Returns:
        padded_action: (b, h, model_dim) padded tensor
        padded_dim_is_pad: (b, model_dim) updated padding mask
        padded_op_mask: Same shape as original action_op_mask but model_dim
        is_subset: True if padding succeeded
    """
    model_dim = sum(model_parts_meta.values())

    if action.shape[-1] == model_dim:
        return action, action_dim_is_pad, action_op_mask, True

    B, H, D = action.shape
    device = action.device
    dtype = action.dtype

    if isinstance(input_parts_meta, list):
        if len(input_parts_meta) != B:
            raise ValueError(
                f"input_parts_meta list length {len(input_parts_meta)} must match batch size {B}"
            )
        input_parts_meta_list = [OrderedDict(m) for m in input_parts_meta]
    else:
        input_parts_meta_list = [OrderedDict(input_parts_meta) for _ in range(B)]

    model_offsets = {}
    cursor = 0
    for key, dim in model_parts_meta.items():
        model_offsets[key] = (cursor, cursor + dim)
        cursor += dim

    padded_action = torch.zeros(B, H, model_dim, device=device, dtype=dtype)

    if action_dim_is_pad is not None:
        padded_dim_is_pad = torch.ones(B, model_dim, device=device, dtype=action_dim_is_pad.dtype)
    else:
        padded_dim_is_pad = torch.ones(B, model_dim, device=device, dtype=torch.bool)

    if action_op_mask is not None:
        if action_op_mask.dim() == 3:
            op_H = action_op_mask.shape[1]
            padded_op_mask = torch.zeros(
                B, op_H, model_dim, device=device, dtype=action_op_mask.dtype
            )
        else:
            padded_op_mask = torch.zeros(B, model_dim, device=device, dtype=action_op_mask.dtype)
    else:
        padded_op_mask = None

    for b in range(B):
        current_parts_meta = input_parts_meta_list[b]
        is_subset, _, mismatch_keys = check_parts_meta_subset(current_parts_meta, model_parts_meta)
        if not is_subset:
            return action, action_dim_is_pad, action_op_mask, False

        input_dim = sum(current_parts_meta.values())
        if input_dim != D:
            raise ValueError(
                f"input_parts_meta sums to {input_dim}, but action.shape[-1]={D}. "
                f"parts_meta={current_parts_meta}"
            )

        input_offsets = {}
        cursor = 0
        for key, dim in current_parts_meta.items():
            input_offsets[key] = (cursor, cursor + dim)
            cursor += dim

        for key in current_parts_meta.keys():
            in_s, in_e = input_offsets[key]
            out_s, out_e = model_offsets[key]
            if (in_e - in_s) != (out_e - out_s):
                raise ValueError(
                    f"Dim mismatch for key {key}: input={in_e - in_s}, model={out_e - out_s}"
                )
            padded_action[b, :, out_s:out_e] = action[b, :, in_s:in_e]
            if action_dim_is_pad is not None:
                padded_dim_is_pad[b, out_s:out_e] = action_dim_is_pad[b, in_s:in_e]
            else:
                padded_dim_is_pad[b, out_s:out_e] = False

            if padded_op_mask is not None:
                if action_op_mask.dim() == 3:
                    padded_op_mask[b, :, out_s:out_e] = action_op_mask[b, :, in_s:in_e]
                else:
                    padded_op_mask[b, out_s:out_e] = action_op_mask[b, in_s:in_e]

    return padded_action, padded_dim_is_pad, padded_op_mask, True


def unpad_action_to_input_dim(
    action: torch.Tensor,
    input_action_dim: int,
    action_dim_is_pad: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpad action from model dimension back to input dimension (prefix truncation).

    WARNING: This is a simple prefix truncation and is only correct when the input
    parts form a strict prefix of the model parts_meta ordering. For cross-embodiment
    cases where input parts are not a prefix (for example single-arm mapped to a larger layout),
    use unpad_action_by_parts_meta instead.

    Args:
        action: (b, h, model_dim) action tensor from decode
        input_action_dim: Original input dimension before padding
        action_dim_is_pad: (b, model_dim) or None

    Returns:
        unpadded_action: (b, h, input_action_dim)
        output_dim_is_pad: (b, input_action_dim) padding mask for output
    """
    model_dim = action.shape[-1]

    if input_action_dim >= model_dim:
        return action, action_dim_is_pad

    unpadded_action = action[:, :, :input_action_dim].clone()

    if action_dim_is_pad is not None:
        output_dim_is_pad = action_dim_is_pad[:, :input_action_dim].clone()
    else:
        output_dim_is_pad = torch.zeros(
            action.shape[0], input_action_dim, device=action.device, dtype=torch.bool
        )

    return unpadded_action, output_dim_is_pad


def unpad_action_by_parts_meta(
    action: torch.Tensor,
    input_parts_meta: OrderedDict,
    model_parts_meta: OrderedDict,
) -> torch.Tensor:
    """Semantic unpad: extract parts from model_dim space by key name, in input order.

    This is the symmetric inverse of pad_action_to_model_dim. For each key in
    input_parts_meta, it extracts the corresponding slice from the model-dim action
    tensor (using model_parts_meta offsets) and concatenates them in input order.

    Args:
        action: (b, h, model_dim) decoded action tensor
        input_parts_meta: Original input parts layout, e.g. {right_ee_pose: 9, right_gripper: 1}
        model_parts_meta: Full model parts layout.

    Returns:
        Tensor (b, h, input_dim) with correct semantic content
    """
    B, H, model_dim = action.shape
    input_dim = sum(input_parts_meta.values())

    # Compute model offsets
    model_offsets = {}
    cursor = 0
    for key, dim in model_parts_meta.items():
        model_offsets[key] = (cursor, cursor + dim)
        cursor += dim

    result = torch.zeros(B, H, input_dim, device=action.device, dtype=action.dtype)
    out_cursor = 0
    for key, dim in input_parts_meta.items():
        if key not in model_offsets:
            raise KeyError(
                f"unpad_action_by_parts_meta: key '{key}' not found in model_parts_meta. "
                f"Model keys: {list(model_parts_meta.keys())}"
            )
        src_s, src_e = model_offsets[key]
        result[:, :, out_cursor : out_cursor + dim] = action[:, :, src_s:src_e]
        out_cursor += dim

    return result

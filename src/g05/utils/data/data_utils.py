import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

# ---------------------------------------------------------------------------
# JSON Serialization Utilities
# ---------------------------------------------------------------------------


def to_json_serializable(x: Any) -> Any:
    """Convert tensors/arrays to JSON-serializable types.

    Recursively converts numpy arrays and torch tensors to Python lists,
    handling nested dicts and lists/tuples.

    Args:
        x: Any object, potentially containing tensors/arrays.

    Returns:
        JSON-serializable version of x.
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if np is not None and isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: to_json_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json_serializable(v) for v in x]
    return str(x)


# ---------------------------------------------------------------------------
# Collate Functions
# ---------------------------------------------------------------------------


def custom_collate_fn(batch):
    """Custom collate function for handling nested dicts with inconsistent structures.

    Handles:
    - `gt_action` field: returns as list without stacking
    - Nested dicts: recursively collates
    - Other fields: uses default collate, falls back to list on shape mismatch

    Args:
        batch: List of sample dicts from dataset.

    Returns:
        Collated batch dict.
    """
    if not batch:
        return {}

    elem = batch[0]
    collated = {}

    for key in elem.keys():
        # 1. gt_action: keep as list (may have different shapes)
        if key == "gt_action":
            collated[key] = [d[key] for d in batch]
        # 2. Nested dict: recursive collation
        elif isinstance(elem[key], dict):
            collated[key] = custom_collate_fn([d[key] for d in batch])
        # 3. Other fields: try default collate, fall back to list
        else:
            try:
                collated[key] = default_collate([d[key] for d in batch])
            except Exception:
                # Shape mismatch or other issues: keep as list
                collated[key] = [d[key] for d in batch]

    return collated


def dwt_collate_fn(
    batch: List[Dict[str, Any]],
    parts_meta: Dict[str, int],
) -> Dict[str, Any]:
    """Collate with DWT pre-computation for ActionCodec2.

    Precompute DWT during DataLoader collate to avoid using GPU forward time.
    The extra dwt_parts field is ignored by other tokenizers and does not affect them.

    Args:
        batch: list of data samples
        parts_meta: {part_name: dim} part-dimension mapping
            Example: {"left_arm": 8, "right_arm": 8, "left_gripper": 1, ...}

    Returns:
        Batch containing original fields + dwt_parts:
        - "action": (B, T, D_total) raw action
        - "dwt_parts": {part_name: {"A3":..., "D3":..., "D2":..., "D1":...}}
        - other fields unchanged
    """
    import pywt

    collated = custom_collate_fn(batch)

    if "action" not in collated:
        return collated

    actions = collated["action"]
    if not isinstance(actions, torch.Tensor):
        return collated

    B, T, D_total = actions.shape
    dwt_parts = {}

    start_idx = 0
    for part_name, dim in parts_meta.items():
        end_idx = start_idx + dim
        if end_idx > D_total:
            break

        part_action = actions[:, :, start_idx:end_idx]
        start_idx = end_idx

        part_numpy = part_action.cpu().numpy()
        coeffs = pywt.wavedec(part_numpy, "db4", level=3, axis=1, mode="periodization")

        dwt_parts[part_name] = {
            "A3": torch.from_numpy(coeffs[0]).float(),
            "D3": torch.from_numpy(coeffs[1]).float(),
            "D2": torch.from_numpy(coeffs[2]).float(),
            "D1": torch.from_numpy(coeffs[3]).float(),
        }

    collated["dwt_parts"] = dwt_parts
    return collated


def get_parts_meta_from_processor_cfg(processor_cfg: Any) -> Dict[str, int]:
    """Extract parts_meta from processor config.

    Priority:
    1. processor_cfg.action_state_merger.max_action_shape_meta
    2. default dual_arm config

    Args:
        processor_cfg: processor config (DictConfig or dict)

    Returns:
        {part_name: dim} mapping
    """
    DEFAULT_PARTS_META = {
        "left_arm": 8,
        "right_arm": 8,
        "left_gripper": 1,
        "right_gripper": 1,
        "left_ee_pose": 9,
        "right_ee_pose": 9,
    }

    if processor_cfg is None:
        return DEFAULT_PARTS_META

    merger_cfg = processor_cfg.get("action_state_merger", None)
    if merger_cfg is None:
        return DEFAULT_PARTS_META

    max_action_shape_meta = merger_cfg.get("max_action_shape_meta", None)
    if max_action_shape_meta is None:
        return DEFAULT_PARTS_META

    if hasattr(max_action_shape_meta, "items"):
        return {k: int(v) for k, v in max_action_shape_meta.items()}

    return DEFAULT_PARTS_META


def collate_fn_pad_sequences(batch, padding_input_id: int = 0):
    """Assemble training/inference batch. See models/g05/batch_schema.py for output contract."""
    batch_collated = {}
    if "samples" in batch[0]:
        samples = [item.pop("samples") for item in batch]
        batch_collated["samples"] = samples
    sample_meta_keys = ("idx", "task", "embodiment", "dataset_locator", "frequency")
    batch_collated["sample_meta"] = [
        {key: item.get(key) for key in sample_meta_keys if key in item} for item in batch
    ]

    try:
        batch_collated.update(default_collate(batch))
    except RuntimeError as e:
        # Diagnose collate shape mismatch: print each sample shape per key and its embodiment.
        print(f"[COLLATE-MISMATCH] default_collate failed: {e}", flush=True)
        keys = set()
        for s in batch:
            keys.update(s.keys())
        for k in sorted(keys):
            shapes = []
            for i, s in enumerate(batch):
                v = s.get(k)
                if hasattr(v, "shape"):
                    shapes.append((i, tuple(v.shape)))
                elif isinstance(v, (list, tuple)):
                    shapes.append((i, f"<{type(v).__name__} len={len(v)}>"))
            uniq = {sh for _, sh in shapes if isinstance(sh, tuple)}
            if len(uniq) > 1:
                ds = [
                    item.get("dataset_locator")
                    or item.get("embodiment")
                    or item.get("task")
                    or "?"
                    for item in batch
                ]
                print(f"[COLLATE-MISMATCH] key={k!r}", flush=True)
                for (i, sh), name in zip(shapes, ds):
                    print(f"  sample[{i}] shape={sh}  ds={name}", flush=True)
        raise
    
    return batch_collated


class RandomDropImage(nn.Module):
    def __init__(self, drop_prob: float = 0.3):
        super().__init__()
        self.p = drop_prob

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.p > 0 and torch.rand(1).item() < self.p:
            return torch.zeros_like(img)
        return img


def convert_text_to_paligemma(text, seq="xyxy"):
    # match [x1,y1,x2,y2] or (x1,y1,x2,y2) with optional spaces
    pattern = r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]"

    def convert_match(match):
        if seq == "xyxy":
            x1, y1, x2, y2 = [float(match.group(i)) for i in range(1, 5)]
            reordered_coords = [y1, x1, y2, x2]
        elif seq == "xywh":
            x, y, w, h = [float(match.group(i)) for i in range(1, 5)]
            reordered_coords = [y, x, y + h, x + w]
        else:
            raise ValueError(f"Unsupported seq: {seq}, only support 'xyxy' and 'xywh'.")

        token_strings = []
        for val in reordered_coords:
            val = max(0.0, min(1.0, val))
            quantized_val = int(val * 1024)
            quantized_val = min(quantized_val, 1023)
            token_strings.append(f"<loc{quantized_val:04d}>")
        return "".join(token_strings)

    new_text = re.sub(pattern, convert_match, text)

    return new_text


if __name__ == "__main__":
    # --- test sample ---
    raw_text = (
        "<image>\nWhat is inside the bounding box [(0.689, 0.758, 0.866, 1.000), [0.689, 0.758, 0.866, 1.000]]?"
        "The bounding box is given as [x1, y1, x2, y2], representing the top-left and bottom-right corners in normalized coordinates."
    )

    processed_text = convert_text_to_paligemma(raw_text)

    print("--- Original text ---")
    print(raw_text.strip())
    print("\n--- Converted text ---")
    print(processed_text.strip())

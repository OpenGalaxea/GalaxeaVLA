# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Authoritative batch contract for G05Policy.forward.

Data flow: Dataset.__getitem__ -> collate_fn_pad_sequences (utils/data_utils.py)
        -> batch dict -> G05Policy.forward (g05_policy.py)

These dataclasses are schema documentation and field references only. Runtime still
passes the raw dict produced by collate, without forced conversion. Treat this file
as the source of truth when changing collate or defining custom batches.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class G05TrainBatch:
    """Input for G05Policy.forward(batch, inference_mode=False)."""

    # RoboVQA-format samples containing conversations / proprio / action token info.
    # Length is B, consumed by InputPreprocessor.encode_train.
    samples: List[Dict[str, Any]]
    # Single camera: [B, n_img, 3, H, W]; multi-camera:
    # Dict[camera_key, [B, n_k, 3, H_k, W_k]].
    pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    # Normalized GT action, [B, H(horizon), D(action_dim)].
    action: torch.Tensor
    # [B, H] bool, True means this timestep is padding near episode end.
    action_is_pad: torch.Tensor
    # [B, D] bool, True means this dimension is padding from PaddingActionMerger.
    action_dim_is_pad: torch.Tensor
    # Per-sample source metadata (idx/task/embodiment/dataset_locator/frequency),
    # collected by collate_fn_pad_sequences for diagnostics and logging only.
    sample_meta: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class G05InferenceBatch:
    """Input for G05Policy.forward(batch, inference_mode=True).

    On return, batch is updated in place: batch["action"] receives prediction
    [B, H, D], plus extra fields such as CoT text from forward_inference results.
    """

    samples: List[Dict[str, Any]]
    pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    # Optional GT action, only for eval comparison with predictions.
    action: Optional[torch.Tensor] = None
    action_dim_is_pad: Optional[torch.Tensor] = None

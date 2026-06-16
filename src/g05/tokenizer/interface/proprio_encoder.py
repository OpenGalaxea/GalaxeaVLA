# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import re
from typing import Any, Optional, Tuple

import numpy as np
import torch


class ProprioEncoder:
    """Standalone proprio bin encoder independent of action_tokenizer."""

    def __init__(self, n_bins: int = 256, min_action: float = -1, max_action: float = 1):
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, n_bins + 1)[:-1]
        edges = np.linspace(min_action, max_action, n_bins + 1)
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0

    def encode(self, proprio: Any) -> str:
        """proprio tensor/ndarray/dict → "idx idx idx ..." string."""
        if isinstance(proprio, dict):
            pad_mask = proprio.get("proprio_dim_is_pad")
            if pad_mask is not None:
                proprio = proprio["value"][..., ~pad_mask]
            else:
                proprio = proprio["value"]
        proprio_np = torch.as_tensor(proprio).detach().cpu().numpy().astype(np.float32)
        discretized = np.digitize(proprio_np.flatten(), bins=self.bins) - 1
        return " ".join(map(str, discretized))

    def decode(self, text: str, out_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Reverse decode, needed by processor_inout.py backward."""
        parts = [p for p in re.split(r"\s+", text.strip()) if p]
        idx = torch.tensor([int(x) for x in parts], dtype=torch.long)
        idx = torch.clamp(idx, 0, self.n_bins - 1)
        vals = torch.tensor(self.bin_centers, dtype=torch.float32)[idx]
        if out_shape is not None:
            vals = vals.reshape(out_shape)
        return vals

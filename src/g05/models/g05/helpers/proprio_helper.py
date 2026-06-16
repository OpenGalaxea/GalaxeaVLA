# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Proprio MLP encoder and related utilities.

- `ProprioEmbedder`: nn.Module, maps proprio tensor -> hidden embedding for scatter
  into `<state>` token positions, mirroring the image token replacement mechanism.
- `build_proprio_batch`: extracts proprio from samples, stacks to [B, T, D], and
  zeroes padded dimensions because the MLP input dimension must be fixed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ProprioEmbedder(nn.Module):
    """Proprio → hidden embedding (MLP)。

    Structure: Linear -> GELU -> LayerNorm (pre-norm) -> Linear (zero-init).
    DiT-style zero-init sets the final Linear weight/bias to zero, so initial output
    is all zeros. This protects the pretrained VLM and gradually introduces proprio
    signal during training. LayerNorm sits before the zero Linear to avoid the
    gradient-fragile anti-pattern of "LayerNorm(zero)".

    Parameters are forced to float32; precision management is encapsulated in
    forward. The caller is responsible for casting output back to the downstream dtype.

    `state_token_id` is unknown at construction time because tokenizer add_tokens
    decides it dynamically. `G05Policy` fills it after InputPreprocessor registers
    the `<state>` token.
    """

    def __init__(self, proprio_dim: int, hidden_size: int) -> None:
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(int(proprio_dim), int(hidden_size)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(hidden_size)),
        )
        nn.init.zeros_(mlp[3].weight)
        nn.init.zeros_(mlp[3].bias)
        # Force fp32 so Mixture/Model .to(bf16) does not downcast this module.
        self.mlp = mlp.float()
        self.state_token_id: Optional[int] = None

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio [B, T, D_in] in any dtype -> [B, T, hidden_size] fp32.

        Caller (G05Model._forward_embed) casts back to final_embedding dtype.
        """
        with torch.autocast(proprio.device.type, enabled=False):
            return self.mlp(proprio.float())


def build_proprio_batch(
    samples: List[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    zero_values: bool = False,
) -> Optional[torch.Tensor]:
    """Build a batched proprio tensor for `ProprioEmbedder`.

    Precondition: caller has confirmed mlp mode is enabled (`proprio_embedder is not None`)
    and the current batch is not VLM-only. This function only stacks and zero-pads;
    it does not perform a mode gate.

    Returns:
        [B, T, D] float tensor; returns None if samples have no proprio field, allowing
        missing-field tolerance.
    """
    if not samples or any("proprio" not in s for s in samples):
        return None

    values = []
    pads = []
    for s in samples:
        p = s["proprio"]
        v = p["value"] if isinstance(p, dict) else p
        pad = p.get("proprio_dim_is_pad") if isinstance(p, dict) else None
        v = torch.as_tensor(v)
        if v.ndim == 1:
            v = v.unsqueeze(0)  # [D] → [1, D]
        values.append(v)
        if pad is not None:
            pads.append(torch.as_tensor(pad, dtype=torch.bool))
        else:
            pads.append(torch.zeros(v.shape[-1], dtype=torch.bool))

    proprio_batch = torch.stack(values, dim=0).to(device=device, dtype=dtype)  # [B, T, D]
    pad_batch = torch.stack(pads, dim=0).to(device=device)  # [B, D]
    # Zero-out padded dims so MLP sees uniform zero padding instead of stale values.
    proprio_batch = proprio_batch.masked_fill(pad_batch[:, None, :], 0.0)
    if zero_values:
        proprio_batch = torch.zeros_like(proprio_batch)
    return proprio_batch

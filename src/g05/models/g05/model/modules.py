# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0 AND Apache-2.0
# Copyright (c) 2026 Galaxea
# Copyright 2024 Big Vision Authors.
# Portions are adapted from OpenPI's Big Vision-based Gemma implementation.
# See THIRD_PARTY_NOTICES.md for the upstream Apache-2.0 notice.

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPosEmbPi0(nn.Module):
    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        if dim % 2 != 0:
            raise ValueError(f"dimension ({dim}) must be divisible by 2")

    @torch.autocast("cuda", enabled=False)  # Critical!!!
    def forward(
        self,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if t.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

        # Use float64 for high-precision frequency calculation
        dtype = get_safe_dtype(torch.float64, t.device.type)
        fraction = torch.linspace(0.0, 1.0, self.dim // 2, dtype=dtype, device=t.device)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * t[:, None]
        emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

        return emb.to(t.dtype)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # Match openpi implementation:
        # one Linear layer named 'dense' with output 3 * dim.
        self.dense = nn.Linear(dim_cond, dim * 3, bias=True)

        # Initialization logic matching the openpi/jax conversion code.
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    @torch.autocast("cuda", enabled=False)  # Critical!!!
    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # [Batch, Seq, Dim]
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        # [Batch, Dim_Cond] -> [Batch, 3 * Dim]
        # Force fp32: FSDP mp may have bf16 weights but we need fp32 precision here
        modulation = torch.nn.functional.linear(
            cond.float(),
            self.dense.weight.float(),
            self.dense.bias.float() if self.dense.bias is not None else None,
        )

        # Reshape modulation to match x: [Batch, 1, 3 * Dim]
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)

        # Split into scale, shift, gate
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        # Apply modulation: (1 + scale) * x + shift
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

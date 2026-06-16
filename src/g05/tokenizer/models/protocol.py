# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# src/g05/tokenizer/models/protocol.py
"""Codec backend protocol and metadata for the tokenizer interface layer.

All tokenizer backends must satisfy CodecBackendProtocol.
VQActionTokenizer interacts exclusively through this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class CodecMetadata:
    """Optional per-call metadata for encode/decode.

    Each backend reads only the fields it supports; unused fields are ignored.
    Replaces the ad-hoc **kwargs pattern previously used by each backend.
    """

    frequency: Optional[torch.Tensor] = None
    """Control frequency in Hz, shape (B,)."""

    embodiment: Optional[List[str]] = None
    """Embodiment name list, length B. Mutually exclusive with embodiment_ids."""

    embodiment_ids: Optional[torch.Tensor] = None
    """Pre-resolved embodiment indices, shape (B,). Takes priority over `embodiment`."""

    action_dim_is_pad: Optional[torch.Tensor] = None
    """Dimension padding mask, shape (B, D). True = padded dimension."""

    action_is_pad: Optional[torch.Tensor] = None
    """Temporal padding mask, shape (B, T). True = padded time step."""

    parts_meta: Optional[Dict[str, int]] = None
    """Active parts for this call, e.g. {"left_arm": 7, "right_arm": 7, "left_gripper": 1}."""

    target_time_steps: Optional[int] = None
    """Desired output time steps for decode (for upsampling/downsampling)."""


@dataclass
class DecodeMetadata:
    """Backend self-description for VQActionTokenizer decode decisions.

    VQActionTokenizer reads only this dataclass — it never probes backend
    internals directly.  Each backend overrides get_decode_metadata() to
    describe its own layout; BaseCodecWrapper provides a sensible default.
    """

    expected_lengths: List[int]
    """Valid decode token lengths.  Flat backend: [total_tokens].
    Rule-based backend: [nn_only_len, nn_plus_rule_len].
    WBC backend: one length per possible num_residuals."""

    total_residuals: int
    """Total RVQ codebook count.  0 if not applicable."""

    canonical_parts_meta: Dict[str, int]
    """Single source of truth: {key_name: dim}.
    Checkpoint > config priority resolved by backend at load time."""

    nn_key_names: List[str]
    """NN-path key names in serialization order."""

    rule_key_names: List[str]
    """Rule-based-path key names."""

    code_len: int = 0
    """Tokens per NN key per level."""

    rule_tokens_per_key: int = 0
    """Tokens per rule key (flat, not per level)."""

    nn_chunked_key_count: int = 0
    """Number of NN keys after chunk-expansion (>= len(nn_key_names)).
    Keys whose dim > max_component_dim are split into multiple chunks at runtime;
    this count reflects the actual serialized key count used for action_token_len."""


def _validate_backend(backend: object) -> None:
    """Validate that a backend implements CodecBackendProtocol.

    Called once during VQActionTokenizer.__init__ so violations are caught
    at construction time with an actionable error message.

    Raises:
        TypeError: if any required attribute or method is missing, or if
                   code_parts is not a non-empty dict.
    """
    required_attrs = ["code_parts", "horizon", "action_dim", "vocab_size"]
    required_methods = ["encode", "decode", "configure_eval"]
    missing = [a for a in required_attrs + required_methods if not hasattr(backend, a)]
    if missing:
        raise TypeError(
            f"{type(backend).__name__} does not implement CodecBackendProtocol. "
            f"Missing: {missing}. "
            f"See docs/tokenizer_backend_guide.md for the 5-step onboarding guide."
        )
    cp = backend.code_parts
    if not isinstance(cp, dict) or len(cp) == 0:
        raise TypeError(
            f"{type(backend).__name__}.code_parts must be a non-empty dict, got: {cp!r}. "
            f"Flat backends return {{'codes': N}}; WBC backends return {{part_name: N, ...}}."
        )

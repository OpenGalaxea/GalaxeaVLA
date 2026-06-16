# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# src/g05/tokenizer/models/base_wrapper.py
"""BaseCodecWrapper — minimal base class for all tokenizer codec backends.

Provides:
- Eval-mode locking (configure_eval / configure_train_eval).
- Legacy configure_train_eval() compat alias.
- _read_parts_meta() classmethod for resolving legacy config key names.

Does NOT include loss infrastructure — each backend keeps its own vqvae_update.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class BaseCodecWrapper(nn.Module):
    """Base class for all tokenizer codec backends.

    Subclasses must implement CodecBackendProtocol (code_parts, horizon,
    action_dim, vocab_size, encode, decode). This base class does not enforce
    the protocol at the class level — enforcement happens at runtime via
    _validate_backend() in VQActionTokenizer.__init__.

    Eval-mode locking
    -----------------
    When used as a frozen tokenizer inside a VLA model, the backend must stay
    in eval mode even when the outer model calls model.train(). After
    configure_eval() is called, train() calls are silently ignored.

    This fixes the eval-mode bug where self.autoencoder.eval() was called
    instead of self.eval(), leaving the wrapper's own training flag True and
    enabling stochastic operations (dropout, random masking) during inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self._frozen_for_inference: bool = False

    def configure_eval(self) -> None:
        """Lock this backend into inference mode.

        After this call, subsequent train() invocations are silently ignored.
        Safe to call multiple times.
        """
        self.eval()  # sets self.training = False on self AND all submodules
        self._frozen_for_inference = True
        logger.debug("%s locked into inference mode.", type(self).__name__)

    def configure_train_eval(self, eval: bool = True, **kwargs) -> None:
        """Legacy compat alias for configure_eval().

        Old callers use configure_train_eval(eval=True). New code should call
        configure_eval() directly.
        """
        if eval:
            self.configure_eval()
        else:
            self._frozen_for_inference = False
            self.train()

    def train(self, mode: bool = True):
        """Override to silently ignore train(True) after configure_eval()."""
        if self._frozen_for_inference and mode:
            return self  # stay in eval — do not propagate train() to submodules
        return super().train(mode)

    @classmethod
    def _read_parts_meta(cls, cfg) -> Dict[str, int]:
        """Read parts_meta from cfg, supporting legacy key names.

        Priority: parts_meta > max_action_shape_meta > component_dims.
        Returns empty dict if none found.
        """
        for key in ["parts_meta", "max_action_shape_meta", "component_dims"]:
            val = cfg.get(key, None) if hasattr(cfg, "get") else None
            if val is not None:
                if isinstance(val, DictConfig):
                    return OmegaConf.to_container(val, resolve=True)
                if isinstance(val, dict):
                    return val
        return {}

    def get_decode_metadata(self):
        from .protocol import DecodeMetadata

        cp = self.code_parts
        total = sum(cp.values())
        return DecodeMetadata(
            expected_lengths=[total],
            total_residuals=0,
            canonical_parts_meta=dict(cp),
            nn_key_names=list(cp.keys()),
            rule_key_names=[],
            code_len=0,
            rule_tokens_per_key=0,
        )

    def serialize_codes(self, codes_dict, num_residuals=None):
        if "codes" in codes_dict:
            return codes_dict["codes"]
        raise NotImplementedError(
            f"{type(self).__name__}: codes_dict has no 'codes' key and "
            f"serialize_codes() is not overridden."
        )

    def deserialize_codes(self, flat_tensor, num_residuals=None):
        return {"codes": flat_tensor}

    def kv_to_flat(self, kv_dict, key_order=None):
        if key_order is None:
            key_order = list(kv_dict.keys())
        return torch.cat([kv_dict[k] for k in key_order if k in kv_dict], dim=-1)

    def flat_to_kv(self, flat_tensor, parts_meta=None):
        pm = parts_meta or self.code_parts
        dims = list(pm.values())
        total = sum(dims)
        splits = torch.split(flat_tensor[..., :total], dims, dim=-1)
        return dict(zip(pm.keys(), splits))

    @staticmethod
    def validate_and_pad_kv(input_kv, canonical_parts_meta):
        if canonical_parts_meta is None:
            return input_kv

        unknown = set(input_kv.keys()) - set(canonical_parts_meta.keys())
        if unknown:
            raise ValueError(
                f"Input keys {sorted(unknown)} not in canonical parts_meta "
                f"{list(canonical_parts_meta.keys())}."
            )

        if set(input_kv.keys()) == set(canonical_parts_meta.keys()):
            return input_kv

        sample = next(iter(input_kv.values()))
        batch_shape = sample.shape[:-1]
        device, dtype = sample.device, sample.dtype
        padded = {}
        for key, canonical_dim in canonical_parts_meta.items():
            if key in input_kv:
                t = input_kv[key]
                if t.shape[-1] < canonical_dim:
                    pad = torch.zeros(
                        *batch_shape, canonical_dim - t.shape[-1], dtype=dtype, device=device
                    )
                    padded[key] = torch.cat([t, pad], dim=-1)
                else:
                    padded[key] = t
            else:
                padded[key] = torch.zeros(*batch_shape, canonical_dim, dtype=dtype, device=device)
        return padded

    def is_rule_based_key(self, key: str) -> bool:
        return False

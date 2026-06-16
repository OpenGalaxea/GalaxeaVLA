# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import logging
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Sequence, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf


ActionIdxBatch = List[List[int]]
TokenIds = Union[List[int], np.ndarray, torch.Tensor]


class DecodeResult(NamedTuple):
    action: torch.Tensor
    absent_keys: set[str]


class BaseActionTokenizer(ABC):
    """
    Base action tokenizer: maps discrete token indices produced by the action
    discretizer (action_tokenizer) into the main LLM/VLM tokenizer token-id space,
    and provides a unified encode/decode interface.

    Subclasses only need to implement:
    - `_encode_action_indices(...)`: continuous actions -> discrete action indices (batch)
    - `_decode_action_indices(...)`: discrete action indices (batch) -> continuous actions

    Note: this class no longer owns the HF tokenizer. All token registration is done
    by the outer TokenRegistry.
    """

    def __init__(
        self,
        *,
        vq_config: DictConfig | None = None,
        cache_dir: str | None = None,
        vocab_offset: int = 0,
        hf_vocab_size: int | None = None,
    ) -> None:
        self._vocab_offset = vocab_offset
        self._hf_vocab_size = hf_vocab_size
        self.vq_config = vq_config if vq_config is not None else OmegaConf.create()
        OmegaConf.set_struct(self.vq_config, False)
        self.cache_dir = cache_dir

        self.use_extra_tokens = bool(self.vq_config.get("use_extra_tokens", False))
        self.relaxed_decoding = bool(self.vq_config.get("relaxed_decoding", False))
        self.zero_action = bool(self.vq_config.get("zero_action", False))

        self._new_action_tokens: list[str] = []

        if not hasattr(self, "action_tokenizer"):
            raise ValueError("Subclasses must set self.action_tokenizer before super().__init__")
        if not hasattr(self, "_action_vocab_size"):
            raise ValueError("Subclasses must set self._action_vocab_size before super().__init__")
        if not hasattr(self, "_codebook_size"):
            self._codebook_size = self._action_vocab_size

        self._init_token_configuration(self.vq_config)
        self.action_template: str = self.vq_config.get("action_template", "Action: {action}|")
        self.block_wise_autoregressive = bool(
            self.vq_config.get("block_wise_autoregressive", False)
        )
        if self.block_wise_autoregressive:
            self._init_block_wise_configuration(self.vq_config)

    @property
    def new_action_tokens(self) -> list[str]:
        """All token names that must be registered to the HF tokenizer, consumed by TokenRegistry."""
        return list(self._new_action_tokens)

    @property
    def bar_config(self) -> dict | None:
        """BAR parameters used by BARBuilder; non-None only in BAR mode."""
        if not self.block_wise_autoregressive:
            return None
        return {
            "block_size": self.block_size,
            "bos_blk_id": self._bos_blk_id,
            "eos_blk_id": self._eos_blk_id,
            "pad_action_id": self.pad_token_id,
            "ar_block_num": self.ar_block_num,
            "action_template": self.action_template,
            "num_block": self.num_block,
        }

    def set_bar_ids(self, bos_blk_id: int, eos_blk_id: int, pad_action_id: int):
        """Fill real IDs after TokenRegistry registers BAR tokens."""
        self._bos_blk_id = bos_blk_id
        self._eos_blk_id = eos_blk_id
        self.pad_token_id = pad_action_id

    def action_indices_to_token_ids(self, action_indices: ActionIdxBatch) -> ActionIdxBatch:
        """action index(batch) -> tokenizer token ids(batch)"""
        return [[int(self._action_idx_to_tokenizer_idx(x)) for x in row] for row in action_indices]

    def to(self, device_id):
        """Move nn.Module attributes to device and call `.to(...)` on action_tokenizer if possible."""
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(attr, torch.nn.Module):
                setattr(self, attr_name, attr.to(device_id))
        self.device = torch.device(device_id)
        if hasattr(self, "action_tokenizer") and hasattr(self.action_tokenizer, "to"):
            try:
                self.action_tokenizer.to(device_id)
            except Exception:
                pass

    def __call__(
        self,
        action: torch.Tensor,
        *,
        encode_kwargs: dict | None = None,
    ):
        """
        Args:
            action: [batch, time_horizon, action_dim] (torch.Tensor)
        Returns:
            List[List[int]] — discrete action token ids in HF tokenizer space
        """
        discretized_action_idx = self._encode_action_indices(action, encode_kwargs=encode_kwargs)
        return self.action_indices_to_token_ids(discretized_action_idx)

    def decode_token_ids_to_actions(
        self,
        action_token_ids: TokenIds,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> torch.Tensor | DecodeResult:
        """
        Decode tokenizer token ids -> continuous actions.
        action_token_ids: 1D sequence of tokenizer token ids
        return: torch.Tensor [time_horizon, action_dim] or model-specific shape
        """
        if isinstance(action_token_ids, torch.Tensor):
            token_ids = action_token_ids.detach().cpu()
        else:
            token_ids = torch.tensor(action_token_ids, dtype=torch.long).cpu()

        action_idx = self._tokenizer_idx_to_action_idx(token_ids)
        action_idx = torch.clamp(action_idx, 0, int(self._codebook_size) - 1)
        assert action_idx.ndim == 1, f"action_idx must be a 1D tensor, but got {action_idx.shape}"
        actions = self._decode_action_indices(
            [action_idx.tolist()],
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return torch.from_numpy(actions)[0]

    def decode_token_ids_to_actions_batch(
        self,
        action_token_ids: Sequence[TokenIds],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Batch decode tokenizer token ids -> continuous actions."""
        batch_action_idx: ActionIdxBatch = []
        for a in action_token_ids:
            if isinstance(a, torch.Tensor):
                t = a.detach().cpu()
            else:
                t = torch.tensor(a, dtype=torch.long).cpu()
            idx = self._tokenizer_idx_to_action_idx(t)
            idx = torch.clamp(idx, 0, int(self._codebook_size) - 1)
            batch_action_idx.append(idx.tolist())

        actions = self._decode_action_indices(
            batch_action_idx,
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return torch.from_numpy(actions)

    def _action_idx_to_tokenizer_idx(self, action_idx: int | np.ndarray) -> int | np.ndarray:
        if self.use_extra_tokens:
            return self.action_token_begin_idx + action_idx
        return self._hf_vocab_size - 1 - self._fast_skip_tokens - action_idx

    def _tokenizer_idx_to_action_idx(
        self, tokenizer_idx: torch.Tensor | np.ndarray | List[int]
    ) -> torch.Tensor | np.ndarray:
        if isinstance(tokenizer_idx, list):
            tokenizer_idx = np.array(tokenizer_idx)
        if self.use_extra_tokens:
            return tokenizer_idx - self.action_token_begin_idx
        return self._hf_vocab_size - 1 - self._fast_skip_tokens - tokenizer_idx

    @property
    def vocab_size(self) -> int:
        """Action vocab size."""
        return int(self._action_vocab_size)

    @property
    def token_config(self) -> dict:
        """Expose all action-related token IDs for consumers such as G05Policy."""
        cfg = {
            "action_token_begin_idx": self.action_token_begin_idx,
            "action_token_end_idx": self.action_token_end_idx,
        }
        if self.block_wise_autoregressive:
            cfg["bos_blk_id"] = self._bos_blk_id
            cfg["eos_blk_id"] = self._eos_blk_id
            cfg["block_size"] = self.block_size
        return cfg

    def _init_token_configuration(self, vq_config: dict):
        if self.use_extra_tokens:
            action_tokens = [f"<action{i:0>4}>" for i in range(int(self._codebook_size))]
            self.action_token_begin_idx = self._vocab_offset
            self.action_token_end_idx = self.action_token_begin_idx + len(action_tokens)
            self.action_tokens = action_tokens
            self._new_action_tokens = list(action_tokens)
        else:
            self._fast_skip_tokens = int(vq_config.get("fast_skip_tokens", 128))
            self.action_token_begin_idx = (
                self._hf_vocab_size - 1 - self._fast_skip_tokens - int(self._codebook_size)
            )
            self.action_token_end_idx = self.action_token_begin_idx + int(self._codebook_size)
            self._new_action_tokens = []

        logger.info(
            f"action_token_begin_idx: {self.action_token_begin_idx}, action_token_end_idx: {self.action_token_end_idx}"
        )

    def _init_block_wise_configuration(self, vq_config: dict):
        """Configure block-wise autoregressive (BAR) tokens and parameters.

        Appends BAR special tokens (<bos_blk>, <eos_blk>, <pad_action_token>) to
        the token registry, and sets block_size / num_block / ar_block_num.

        IMPORTANT: This method uses an incremental pattern for action_token_len.
        If action_token_len was already set (e.g. by _init_grouped_tokens in a
        subclass), it is preserved.  Otherwise it defaults to None (to be set
        by the subclass after super().__init__).
        """
        bar_token_strs = ["<bos_blk>", "<eos_blk>", "<pad_action_token>"]
        self.action_token_end_idx = self.action_token_end_idx + len(bar_token_strs)
        self._new_action_tokens.extend(bar_token_strs)
        self._bos_blk_id = None
        self._eos_blk_id = None
        self.pad_token_id = None

        self.ar_block_num = int(vq_config.get("ar_block_num", 0) or 0)
        self.block_size = vq_config.get("block_size", None)
        assert self.block_size is not None, "block_wise_autoregressive=True requires block_size"
        self.num_block = vq_config.get("num_block", None)
        if not hasattr(self, "action_token_len") or self.action_token_len is None:
            self.action_token_len = None

    @abstractmethod
    def _encode_action_indices(
        self, action: torch.Tensor, encode_kwargs: dict | None = None
    ) -> ActionIdxBatch:
        """Continuous actions -> discrete action indices (batch)."""

    @abstractmethod
    def _decode_action_indices(
        self,
        action_indices: ActionIdxBatch,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, list[set[str]]]:
        """Discrete action indices (batch) -> continuous actions (np.array)."""

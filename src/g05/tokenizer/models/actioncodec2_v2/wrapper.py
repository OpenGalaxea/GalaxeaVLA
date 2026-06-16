# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""ActionCodecV2Wrapper — lightweight wrapper around ActionCodecV2Model.

Responsibilities:
- Classify input keys into NN path vs rule-based path
- Pad/chunk dimensions to match model's max_component_dim
- Encode: flat tensor or kv dict → codes dict (always contains "codes" key)
- Decode: codes dict → flat tensor or kv dict (mirror symmetry with encode input)

Training integration:
- ``forward(batch)`` returns ``(loss, log_dict)`` when input contains ``"action"`` key
- ``forward(component_dict)`` returns ``(recon_dict, codes_dict, loss_dict)`` for
  stand-alone autoencoder use

Note on normalization: tokenizer-level z-score normalization (set_normalizer / normalize /
denormalize) was removed intentionally. All normalization is handled by the processor layer
via LinearNormalizer. All existing checkpoints used processor-level normalization only.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


from ..base_wrapper import BaseCodecWrapper
from ..binary_sequence import ConstrainedSequenceTokenizer
from .configuration_actioncodec2v2 import ActionCodecV2Config
from .consistency import ConsistencyAugmenter
from .modeling_actioncodec2v2 import ActionCodecV2Model
from .partitioner import ActionPartitioner, PartitionState
from ...utils.parts_meta_utils import compute_grouped_keys

logger = logging.getLogger(__name__)


def _resolve_to_dict(cfg) -> dict:
    """Resolve OmegaConf or dict to plain dict. Raises TypeError on invalid input."""
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    raise TypeError(f"cfg must be dict or OmegaConf, got {type(cfg)}")


def _apply_deprecated_key_compat(cfg: dict) -> dict:
    """Translate deprecated config keys to current names, warning on each occurrence.

    Deprecated → current:
      rule_based_tokenizer_for_gripper (bool) → rule_based_key_patterns (list)
      gripper_min_block_len                   → rule_based_min_block_len
      gripper_binarize_threshold              → rule_based_binarize_threshold
    """
    overrides: dict = {}
    if "rule_based_key_patterns" not in cfg and cfg.get("rule_based_tokenizer_for_gripper"):
        logger.warning(
            "Deprecated config key 'rule_based_tokenizer_for_gripper'; "
            "use 'rule_based_key_patterns: [\"gripper\"]' instead."
        )
        overrides["rule_based_key_patterns"] = ["gripper"]
    if "rule_based_min_block_len" not in cfg and "gripper_min_block_len" in cfg:
        logger.warning(
            "Deprecated config key 'gripper_min_block_len'; use 'rule_based_min_block_len' instead."
        )
        overrides["rule_based_min_block_len"] = cfg["gripper_min_block_len"]
    if "rule_based_binarize_threshold" not in cfg and "gripper_binarize_threshold" in cfg:
        logger.warning(
            "Deprecated config key 'gripper_binarize_threshold'; "
            "use 'rule_based_binarize_threshold' instead."
        )
        overrides["rule_based_binarize_threshold"] = cfg["gripper_binarize_threshold"]
    return {**cfg, **overrides} if overrides else cfg


class ActionCodecV2Wrapper(BaseCodecWrapper):
    """Wrapper around ActionCodecV2Model.

    Delegates key routing and chunking to ActionPartitioner.
    Mirror symmetry: KV input → KV output, flat input → flat output.

    Construction:
        cfg: dict — plain dict or OmegaConf DictConfig. Required field:
            model_arch: dict — passed to ActionCodecV2Config(**model_arch)
        Optional fields (with defaults):
            key_dims: dict            — {key: dim} per action component, default {}
            parts_meta: dict          — legacy alias for key_dims, default {}
            merge_spec: dict          — merge specification, default {}
            rule_based_key_patterns: list — substrings for rule-based routing, default ["gripper"]
            rule_based_min_block_len: int — default 1
            rule_based_binarize_threshold: float — default 0.0
            num_residuals: int | None — default None (use model's n_codebooks)
            eval: bool               — default True
            device: str              — default "cpu"
            ckpt_dir: str | None     — default None
    """

    def __init__(self, cfg):
        cfg = _resolve_to_dict(cfg)
        super().__init__()
        self._init_from_cfg(cfg)

    def _init_from_cfg(self, cfg: dict):
        cfg = _apply_deprecated_key_compat(cfg)
        model, key_dims, model_config = self._build_model(cfg)

        self.model = model
        self.key_dims = key_dims  # {key: dim} — single source of truth for key layout
        self._model_arch_cfg = model.config
        self._augmenter: Optional[ConsistencyAugmenter] = None
        self._last_encode_state: Optional[PartitionState] = None  # preserved for decode roundtrip
        self._user_key_dims = bool(key_dims)  # True if user explicitly set parts_meta/key_dims

        self._partitioner = self._build_partitioner(cfg, key_dims, model)
        self._rule_tokenizer = self._build_rule_tokenizer(cfg, model)
        self._default_num_residuals: Optional[int] = cfg.get("num_residuals")

        self._log_init_summary(model)

        if cfg.get("eval", True) and cfg.get("ckpt_dir"):
            self.load_model(cfg["ckpt_dir"])

    def _build_model(self, cfg: dict):
        model_arch = cfg["model_arch"]
        config = ActionCodecV2Config(**model_arch)
        model = ActionCodecV2Model(config)

        if not cfg.get("eval", True):
            model.train()
        model = model.to(cfg.get("device", "cpu"))

        # Resolve key_dims from config — three ways, in priority order:
        # 1. Direct "key_dims" (already merged)
        # 2. "parts_meta" + "merge_spec" → compute_grouped_keys merges them
        # 3. "parts_meta" alone (no merging)
        if "key_dims" in cfg:
            key_dims = dict(cfg["key_dims"])
        elif cfg.get("merge_spec") and cfg.get("parts_meta"):
            key_dims = compute_grouped_keys(cfg["parts_meta"], cfg["merge_spec"])
        else:
            key_dims = dict(cfg.get("parts_meta", {}))

        # OmegaConf deep-merge preserves parent dict keys; child configs null them
        # out to remove unwanted keys (e.g. left_control for single-arm).
        # Filter out None-valued entries so downstream code sees only real dims.
        key_dims = {k: v for k, v in key_dims.items() if v is not None}

        return model, key_dims, config

    def _build_partitioner(self, cfg: dict, key_dims: Dict[str, int], model):
        rule_key_patterns = cfg.get("rule_based_key_patterns", ["gripper"])
        return ActionPartitioner(
            key_dims=key_dims,
            max_component_dim=model.config.max_component_dim,
            rule_based_key_patterns=rule_key_patterns,
            merge_preprocessor=None,
        )

    def _build_rule_tokenizer(self, cfg: dict, model):
        rule_key_patterns = cfg.get("rule_based_key_patterns", ["gripper"])
        if not rule_key_patterns or model is None:
            return None
        return ConstrainedSequenceTokenizer(
            seq_len=model.config.horizon,
            min_block_len=cfg.get("rule_based_min_block_len", 1),
            vocab_size=model.config.vocab_size_with_specials,
            binarize_threshold=cfg.get("rule_based_binarize_threshold", 0.0),
        )

    def _log_init_summary(self, model):
        if not hasattr(model, "_init_summary_lines"):
            if self._rule_tokenizer is not None:
                logger.info(f"rule-based tokenizer: {repr(self._rule_tokenizer)}")
            return
        lines = list(model._init_summary_lines)
        if self._rule_tokenizer is not None:
            closing = lines.pop()
            W = 58
            lines.append(f"╠{'═' * W}╣")
            rule_repr = repr(self._rule_tokenizer)
            for i, rule_line in enumerate(rule_repr.split("\n")):
                if i == 0:
                    rule_str = f"  rule-based    {rule_line}"
                else:
                    rule_str = f"  │            {rule_line}"
                lines.append(f"║{rule_str:<{W}s}║")
            lines.append(closing)
        logger.info("\n" + "\n".join(lines))

    def is_rule_based_key(self, key: str) -> bool:
        return self._partitioner.is_rule_based_key(key)

    @classmethod
    def from_pretrained(cls, model_path: str, key_dims=None, **kwargs):
        model = ActionCodecV2Model.from_pretrained(model_path, **kwargs)
        cfg = {
            "model_arch": model.config.to_dict(),
            "parts_meta": key_dims or {},
        }
        return cls(cfg)

    @classmethod
    def from_config(cls, config: ActionCodecV2Config, key_dims=None):
        cfg = {
            "model_arch": config.to_dict(),
            "parts_meta": key_dims or {},
        }
        return cls(cfg)

    # ── VQActionTokenizer interface properties ─────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size_with_specials

    @property
    def horizon(self) -> int:
        return self.model.config.horizon

    @property
    def action_dim(self) -> int:
        return sum(self.key_dims.values()) if self.key_dims else self.model.config.max_component_dim

    @property
    def tokens_per_key(self) -> int:
        return self.model.config.tokens_per_key

    # ── Cached key classification (computed once, used by multiple methods) ──────

    @property
    def _nn_keys(self) -> List[str]:
        """NN-path keys in key_dims order."""
        if not hasattr(self, "_nn_keys_cached"):
            self._nn_keys_cached = [k for k in self.key_dims if not self.is_rule_based_key(k)]
        return self._nn_keys_cached

    @property
    def _rule_keys(self) -> List[str]:
        """Rule-based keys in key_dims order."""
        if not hasattr(self, "_rule_keys_cached"):
            self._rule_keys_cached = [k for k in self.key_dims if self.is_rule_based_key(k)]
        return self._rule_keys_cached

    @property
    def code_parts(self) -> Dict[str, int]:
        nr = self._default_num_residuals or self.model.config.n_codebooks
        code_len = self.model.config.code_h * self.model.config.code_a
        if self.key_dims and self._rule_tokenizer is not None:
            nn_key_count = sum(
                max(
                    1,
                    (d + self._partitioner.max_component_dim - 1)
                    // self._partitioner.max_component_dim,
                )
                if d > self._partitioner.max_component_dim
                else 1
                for k, d in self.key_dims.items()
                if not self.is_rule_based_key(k)
            )
            total = (
                nn_key_count * nr * code_len
                + len(self._rule_keys) * self._rule_tokenizer.num_tokens
            )
        elif self.key_dims:
            nn_key_count = sum(
                max(
                    1,
                    (d + self._partitioner.max_component_dim - 1)
                    // self._partitioner.max_component_dim,
                )
                if d > self._partitioner.max_component_dim
                else 1
                for k, d in self.key_dims.items()
                if not self.is_rule_based_key(k)
            )
            total = nn_key_count * nr * code_len
        else:
            total = nr * code_len
        return {"codes": total}

    @property
    def action_token_len(self) -> int:
        return sum(self.code_parts.values())

    # ── Protocol method overrides ──────────────────────────────────────────────

    def get_decode_metadata(self):
        from ..protocol import DecodeMetadata

        has_rule = self._rule_tokenizer is not None
        nn_key_names = self._nn_keys
        rule_key_names = self._rule_keys

        total_n_cb = self.model.config.n_codebooks
        code_len = self.model.config.code_h * self.model.config.code_a
        rule_tokens_per_key = self._rule_tokenizer.num_tokens if has_rule else 0

        # Count chunk-expanded NN keys (keys with dim > max_component_dim are split at runtime)
        mc = self._partitioner.max_component_dim
        nn_chunked_key_count = sum(
            math.ceil(self.key_dims[k] / mc) if self.key_dims[k] > mc else 1 for k in nn_key_names
        )

        # Each entry corresponds to using [1..total_n_cb] residual codebooks
        expected_lengths = [
            nn_chunked_key_count * nr * code_len + len(rule_key_names) * rule_tokens_per_key
            for nr in range(1, total_n_cb + 1)
        ]

        return DecodeMetadata(
            expected_lengths=expected_lengths,
            total_residuals=total_n_cb,
            canonical_parts_meta=dict(self.key_dims),
            nn_key_names=nn_key_names,
            rule_key_names=rule_key_names,
            code_len=code_len,
            rule_tokens_per_key=rule_tokens_per_key,
            nn_chunked_key_count=nn_chunked_key_count,
        )

    def serialize_codes(self, codes_dict, num_residuals=None):
        """Serialize per-key codes dict → flat (B, total_nn_tokens) tensor.

        Level-first ordering: [key0_cb0, key0_cb1, …, key1_cb0, key1_cb1, …]
        Each per-key code has shape (B, n_cb, code_len).
        """
        key_names = list(codes_dict.keys())
        n_cb = num_residuals if num_residuals is not None else self.model.config.n_codebooks
        if num_residuals is not None:
            _max_cb = self.model.config.n_codebooks
            if not (1 <= num_residuals <= _max_cb):
                raise ValueError(f"num_residuals={num_residuals} must be in [1, {_max_cb}]")

        stacked = torch.stack([codes_dict[k] for k in key_names])  # (N, B, n_cb, code_len)
        stacked = stacked[:, :, :n_cb]
        # Permute to (B, n_cb, N, code_len) then flatten the last 3 dims
        return stacked.permute(1, 2, 0, 3).reshape(stacked.shape[1], -1).long()

    def deserialize_codes(self, flat_codes, key_names, num_residuals=None):
        """Deserialize flat (B, N*n_cb*code_len) tensor → per-key codes dict.

        Inverse of serialize_codes.
        """
        B = flat_codes.shape[0]
        n_cb = num_residuals if num_residuals is not None else self.model.config.n_codebooks
        code_len = self.model.config.code_h * self.model.config.code_a
        N = len(key_names)

        reshaped = flat_codes[:, : N * n_cb * code_len].reshape(B, n_cb, N, code_len)
        tensors = reshaped.permute(2, 0, 1, 3)  # (N, B, n_cb, code_len)
        return dict(zip(key_names, tensors.unbind(0)))

    def deserialize_flat_codes(self, flat_tensor, num_residuals=None):
        """Protocol-compatible deserialize: flat codes tensor → codes_dict (NN + rule keys)."""
        nn_codes, rule_flat = self._deserialize_flat_codes(
            flat_tensor, self._last_encode_state, num_residuals
        )
        if rule_flat is not None:
            num_tok = self._rule_tokenizer.num_tokens
            for i, rk in enumerate(self._rule_keys):
                nn_codes[rk] = rule_flat[:, i * num_tok : (i + 1) * num_tok]
        return nn_codes

    def kv_to_flat(self, kv_dict, key_order=None):
        if key_order is None:
            key_order = list(self.key_dims.keys())
        return torch.cat([kv_dict[k] for k in key_order if k in kv_dict], dim=-1)

    def flat_to_kv(self, flat_tensor, parts_meta=None):
        pm = self.key_dims
        dims = list(pm.values())
        total = sum(dims)
        splits = torch.split(flat_tensor[..., :total], dims, dim=-1)
        return dict(zip(pm.keys(), splits))

    # ── Core encode / decode ───────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_impl(
        self,
        component_dict: Union[Dict[str, torch.Tensor], torch.Tensor],
        *,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        action_op_mask: Optional[torch.Tensor] = None,
        num_residuals: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        is_flat_input = isinstance(component_dict, torch.Tensor)

        # Step 1: If flat tensor, split into per-key dict using key_dims layout
        if is_flat_input:
            if not self.key_dims:
                raise ValueError("Flat tensor encode requires key_dims in config.")
            dims = list(self.key_dims.values())
            total = sum(dims)
            splits = torch.split(component_dict[..., :total], dims, dim=-1)
            component_dict = dict(zip(self.key_dims.keys(), splits))

        # Step 2: Partition → merge → chunk; route rule keys out, pad/chunk NN keys
        nn_chunked, state = self._partitioner.encode_partition(
            component_dict,
            action_dim_is_pad=action_dim_is_pad,
            action_op_mask=action_op_mask,
            is_flat_input=is_flat_input,
        )
        # Step 3: NN path — VQ-VAE encode chunked tensors → per-key codes (B, n_cb, code_len)
        nn_codes = self.model.encode(nn_chunked)
        self._last_encode_state = state

        result: Dict[str, torch.Tensor] = {}

        # Step 4: Rule-based path — binarize + encode each rule key
        if self._rule_tokenizer is not None and state.rule_dict:
            result.update(self._encode_rule_keys(state.rule_dict))

        # Step 5: Collect NN codes in chunk_map order (preserves key ordering)
        nn_key_names = []
        for orig_k in state.original_key_dims:
            for ck in state.chunk_map[orig_k]:
                if ck in nn_codes:
                    nn_key_names.append(ck)
                    result[ck] = nn_codes[ck]

        # Step 6: Serialize NN codes to flat (B, nn_total) + concat rule tokens → "codes"
        nn_flat = self.serialize_codes(
            {k: nn_codes[k] for k in nn_key_names}, num_residuals=num_residuals
        )
        parts: List[torch.Tensor] = [nn_flat]
        if self._rule_tokenizer is not None and state.rule_dict:
            parts.extend(result[rk] for rk in state.rule_dict)
        result["codes"] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        result["_partition_state"] = state  # travels with codes for stateless decode

        return result

    def _encode_rule_keys(self, rule_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode rule-based keys: binarize each (B,T,1) column → integer tokens (B, num_tokens)."""
        result: Dict[str, torch.Tensor] = {}
        for rk, rv in rule_dict.items():
            if rv.shape[-1] != 1:
                raise ValueError(
                    f"rule_based tokenizer expects D=1 for key '{rk}', got D={rv.shape[-1]}"
                )
            col = rv[..., 0].float()  # (B, T)
            B = col.shape[0]
            # Batched: binarize entire batch, then encode per sample
            binarized = [self._rule_tokenizer.binarize(col[b].tolist()) for b in range(B)]
            tokens_batch = [self._rule_tokenizer.encode(b) for b in binarized]
            result[rk] = torch.tensor(tokens_batch, dtype=torch.long, device=rv.device)
        return result

    @torch.no_grad()
    def _decode_impl(
        self,
        codes_dict: Dict[str, torch.Tensor],
        *,
        key_dims: Optional[Dict[str, int]] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        num_residuals: Optional[int] = None,
        target_time_steps: Optional[int] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # Prefer state embedded in codes_dict (by _encode_impl) to avoid race
        # when encode is called multiple times before decode (e.g., eval loops).
        state = codes_dict.get("_partition_state", self._last_encode_state)
        was_flat = state.is_flat_input if state else False

        # Determine decode path: "flat" means codes are serialized in a single "codes" tensor;
        # otherwise codes are per-key (B, n_cb, code_len) tensors in the dict.
        is_flat_path = was_flat and "codes" in codes_dict
        if not is_flat_path:
            is_flat_path = "codes" in codes_dict and not any(
                isinstance(v, torch.Tensor) and v.ndim == 3
                for k, v in codes_dict.items()
                if k != "codes"
            )

        # Step 1: Deserialize flat codes into per-key NN codes + rule token slice
        if is_flat_path:
            nn_codes, rule_flat = self._deserialize_flat_codes(
                codes_dict["codes"], state, num_residuals
            )
        else:
            nn_codes = {
                k: v for k, v in codes_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 3
            }
            rule_flat = None

        # Step 2: Trim residual codebooks if using fewer than total
        if num_residuals is not None:
            nn_codes = {k: v[:, :num_residuals] if v.ndim == 3 else v for k, v in nn_codes.items()}

        # Step 3: NN path — VQ-VAE decode + unchunk/unmerge
        d_orig = key_dims if key_dims is not None else (self.key_dims or None)
        recon = self.model.decode(nn_codes, d_original=d_orig)
        if state:
            recon = self._partitioner.decode_partition(recon, state)

        # Step 4: Rule-based path — decode integer tokens back to binary signals
        if self._rule_tokenizer is not None:
            self._decode_rule_keys(recon, codes_dict, rule_flat, is_flat_path)

        # Step 5: If input was flat, concatenate per-key recon back to flat tensor
        if is_flat_path:
            key_order = state.key_order if state else list(self.key_dims.keys())
            return torch.cat([recon[k] for k in key_order if k in recon], dim=-1)

        return recon

    def _decode_rule_keys(
        self,
        recon: Dict[str, torch.Tensor],
        codes_dict: Dict[str, torch.Tensor],
        rule_flat: Optional[torch.Tensor],
        is_flat_path: bool,
    ) -> None:
        """Decode rule-based tokens → binary signals, inserting into recon dict in-place."""
        rule_keys = self._rule_keys
        if not rule_keys:
            return
        num_tok = self._rule_tokenizer.num_tokens

        if is_flat_path and rule_flat is not None:
            # Flat path: rule tokens are sliced from the tail of the flat codes tensor
            B = rule_flat.shape[0]
            for i, rk in enumerate(rule_keys):
                toks = rule_flat[:, i * num_tok : (i + 1) * num_tok]
                recon[rk] = self._rule_tokens_to_binary(toks, B, rule_flat.dtype, rule_flat.device)
        elif not is_flat_path:
            # Per-key path: rule tokens are separate entries in codes_dict
            for rk in rule_keys:
                if rk in codes_dict:
                    toks = codes_dict[rk]
                    B = toks.shape[0]
                    recon[rk] = self._rule_tokens_to_binary(toks, B, torch.float32, toks.device)

    def _rule_tokens_to_binary(
        self, toks: torch.Tensor, B: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Decode (B, num_tokens) integer tokens → (B, T, 1) float tensor with ±1.0 values."""
        binary_batch = [self._rule_tokenizer.decode(toks[b].tolist()) for b in range(B)]
        return torch.tensor(
            [[[1.0 if v == 1 else -1.0] for v in seq] for seq in binary_batch],
            dtype=dtype,
            device=device,
        )

    def _deserialize_flat_codes(
        self,
        flat_all: torch.Tensor,
        state: Optional[PartitionState],
        num_residuals,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Split flat (B, total_tokens) into per-key NN codes dict + rule token slice.

        Layout: [NN tokens for all NN keys] [rule tokens for all rule keys]
        NN tokens are serialized as level-first: (key0_cb0, key0_cb1, …, key1_cb0, …)
        Rule tokens are simply concatenated per key in key_dims order.
        """
        has_rule = self._rule_tokenizer is not None
        chunk_map = (
            state.chunk_map
            if state
            else self._partitioner.chunk(
                {k: torch.zeros(1, 1, v) for k, v in self.key_dims.items()}
            )[1]
        )

        # Build ordered list of NN key names, expanding chunk_map (e.g. key→[key__chunk_0, key__chunk_1])
        nn_key_names = []
        for k in self._nn_keys:
            nn_key_names.extend(chunk_map.get(k, [k]))
        # If no rule tokenizer exists, rule keys also go through NN path
        if not has_rule:
            for k in self._rule_keys:
                nn_key_names.extend(chunk_map.get(k, [k]))

        # Split off rule tokens from the tail
        rule_flat = None
        if has_rule and self._rule_keys:
            num_tok = self._rule_tokenizer.num_tokens
            gr_total = len(self._rule_keys) * num_tok
            rule_flat = flat_all[:, -gr_total:]
            flat_all = flat_all[:, :-gr_total]

        nn_codes = self.deserialize_codes(flat_all, nn_key_names, num_residuals=num_residuals)
        return nn_codes, rule_flat

    def encode(
        self,
        action: Union[Dict[str, torch.Tensor], torch.Tensor],
        *,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        action_op_mask: Optional[torch.Tensor] = None,
        num_residuals: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        nr = num_residuals if num_residuals is not None else self._default_num_residuals
        return self._encode_impl(
            action,
            action_dim_is_pad=action_dim_is_pad,
            action_op_mask=action_op_mask,
            num_residuals=nr,
        )

    def decode(
        self,
        codes: Dict[str, torch.Tensor],
        *,
        key_dims: Optional[Dict[str, int]] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        num_residuals: Optional[int] = None,
        target_time_steps: Optional[int] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        nr = num_residuals if num_residuals is not None else self._default_num_residuals
        return self._decode_impl(
            codes,
            key_dims=key_dims,
            action_dim_is_pad=action_dim_is_pad,
            num_residuals=nr,
            target_time_steps=target_time_steps,
        )

    def forward(self, batch_or_component_dict: Dict, key_dims=None):
        if isinstance(batch_or_component_dict, dict) and "action" in batch_or_component_dict:
            action = batch_or_component_dict["action"]
            if isinstance(action, (torch.Tensor, dict)):
                return self._training_forward(batch_or_component_dict)

        d_orig = key_dims if key_dims is not None else (self.key_dims or None)
        recon_dict, codes_dict, loss_dict = self.model(batch_or_component_dict, d_original=d_orig)
        return recon_dict, codes_dict, loss_dict

    def _training_forward(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Training forward: action → VQ-VAE reconstruction loss.

        Handles both flat tensor and kv dict actions.
        Rule-based keys are filtered out (only NN keys are trained).
        Consistency augmentation is applied when configured.
        """
        action = batch["action"]
        step = int(batch.get("_step", 0))
        max_steps = int(batch.get("_max_steps", 1))

        if isinstance(action, torch.Tensor):
            if not self.key_dims:
                raise ValueError(
                    "ActionCodecV2Wrapper: flat action tensor received but "
                    "key_dims is empty. Set cfg.key_dims in the tokenizer config."
                )
            dims = list(self.key_dims.values())
            total = sum(dims)
            splits = torch.split(action[..., :total], dims, dim=-1)
            component_dict = dict(zip(self.key_dims.keys(), splits))
            # Training only uses NN keys; rule keys are not trainable
            if self._partitioner.rule_based_key_patterns:
                component_dict = {
                    k: v for k, v in component_dict.items() if not self.is_rule_based_key(k)
                }

            normed_pos = None
            layer_weights = None
            model_cfg = self._model_arch_cfg
            if (
                model_cfg is not None
                and float(getattr(model_cfg, "consistency_loss_weight", 0.0)) > 0
            ):
                if self._augmenter is None:
                    self._augmenter = ConsistencyAugmenter(model_cfg)
                if self._augmenter.is_active(step, max_steps):
                    _, _, layer_weights = self._augmenter.get_schedule(step, max_steps)
                    action_pos = self._augmenter.augment(action, step, max_steps)
                    splits_pos = torch.split(action_pos[..., :total], dims, dim=-1)
                    normed_pos = dict(zip(self.key_dims.keys(), splits_pos))

        elif isinstance(action, dict):
            component_dict = action
            normed_pos = None
            layer_weights = None
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        d_orig = self.key_dims or None
        _recon_normed, _codes_dict, loss_dict = self.model(
            component_dict,
            d_original=d_orig,
            x_pos_dict=normed_pos,
            layer_weights=layer_weights,
        )

        log_dict = {k: float(v) for k, v in loss_dict.items()}
        return loss_dict["loss"], log_dict

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        logger.info(f"ActionCodecV2Wrapper: model saved to {save_directory}")

    def load_model(self, load_path: str):
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        if isinstance(checkpoint, dict) and "tokenizer_meta" in checkpoint:
            meta = checkpoint["tokenizer_meta"]
            pm = meta.get("parts_meta", None)
            if pm and not self._user_key_dims:
                self.key_dims = dict(pm)
                self.__dict__.pop("_nn_keys_cached", None)
                self.__dict__.pop("_rule_keys_cached", None)
                logger.info(
                    f"ActionCodecV2Wrapper: loaded key_dims from checkpoint: "
                    f"{list(self.key_dims.keys())}"
                )
            elif pm and self._user_key_dims:
                logger.info(
                    f"ActionCodecV2Wrapper: config key_dims takes priority over "
                    f"checkpoint key_dims (config={dict(self.key_dims)}, "
                    f"ckpt={list(pm.keys())})"
                )
        else:
            logger.warning(
                f"ActionCodecV2Wrapper: checkpoint '{load_path}' has no tokenizer_meta. "
                f"Falling back to YAML key_dims. Re-save with train_vq.py to embed it."
            )

        result = self.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"load_model missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"load_model unexpected keys: {result.unexpected_keys}")
        logger.info(f"ActionCodecV2Wrapper: loaded weights from {load_path}")

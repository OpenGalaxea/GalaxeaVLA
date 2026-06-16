# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""VQ Action Tokenizer — adapter between VQ codec backend and VLM token space.

Responsibilities:
  - Register action/marker tokens in the HF tokenizer's vocab
  - Encode: action tensor → backend.encode → serializer (inserts markers) → action indices
  - Decode: action indices → strip markers → backend.decode → action tensor
  - Map action_idx ↔ HF tokenizer_idx (offset or fast_skip)

Markers are appended after the codebook range:
  [0, codebook_size)          → VQ code indices
  [codebook_size, codebook_size + n_markers) → marker indices
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from g05.tokenizer.models.protocol import DecodeMetadata, _validate_backend
from g05.utils.common.import_utils import get_obj_from_str

from .base_action_tokenizer import ActionIdxBatch, BaseActionTokenizer, DecodeResult, TokenIds
from .serialization import (
    FlatSerializer,
    GroupedFlatSerializer,
    _aggregate_actions,
    PartSignSerializer,
)

logger = logging.getLogger(__name__)


def _compute_marker_strs(nn_keys: List[str], rule_keys: List[str], max_residuals: int) -> List[str]:
    """Marker token strings for ALL residual levels (registered at init, subset emitted at runtime).

    NN keys: <key_level> per level if max_residuals > 1, else <key>.
    Rule keys: <key> (always single marker, not per-level).
    """
    if max_residuals > 1:
        return [f"<{k}_{l}>" for l in range(max_residuals) for k in nn_keys] + [
            f"<{k}>" for k in rule_keys
        ]
    return [f"<{k}>" for k in nn_keys] + [f"<{k}>" for k in rule_keys]


class VQActionTokenizer(BaseActionTokenizer):
    """Flat-path VQ action tokenizer — maps between backend VQ codes and LLM token IDs.

    Args:
        tokenizer: HuggingFace tokenizer (used to register new action/marker tokens).
        vq_config: Configuration dict. Required keys: vqvae_type. See YAML configs for full schema.
        cache_dir: Unused, kept for BaseActionTokenizer compat.
        action_dim: Override for backend action_dim (rarely needed).
        device: Device string, forwarded to backend. Defaults to "cpu".
    """

    def __init__(
        self,
        vq_config: DictConfig | None = None,
        cache_dir: str | None = None,
        *,
        vocab_offset: int = 0,
        hf_vocab_size: int | None = None,
        action_dim: int | None = None,
        device: str = "cpu",
    ) -> None:
        if vq_config is None:
            vq_config = OmegaConf.create()
        if not isinstance(vq_config, DictConfig):
            raise TypeError(f"vq_config must be OmegaConf DictConfig, got: {type(vq_config)}")
        OmegaConf.set_struct(vq_config, False)

        self.device = device
        self.action_dim = action_dim
        self.n_quantizers = vq_config.get("n_quantizers", None)
        self._last_n_quantizers: int | None = None
        self._default_num_residuals: int | None = vq_config.get("num_residuals", None)
        self.dropout_noop_parts = bool(vq_config.get("dropout_noop_parts", False))
        self.rule_based_key_patterns: List[str] = list(vq_config.get("rule_based_key_patterns", []))
        self.rule_based_binarize_threshold: float = float(
            vq_config.get("rule_based_binarize_threshold", 0.0)
        )
        self.absent_key_fill_value: float | None = (
            float(v) if (v := vq_config.get("absent_key_fill_value", None)) is not None else None
        )

        # ── Load backend ────────────────────────────────────────────────────
        vqvae_type: str | None = vq_config.get("vqvae_type", None)
        if not vqvae_type:
            raise ValueError("vqvae_type must be specified in vq_config.")
        vq_config.setdefault("device", device)
        self.action_tokenizer = get_obj_from_str(vqvae_type)(vq_config)
        self._action_vocab_size = int(getattr(self.action_tokenizer, "vocab_size"))

        if hasattr(self.action_tokenizer, "configure_eval"):
            _validate_backend(self.action_tokenizer)

        # ── Cache backend metadata (read once, used throughout lifetime) ────
        self._decode_meta: DecodeMetadata | None = None
        if hasattr(self.action_tokenizer, "get_decode_metadata"):
            self._decode_meta = self.action_tokenizer.get_decode_metadata()

        # ── Prepare defaults for BaseActionTokenizer ────────────────────────
        vq_config.setdefault("use_extra_tokens", True)
        vq_config.setdefault("fast_skip_tokens", 128)

        # ── Compute action_token_len before super().__init__ ────────────────
        # code_parts: dict of part_name -> token count (codec-side, WITHOUT markers).
        # _raw_token_len: total codec tokens (without markers).  In grouped mode
        # markers are added later by _init_grouped_tokens, so action_token_len
        # ends up larger than _raw_token_len.
        code_parts = getattr(self.action_tokenizer, "code_parts", None)
        _raw_token_len = (
            int(sum(code_parts.values()))
            if code_parts
            else int(getattr(self.action_tokenizer, "action_token_len", 1))
        )

        block_wise = vq_config.get("block_wise_autoregressive", False)
        if block_wise:
            vq_config["ar_block_num"] = int(vq_config.get("ar_block_num", 0))
            block_size = vq_config.get("block_size", None)
            assert block_size is not None, "block_wise_autoregressive requires block_size"
            _codec_action_token_len_real = _raw_token_len
            _codec_action_token_len = int(
                math.ceil(_codec_action_token_len_real / int(block_size))
            ) * int(block_size)
            vq_config["num_block"] = _codec_action_token_len // int(block_size)
        else:
            _codec_action_token_len = _raw_token_len
            _codec_action_token_len_real = _raw_token_len

        super().__init__(
            vq_config=vq_config,
            cache_dir=cache_dir,
            vocab_offset=vocab_offset,
            hf_vocab_size=hf_vocab_size,
        )

        # Post-super: set action_token_len_real and apply BAR padding if needed.
        # _init_grouped_tokens already set action_token_len (with markers);
        # _init_block_wise_configuration now preserves it (incremental pattern).
        # Flat mode still needs action_token_len set here.
        grouped_with_markers = isinstance(getattr(self, "serializer", None), GroupedFlatSerializer)
        if grouped_with_markers:
            self.action_token_len_real = self.action_token_len
            if block_wise:
                bs = int(vq_config.get("block_size", block_size))
                self.action_token_len = int(math.ceil(self.action_token_len / bs) * bs)
                self.num_block = self.action_token_len // bs
                vq_config["num_block"] = self.num_block
        else:
            self.action_token_len = int(_codec_action_token_len)
            self.action_token_len_real = int(_codec_action_token_len_real)

        assert self.action_token_len is not None and self.action_token_len > 0, (
            f"action_token_len must be positive after init, got {self.action_token_len}"
        )
        assert self.action_token_len_real is not None and self.action_token_len_real > 0, (
            f"action_token_len_real must be positive after init, got {self.action_token_len_real}"
        )

    # =========================================================================
    # Token configuration
    # =========================================================================

    def _init_token_configuration(self, vq_config: dict):
        self.use_extra_tokens = vq_config.get("use_extra_tokens", True)
        self.relaxed_decoding = vq_config.get("relaxed_decoding", False)
        self.new_tokens = []

        if bool(vq_config.get("use_group_markers", False)):
            self._init_grouped_tokens(vq_config)
        else:
            self._init_flat_tokens(vq_config)

    def _init_flat_tokens(self, vq_config: dict):
        """Register flat action tokens (no group markers)."""
        self.serializer = FlatSerializer(
            total_codes=sum(self.action_tokenizer.code_parts.values())
            if hasattr(self.action_tokenizer, "code_parts")
            else int(getattr(self.action_tokenizer, "action_token_len", 1))
        )

        if self.use_extra_tokens:
            action_tokens = [f"<action{i:0>4}>" for i in range(self._codebook_size)]
            self.action_token_begin_idx = self._vocab_offset
            self.action_token_end_idx = self.action_token_begin_idx + len(action_tokens)
            self.action_tokens = action_tokens
            self._new_action_tokens = list(action_tokens)
        else:
            self._fast_skip_tokens = int(vq_config.get("fast_skip_tokens", 128))
            self.action_token_begin_idx = (
                self._hf_vocab_size - 1 - self._fast_skip_tokens - int(self._action_vocab_size)
            )
            self.action_token_end_idx = self.action_token_begin_idx + int(self._action_vocab_size)

        logger.info(
            f"action_token_begin_idx: {self.action_token_begin_idx}, action_token_end_idx: {self.action_token_end_idx}"
        )

    def _init_grouped_tokens(self, vq_config: dict):
        """Register action tokens + group markers appended after the codebook.

        Key names come from self._decode_meta (wrapper already resolved merge_spec/parts_meta).
        Markers for ALL residual levels are registered at init; runtime emits only num_residuals levels.
        """
        if not self.use_extra_tokens:
            raise NotImplementedError("use_group_markers requires use_extra_tokens=True")

        meta = self._decode_meta
        if meta is None:
            logger.warning("_decode_meta not available; falling back to flat mode")
            self._init_flat_tokens(vq_config)
            return

        nn_key_names = meta.nn_key_names
        rule_key_names = meta.rule_key_names
        if not nn_key_names and not rule_key_names:
            logger.warning("No NN or rule keys; falling back to flat mode")
            self._init_flat_tokens(vq_config)
            return

        total_residuals = meta.total_residuals or 1
        num_residuals = int(
            vq_config.get("num_residuals", None) or self._default_num_residuals or total_residuals
        )

        all_marker_strs = _compute_marker_strs(nn_key_names, rule_key_names, total_residuals)
        n_markers = len(all_marker_strs)

        # Register codebook tokens + marker tokens
        action_tokens = [f"<action{i:0>4}>" for i in range(self._codebook_size)]
        action_tokens.extend(all_marker_strs)

        self.action_token_begin_idx = self._vocab_offset
        self.action_token_end_idx = self.action_token_begin_idx + len(action_tokens)
        self.action_tokens = action_tokens
        self._new_action_tokens = list(action_tokens)

        # Markers are appended after codebook range
        self._action_vocab_size = self._codebook_size + n_markers
        self._group_marker_action_indices: dict[str, int] = {
            tok: self._codebook_size + i for i, tok in enumerate(all_marker_strs)
        }

        self.serializer = GroupedFlatSerializer(
            nn_key_names=nn_key_names,
            rule_key_names=rule_key_names,
            code_len=meta.code_len,
            rule_tokens_per_key=meta.rule_tokens_per_key,
            num_residuals=num_residuals,
            max_residuals=total_residuals,
            group_marker_action_indices=self._group_marker_action_indices,
            group_order_shuffle=bool(vq_config.get("group_order_shuffle", False)),
        )

        # action_token_len = codes + markers at runtime num_residuals.
        # nn_chunked_key_count accounts for keys split into chunks at encode time
        # (when dim > max_component_dim); markers use the original key count.
        nn_chunked = meta.nn_chunked_key_count or len(nn_key_names)
        nn_code_total = nn_chunked * num_residuals * meta.code_len
        rule_code_total = len(rule_key_names) * meta.rule_tokens_per_key
        runtime_markers = len(nn_key_names) * num_residuals + len(rule_key_names)
        self.action_token_len = nn_code_total + rule_code_total + runtime_markers
        self.action_token_len_real = self.action_token_len

        logger.info(
            f"action_token_begin_idx: {self.action_token_begin_idx}, "
            f"action_token_end_idx: {self.action_token_end_idx}, "
            f"group_markers({n_markers}): {all_marker_strs} "
            f"(action_idx range [{self._codebook_size}, {self._codebook_size + n_markers})"
        )

    def _n_group_markers(self) -> int:
        """Number of registered group marker tokens (all residual levels)."""
        return (
            len(self._group_marker_action_indices)
            if isinstance(self.serializer, GroupedFlatSerializer)
            else 0
        )

    # =========================================================================
    # Encode
    # =========================================================================

    def _encode_action_indices(
        self, action: torch.Tensor, encode_kwargs: dict | None = None
    ) -> ActionIdxBatch:
        kw = dict(encode_kwargs or {})
        if "num_residuals" not in kw and self._default_num_residuals is not None:
            kw["num_residuals"] = self._default_num_residuals

        use_amp = torch.cuda.is_available()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32, enabled=use_amp):
            codes_dict = self.action_tokenizer.encode(
                action,
                action_dim_is_pad=kw.get("action_dim_is_pad"),
                action_op_mask=kw.get("action_op_mask"),
                num_residuals=kw.get("num_residuals"),
            )

        # Discard noop parts only when dropout_noop_parts is enabled
        noop_keys_per_sample = None
        if (
            self.dropout_noop_parts
            and isinstance(self.serializer, GroupedFlatSerializer)
            and self._decode_meta is not None
        ):
            noop_keys_per_sample = self._derive_noop_keys(
                action_op_mask=kw.get("action_op_mask"),
                action_dim_is_pad=kw.get("action_dim_is_pad"),
                action=action,
            )

        serializer_nr = (
            kw.get("num_residuals") if isinstance(self.serializer, GroupedFlatSerializer) else None
        )
        return self.serializer.codes_to_action_indices(
            codes_dict, noop_keys_per_sample=noop_keys_per_sample, num_residuals=serializer_nr
        )

    def _derive_noop_keys(
        self,
        action_op_mask=None,
        action_dim_is_pad=None,
        action=None,
    ) -> "list[set[str]] | None":
        """Derive noop keys per sample from action_op_mask and/or action_dim_is_pad.

        A key is noop if ALL its dimensions are either:
        - inactive in action_op_mask (all False), OR
        - padding in action_dim_is_pad (all True).

        Always returns List[set[str]] (per-sample), or None if no source available.
        Dict inputs (single-sample) are broadcast to batch-size-1 lists.
        """
        if self._decode_meta is None or (
            action_op_mask is None and action_dim_is_pad is None and action is None
        ):
            return None

        key_dims = self._decode_meta.canonical_parts_meta
        keys = list(key_dims.keys())
        dims = list(key_dims.values())
        offsets = [0] + list(torch.tensor(dims[:-1]).cumsum(0).tolist()) if dims else []

        def _noop_from_op_mask(mask_flat, n) -> set:
            """Keys with all-False dims in op mask (no active operation)."""
            return {
                k
                for k, off, d in zip(keys, offsets[:n], dims[:n])
                if not mask_flat[off : off + d].any()
            }

        def _noop_from_pad(pad_flat, n) -> set:
            """Keys with all-True dims in pad mask (cross-embodiment padding)."""
            return {
                k for k, off, d in zip(keys, offsets[:n], dims[:n]) if pad_flat[off : off + d].all()
            }

        n = len(keys)

        # Resolve batch size from whichever source is available
        batch_size = None
        if isinstance(action_op_mask, torch.Tensor):
            batch_size = action_op_mask.shape[0]
        elif isinstance(action_dim_is_pad, torch.Tensor):
            batch_size = action_dim_is_pad.shape[0]
        elif isinstance(action, torch.Tensor) and action.ndim >= 3:
            batch_size = action.shape[0]

        if batch_size is not None:
            result = []
            for i in range(batch_size):
                noop = set()
                if isinstance(action_op_mask, torch.Tensor):
                    noop |= _noop_from_op_mask(action_op_mask[i].flatten(), n)
                if isinstance(action_dim_is_pad, torch.Tensor):
                    noop |= _noop_from_pad(action_dim_is_pad[i].flatten(), n)
                if action is not None and isinstance(action, torch.Tensor) and action.ndim >= 3:
                    action_flat = action[i].reshape(action.shape[1], -1)
                    for k, off, d in zip(keys, offsets[:n], dims[:n]):
                        if k in noop:
                            continue
                        if not self._is_rule_based_key(k):
                            continue
                        key_slice = action_flat[:, off : off + d]
                        binarized = (key_slice >= self.rule_based_binarize_threshold).long()
                        if (binarized == binarized[0:1]).all():
                            noop.add(k)
                result.append(noop)
            return result

        # Single-sample dict / non-tensor path
        noop: set = set()
        if isinstance(action_op_mask, dict):
            noop |= {k for k in keys if (m := action_op_mask.get(k)) is not None and not m.any()}
        if isinstance(action_dim_is_pad, dict):
            noop |= {k for k in keys if (m := action_dim_is_pad.get(k)) is not None and m.all()}
        if action is not None and isinstance(action, torch.Tensor) and action.ndim >= 2:
            action_flat = action.reshape(action.shape[0], -1)
            for k, off, d in zip(keys, offsets[:n], dims[:n]):
                if k in noop:
                    continue
                if not self._is_rule_based_key(k):
                    continue
                key_slice = action_flat[:, off : off + d]
                binarized = (key_slice >= self.rule_based_binarize_threshold).long()
                if (binarized == binarized[0:1]).all():
                    noop.add(k)
        return [noop]

    def _is_rule_based_key(self, key: str) -> bool:
        return any(p in key for p in self.rule_based_key_patterns)

    # =========================================================================
    # Decode
    # =========================================================================

    def _decode_grouped_with_markers(
        self,
        action_indices: ActionIdxBatch,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, list[set[str]]]:
        """Marker-aware decode for GroupedFlatSerializer partial sequences.

        When the AR model generates only some keys (noop parts absent), parse the
        markers to recover per-key codes, decode with zeros for absent keys, then
        zero those keys' dims in the output so absent parts produce no movement.

        Returns:
            (flat_out, absent_keys_per_sample)
        """
        meta = self._decode_meta
        serializer: GroupedFlatSerializer = self.serializer  # type: ignore[assignment]
        fwd_kwargs = self._prepare_decode_kwargs(decode_kwargs, len(action_indices))

        per_sample = [
            serializer.parse_per_key_codes(row, num_residuals=fwd_kwargs.get("num_residuals"))
            for row in action_indices
        ]

        all_keys = set(serializer.nn_key_names) | set(serializer.rule_key_names)
        absent_keys_per_sample = [
            all_keys - (set(nn_present.keys()) | set(rule_present.keys()))
            for nn_present, rule_present in per_sample
        ]

        # Infer nr from parsed levels (max found across all present keys/samples)
        inferred_nr = fwd_kwargs.get("num_residuals") or serializer.num_residuals
        for nn_present, _ in per_sample:
            for levels in nn_present.values():
                inferred_nr = max(inferred_nr, len(levels))
        fwd_kwargs["num_residuals"] = inferred_nr

        nr = inferred_nr
        code_len = serializer.code_len
        rule_tpk = serializer.rule_tokens_per_key

        codes_dict: Dict = {}
        for k in serializer.nn_key_names:
            key_batch = []
            for nn_present, _ in per_sample:
                levels = nn_present.get(k)
                if levels is not None:
                    padded = levels + [[0] * code_len] * (nr - len(levels))
                    key_batch.append(padded[:nr])
                else:
                    key_batch.append([[0] * code_len] * nr)
            codes_dict[k] = torch.clamp(
                torch.tensor(key_batch, dtype=torch.long, device=self.device),
                0,
                self._codebook_size - 1,
            )  # (B, nr, code_len)

        # Rule keys: marker-aware parsing slices the next `rule_tpk` tokens after each
        # rule marker without validating. In BAR mode the model can emit other group
        # markers (action_idx >= codebook_size) right after <gripper>, polluting the
        # rule slice. Drop any out-of-codec values per sample; if too few valid tokens
        # remain, treat the key as absent (→ [0]*rule_tpk → constrained_tokenizer
        # decodes to a valid index 0 instead of triggering OOR clamp).
        for k in serializer.rule_key_names:
            rule_batch: list[list[int]] = []
            for _, rule_present in per_sample:
                raw = rule_present.get(k)
                if raw is None:
                    rule_batch.append([0] * rule_tpk)
                    continue
                filtered = [t for t in raw if 0 <= t < self._codebook_size]
                if len(filtered) < rule_tpk:
                    rule_batch.append([0] * rule_tpk)
                else:
                    rule_batch.append(filtered[:rule_tpk])
            codes_dict[k] = torch.tensor(rule_batch, dtype=torch.long, device=self.device)
            # (B, rule_tokens_per_key)

        with torch.autocast("cuda", dtype=torch.float32, enabled=True):
            ret = self.action_tokenizer.decode(
                codes_dict,
                key_dims=fwd_kwargs.get("key_dims"),
                action_dim_is_pad=fwd_kwargs.get("action_dim_is_pad"),
                num_residuals=nr,
                target_time_steps=fwd_kwargs.get("target_time_steps"),
            )

        if isinstance(ret, torch.Tensor):
            flat_out = ret.detach().cpu().numpy()
        elif isinstance(ret, dict):
            flat_out = self.action_tokenizer.kv_to_flat(ret).detach().cpu().numpy()
        else:
            flat_out = np.asarray(ret)

        _gripper_fill_val = (
            self.absent_key_fill_value if self.absent_key_fill_value is not None else 0.0
        )
        _default_fill_val = 0.0
        # Fill absent keys' action dims: rule-based (gripper) keys use sentinel, NN keys use 0.0
        if meta is not None and flat_out.ndim >= 2:
            key_order = list(meta.canonical_parts_meta.keys())
            offset = 0
            key_offsets: Dict[str, tuple] = {}
            for k in key_order:
                d = meta.canonical_parts_meta[k]
                key_offsets[k] = (offset, offset + d)
                offset += d

            for b, (nn_present, rule_present) in enumerate(per_sample):
                for k in serializer.nn_key_names:
                    if k not in nn_present and k in key_offsets:
                        s, e = key_offsets[k]
                        flat_out[b, ..., s:e] = _default_fill_val
                for k in serializer.rule_key_names:
                    if k not in rule_present and k in key_offsets:
                        s, e = key_offsets[k]
                        flat_out[b, ..., s:e] = _gripper_fill_val

        return flat_out, absent_keys_per_sample

    def _decode_action_indices(
        self,
        action_indices: ActionIdxBatch,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, list[set[str]]]:
        _empty_absent: list[set[str]] = [set() for _ in action_indices] if action_indices else []
        if not action_indices:
            return np.zeros((0, 0, 0), dtype=np.float32), []

        # Strip group markers and route partial/noop-dropped sequences BEFORE tensor conversion.
        # When dropout_noop_parts=True, different samples may drop different parts, producing
        # rows of different lengths (jagged). torch.tensor() would crash on jagged input, so
        # we must handle this case here — before any tensor operations.
        if isinstance(self.serializer, GroupedFlatSerializer):
            stripped = [self.serializer.strip_markers(row) for row in action_indices]
            row_lens = [len(r) for r in stripped]
            # Route to marker-aware decode if ANY group marker is present in the sequences.
            # This is the primary signal that the sequence was produced with use_group_markers=True
            # and must be parsed key-by-key.  Relying solely on stripped length is insufficient:
            # when dropout_noop_parts=True drops some keys the remaining stripped length can
            # coincidentally equal an expected_length for a different num_residuals (e.g. dropping
            # right_control with nr=2 leaves stripped_len=20 which equals expected_lengths[0] for
            # nr=1), causing the flat path to misassign codes across keys and produce ~1.0 L1.
            marker_values = set(self.serializer.group_marker_action_indices.values())
            has_markers = marker_values and any(
                any(t in marker_values for t in row) for row in action_indices
            )
            if (
                has_markers
                or len(set(row_lens)) > 1
                or (
                    self._decode_meta is not None
                    and row_lens[0] not in self._decode_meta.expected_lengths
                )
            ):
                return self._decode_grouped_with_markers(
                    action_indices,
                    time_horizon=time_horizon,
                    action_dim=action_dim,
                    decode_kwargs=decode_kwargs,
                )
            ids = torch.tensor(stripped, dtype=torch.long)
        else:
            ids = torch.tensor(action_indices, dtype=torch.long)

        # Validate length
        if (
            self._decode_meta is not None
            and ids.shape[-1] not in self._decode_meta.expected_lengths
        ):
            th = int(time_horizon or 1)
            ad = int(action_dim or (self.action_dim or 1))
            return np.zeros((len(action_indices), th, ad), dtype=np.float32), _empty_absent

        ids = torch.clamp(ids, 0, int(self._action_vocab_size) - 1).to(self.device)
        fwd_kwargs = self._prepare_decode_kwargs(decode_kwargs, len(action_indices))

        # Infer num_residuals from actual sequence length to avoid shape mismatch when
        # the model generates fewer codebook levels than the config default (e.g. early
        # in training).  expected_lengths[nr-1] gives the total token count for nr levels,
        # so the position in that list is the ground-truth nr for this sequence.
        if self._decode_meta is not None and ids.shape[-1] in self._decode_meta.expected_lengths:
            inferred_nr = self._decode_meta.expected_lengths.index(ids.shape[-1]) + 1
            if fwd_kwargs.get("num_residuals") != inferred_nr:
                fwd_kwargs["num_residuals"] = inferred_nr

        codes = {"codes": ids}
        if action_dim is not None:
            backend_ad = int(getattr(self.action_tokenizer, "action_dim", 0) or 0)
            if backend_ad and action_dim < backend_ad:
                codes["_input_action_dim"] = int(action_dim)

        with torch.autocast("cuda", dtype=torch.float32, enabled=True):
            ret = self.action_tokenizer.decode(
                codes,
                key_dims=fwd_kwargs.get("key_dims"),
                action_dim_is_pad=fwd_kwargs.get("action_dim_is_pad"),
                num_residuals=fwd_kwargs.get("num_residuals"),
                target_time_steps=fwd_kwargs.get("target_time_steps"),
            )

        if isinstance(ret, torch.Tensor):
            return ret.detach().cpu().numpy(), _empty_absent
        if isinstance(ret, dict):
            return self.action_tokenizer.kv_to_flat(ret).detach().cpu().numpy(), _empty_absent
        return np.asarray(ret), _empty_absent

    def decode_token_ids_to_actions(
        self,
        action_token_ids: TokenIds,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        decode_kwargs: dict | None = None,
    ) -> DecodeResult:
        decode_kwargs = decode_kwargs or {}

        token_ids = (
            action_token_ids.detach().cpu()
            if isinstance(action_token_ids, torch.Tensor)
            else torch.tensor(action_token_ids, dtype=torch.long)
        )
        is_action_token_space = decode_kwargs.get("is_action_token_space", False)
        action_idx = (
            token_ids
            if is_action_token_space
            else torch.clamp(
                self._tokenizer_idx_to_action_idx(token_ids), 0, int(self._action_vocab_size) - 1
            )
        )

        assert action_idx.ndim == 1, f"action_idx must be 1D, got {action_idx.shape}"
        actions_np, absent_keys_list = self._decode_action_indices(
            [action_idx.tolist()],
            time_horizon=time_horizon,
            action_dim=action_dim,
            decode_kwargs=decode_kwargs,
        )
        return DecodeResult(
            action=torch.from_numpy(actions_np)[0],
            absent_keys=absent_keys_list[0] if absent_keys_list else set(),
        )

    def _prepare_decode_kwargs(self, decode_kwargs: dict | None, batch_size: int) -> dict:
        """Filter decode_kwargs to wrapper.decode() accepted keys."""
        prepared = dict(decode_kwargs or {})
        embodiment = prepared.pop("embodiment", None)
        if embodiment is not None and prepared.get("embodiment_ids") is None:
            parse_fn = getattr(self.action_tokenizer, "_parse_embodiment", None)
            if callable(parse_fn):
                ids = parse_fn(embodiment, batch_size, self.device)
                if ids is not None:
                    prepared["embodiment_ids"] = ids

        allowed = {
            "frequency",
            "embodiment_ids",
            "target_time_steps",
            "action_dim_is_pad",
            "num_residuals",
            "key_dims",
        }
        result = {k: v for k, v in prepared.items() if k in allowed}
        if "num_residuals" not in result and self._default_num_residuals is not None:
            result["num_residuals"] = self._default_num_residuals
        return result

    # =========================================================================
    # Block-wise autoregressive
    # =========================================================================

    def build_block_io(self, action_str: str, action_template: str | None = None):
        if not self.block_wise_autoregressive:
            raise ValueError("block_wise_autoregressive=False, cannot call build_block_io")
        return self._build_block_io(
            action_str, self.block_size, self._bos_blk_id, self._eos_blk_id, action_template
        )

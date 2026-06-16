# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Serialization strategies for converting between backend codes and frontend token sequences.

Strategies:
- FlatSerializer: single-part backends (DummyCodec, ActionCodecV2).
  codes dict {"codes": Tensor} ↔ flat list of int indices.
- GroupedFlatSerializer: extends FlatSerializer with per-group marker tokens.
  Markers are APPENDED after the codebook range — they occupy new token IDs,
  not carved from the codebook. Layout is fixed at init from parts_meta config.
  codes dict {"codes": Tensor} ↔ [<m0> code... <m1> code... ...]
- PartSignSerializer: multi-part backends (FasterV2 WBC).
  codes dict {"left_arm": T, "right_arm": T, "lower_body": T} ↔ interleaved sign+code sequence.

These are selected automatically by VQActionTokenizer based on config and backend type.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .bar_builder import ACTION_TOKEN_INDEX, _find_subsequence
from .base_action_tokenizer import ActionIdxBatch


class FlatSerializer:
    """Single-part backend: codes → flat token sequence."""

    def __init__(self, total_codes: int):
        self._total_codes = total_codes

    @property
    def total_codes(self):
        return self._total_codes

    def init_tokens(
        self, tokenizer, action_token_begin_idx: int, action_vocab_size: int
    ) -> List[str]:
        """No extra tokens needed for flat serialization."""
        return []

    def codes_to_action_indices(
        self, codes_dict: Dict[str, torch.Tensor], **kwargs
    ) -> ActionIdxBatch:
        """Convert backend codes dict to action index batch (list of list of int)."""
        vq_code = codes_dict["codes"]
        if isinstance(vq_code, torch.Tensor):
            vq_code = vq_code.detach().cpu().to(torch.long).tolist()
        return [list(map(int, row)) for row in vq_code]


class GroupedFlatSerializer(FlatSerializer):
    """Extends FlatSerializer with per-group marker tokens.

    The flat VQ code sequence has a specific structure:
      - NN portion: level-first serialized, each (key, level) gets code_len tokens
      - Rule portion: flat-appended at end, each rule key gets rule_tokens_per_key tokens
        (NOT split per level)

    Markers are inserted before each group segment:
      - NN keys: one marker per (key, level) → <key_level> for nr>1, <key> for nr=1
      - Rule keys: one marker per key → <key> regardless of nr

    **CRITICAL**: Markers are APPENDED after the codebook range. They do NOT
    occupy codebook token IDs. The action vocab is:
      [0, codebook_size) → VQ code indices
      [codebook_size, codebook_size + n_markers) → marker indices

    **ALL residual level markers are registered at init** (up to max_residuals).
    The `_layout` contains entries for all levels. At runtime, `num_residuals`
    controls which levels are actually emitted in codes_to_action_indices.
    """

    def __init__(
        self,
        nn_key_names: List[str],
        rule_key_names: List[str],
        code_len: int,
        rule_tokens_per_key: int,
        num_residuals: int,
        max_residuals: int,
        group_marker_action_indices: Dict[str, int],
        group_order_shuffle: bool = False,
    ):
        self.nn_key_names = nn_key_names
        self.rule_key_names = rule_key_names
        self.code_len = code_len
        self.rule_tokens_per_key = rule_tokens_per_key
        self.num_residuals = num_residuals
        self.max_residuals = max_residuals
        self.group_marker_action_indices = group_marker_action_indices
        self.group_order_shuffle = group_order_shuffle

        # Build layout for ALL residual levels (0..max_residuals-1).
        # Each entry: (key, level, marker_idx, slice_start, slice_end) in flat tensor.
        self._layout: List[Tuple[str, int, int, int, int]] = []
        offset = 0
        for level in range(max_residuals):
            for k in nn_key_names:
                marker_str = f"<{k}_{level}>" if max_residuals > 1 else f"<{k}>"
                marker_idx = group_marker_action_indices.get(marker_str)
                if marker_idx is None:
                    continue
                n_toks = code_len
                self._layout.append((k, level, marker_idx, offset, offset + n_toks))
                offset += n_toks

        for k in rule_key_names:
            marker_str = f"<{k}>"
            marker_idx = group_marker_action_indices.get(marker_str)
            if marker_idx is None:
                continue
            n_toks = rule_tokens_per_key
            self._layout.append((k, -1, marker_idx, offset, offset + n_toks))
            offset += n_toks

        # NN offset lookup: (key, level) → (marker_idx, abs_start, abs_end).
        # These offsets are stable regardless of runtime num_residuals because the NN
        # section is laid out level-major and levels 0..nr-1 always occupy the same slots.
        self._nn_offsets: Dict[Tuple[str, int], Tuple[int, int, int]] = {
            (k, lvl): (mid, s, e) for k, lvl, mid, s, e in self._layout if lvl >= 0
        }
        # Rule offset lookup: key → (marker_idx, rel_start, rel_end).
        # Relative to the start of the rule section (= nr * n_nn_keys * code_len at runtime).
        self._rule_rel_offsets: Dict[str, Tuple[int, int, int]] = {}
        rel = 0
        for k in rule_key_names:
            marker_str = f"<{k}>"
            marker_idx = group_marker_action_indices.get(marker_str)
            if marker_idx is None:
                rel += rule_tokens_per_key
                continue
            self._rule_rel_offsets[k] = (marker_idx, rel, rel + rule_tokens_per_key)
            rel += rule_tokens_per_key

        # Total codes at runtime num_residuals
        nn_total = len(nn_key_names) * num_residuals * code_len
        rule_total = len(rule_key_names) * rule_tokens_per_key
        code_total = nn_total + rule_total
        n_runtime_markers = len(nn_key_names) * num_residuals + len(rule_key_names)
        super().__init__(total_codes=code_total + n_runtime_markers)

    @property
    def total_codes(self):
        return self._total_codes

    def codes_to_action_indices(
        self,
        codes_dict: Dict[str, torch.Tensor],
        noop_keys_per_sample=None,
        num_residuals: Optional[int] = None,
    ) -> ActionIdxBatch:
        flat = codes_dict["codes"]
        if isinstance(flat, torch.Tensor):
            flat = flat.detach().cpu().to(torch.long)

        nr = num_residuals if num_residuals is not None else self.num_residuals
        batch_size = flat.shape[0]

        # Rule section in the flat tensor starts after all nr NN levels.
        nn_rule_base = nr * len(self.nn_key_names) * self.code_len

        nn_order = list(self.nn_key_names)
        rule_order = list(self.rule_key_names)

        result = []
        for b in range(batch_size):
            noop_keys = set()
            if noop_keys_per_sample is not None and b < len(noop_keys_per_sample):
                noop_keys = noop_keys_per_sample[b]

            if self.group_order_shuffle:
                random.shuffle(nn_order)
                random.shuffle(rule_order)

            row = flat[b].tolist()
            indices = []

            for level in range(nr):
                for k in nn_order:
                    entry = self._nn_offsets.get((k, level))
                    if entry is None or k in noop_keys:
                        continue
                    marker_idx, s, e = entry
                    indices.append(marker_idx)
                    indices.extend(row[s:e])

            for k in rule_order:
                entry = self._rule_rel_offsets.get(k)
                if entry is None or k in noop_keys:
                    continue
                marker_idx, rel_s, rel_e = entry
                indices.append(marker_idx)
                indices.extend(row[nn_rule_base + rel_s : nn_rule_base + rel_e])

            result.append(indices)

        return result

    def _marker_str(self, key: str, level: int) -> str:
        """Return marker token string for (key, level). level=-1 for rule keys."""
        if level < 0:
            return f"<{key}>"
        return f"<{key}_{level}>" if self.max_residuals > 1 else f"<{key}>"

    def strip_markers(self, indices: List[int], num_residuals: Optional[int] = None) -> List[int]:
        marker_values = set(self.group_marker_action_indices.values())
        return [i for i in indices if i not in marker_values]

    def parse_per_key_codes(
        self,
        indices: List[int],
        num_residuals: Optional[int] = None,
    ) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[int]]]:
        """Parse a marker-interspersed token sequence into per-key codes (single sample).

        Returns:
            nn_present:   {key: [[level0_codes], [level1_codes], ...]} for present NN keys
            rule_present: {key: [rule_token_ids]}                      for present rule keys
        Absent keys are not included in either dict.
        """
        nr = num_residuals if num_residuals is not None else self.num_residuals
        # Reverse maps: marker_idx → (key, level) or → key
        nn_marker_map: Dict[int, Tuple[str, int]] = {
            mid: (k, lvl) for (k, lvl), (mid, _s, _e) in self._nn_offsets.items() if lvl < nr
        }
        rule_marker_map: Dict[int, str] = {
            mid: k for k, (mid, _rs, _re) in self._rule_rel_offsets.items()
        }

        nn_per_level: Dict[Tuple[str, int], List[int]] = {}
        rule_per_key: Dict[str, List[int]] = {}

        i = 0
        while i < len(indices):
            tok = indices[i]
            if tok in nn_marker_map:
                key, level = nn_marker_map[tok]
                raw = list(indices[i + 1 : i + 1 + self.code_len])
                if len(raw) < self.code_len:
                    raw += [0] * (self.code_len - len(raw))
                nn_per_level[(key, level)] = raw
                i += 1 + self.code_len
            elif tok in rule_marker_map:
                key = rule_marker_map[tok]
                raw = list(indices[i + 1 : i + 1 + self.rule_tokens_per_key])
                if len(raw) < self.rule_tokens_per_key:
                    raw += [0] * (self.rule_tokens_per_key - len(raw))
                rule_per_key[key] = raw
                i += 1 + self.rule_tokens_per_key
            else:
                i += 1

        nn_present: Dict[str, List[List[int]]] = {}
        for k in self.nn_key_names:
            levels = [nn_per_level.get((k, lvl)) for lvl in range(nr)]
            if any(lc is not None for lc in levels):
                nn_present[k] = [lc if lc is not None else [0] * self.code_len for lc in levels]

        return nn_present, rule_per_key


class PartSignSerializer:
    """Multi-part backend: per-part codes + sign tokens → interleaved sequence.

    For WBC: each part (left_arm, right_arm, lower_body) gets a sign token prefix per block.
    The interleaving order is level-major: for each RVQ level, emit all parts' blocks.
    """

    def __init__(
        self,
        part_names: List[str],
        code_parts: Dict[str, int],
        num_level: int,
        num_block_per_part: Dict[str, int],
        block_size: int,
        *,
        lower_first: bool = False,
        part_order_shuffle: bool = False,
        use_indicator_tokens: bool = False,
    ):
        self.part_names = part_names
        self.code_parts = code_parts
        self.num_level = num_level
        self.num_block_per_part = num_block_per_part
        self._raw_block_size = block_size  # from backend, without sign token
        self.block_size = block_size + 1  # +1 for sign token prefix
        self.lower_first = lower_first
        self.part_order_shuffle = part_order_shuffle
        self.use_indicator_tokens = use_indicator_tokens

        # Compute num_block_per_level for each part
        self.num_block_per_level_dict = {k: num_block_per_part[k] // num_level for k in part_names}
        self.num_block_total = sum(num_block_per_part.values())

        # Internal state: populated after init_tokens
        self.sign_tokens: Dict[str, List[int]] = {}
        self.type_to_action_index: Dict[str, int] = {}
        self._not_use_action_index: int = 0
        # Mapping from aggregate key (regex group) to part name
        self._aggregate_key_to_vq_key: Dict[str, str] = {}

    def codes_to_action_indices(
        self, codes_dict: Dict[str, torch.Tensor], **kwargs
    ) -> ActionIdxBatch:
        """Convert per-part codes dict to action index batch with sign tokens.

        For each batch sample, interleave: [<part_sign> codes...] per part.
        When ``part_order_shuffle`` is enabled, part order is randomized per sample.
        When ``use_indicator_tokens`` is enabled, indicator prefix is prepended.
        """
        # Determine batch size
        first_key = next(k for k in codes_dict if k != "_repr_info")
        batch_size = codes_dict[first_key].shape[0]

        text_token_ids: ActionIdxBatch = []
        for b in range(batch_size):
            # Determine available parts for this sample
            available: List[str] = []
            for k in self.part_names:
                if k not in codes_dict:
                    continue
                if (codes_dict[k][b] == self._not_use_action_index).any():
                    continue
                available.append(k)

            # Per-sample shuffle
            if self.part_order_shuffle:
                order = list(available)
                random.shuffle(order)
            else:
                order = available

            codes: List[int] = []

            # Indicator prefix: <ind_part1> <ind_part2> ... <bos_blk>|<sep_ind>
            if self.use_indicator_tokens:
                for name in order:
                    codes.append(int(self._ind_action_indices[name]))
                # BAR mode reuses <bos_blk> as the separator; non-BAR uses <sep_ind>.
                sep_id = (
                    self._bos_blk_action_index
                    if self._bos_blk_action_index is not None
                    else self._sep_ind_action_index
                )
                codes.append(int(sep_id))

            # Action segment (same logic, using shuffled order)
            for k in order:
                part_sign = int(self.type_to_action_index[k])
                codes.extend([part_sign] + codes_dict[k][b].detach().cpu().to(torch.long).tolist())

            text_token_ids.append(codes)
        return text_token_ids

    def set_decoding_order(
        self,
        parts_strs: Dict[str, Optional[str]],
        tokenizer: Any,
        *,
        order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Set decoding order: add sign token prefix to each block, interleave by level.

        Args:
            parts_strs: {part_name: encoded_str_or_None}
            tokenizer: Main tokenizer for encode/decode.
            order: Explicit part order. If None, uses default (lower_first or part_names).

        Returns:
            1D tensor of interleaved token IDs with sign prefixes.
        """
        if order is not None:
            effective_order = [n for n in order if n in parts_strs]
        elif self.lower_first:
            effective_order = ["lower_body", "left_arm", "right_arm"]
        else:
            effective_order = list(self.part_names)
        order = effective_order

        blocks = []
        for name in order:
            tok_str = parts_strs.get(name)
            if tok_str is None:
                continue
            tok = tokenizer.encode(tok_str, add_special_tokens=False)
            n_blk = self.num_block_per_level_dict[name]
            code_per_block = len(tok) // self.num_level // n_blk
            tok = torch.tensor(tok).reshape(self.num_level, n_blk, code_per_block)
            tok_sign = torch.tensor(self.sign_tokens[name], device=tok.device).reshape(
                self.num_level, n_blk, 1
            )
            tok = torch.cat([tok_sign, tok], dim=-1)
            blocks.append(tok.flatten(1))

        return torch.cat(blocks, dim=-1).reshape(-1).contiguous()

    def strip_indicator_prefix(self, action_str: str) -> str:
        """Strip indicator prefix from action string, return the action body.

        If ``use_indicator_tokens`` is False or no separator found, returns the
        original string unchanged.
        """
        if not self.use_indicator_tokens:
            return action_str
        sep_token = "<bos_blk>" if self._bos_blk_action_index is not None else "<sep_ind>"
        sep_pos = action_str.find(sep_token)
        if sep_pos == -1:
            return action_str
        return action_str[sep_pos + len(sep_token) :]

    def build_block_io(
        self,
        action_str: str,
        ar_block_num: int,
        tokenizer: Any,
        action_template: str,
    ) -> Tuple[List[int], List[float], List[int]]:
        """Build block-wise AR IO for WBC (sign-token interleaved).

        When indicator tokens are present in ``action_str``, they are placed in the
        direct-AR prefix (standard causal attention) before the BAR blocks.

        MRoPE-aware action position_ids constraint, used by the BAR branch in
        g05/models/g05/mask_helper.py MaskHelperQwen35._build_mrope_position_ids:
            **block_size must equal one codebook size (code_per_block)**.
            This constraint makes residual_idx = block_idx % num_residuals meaningful
            in the sequence: one entire BAR block corresponds to one residual layer.
            If block size crosses residual boundaries, H-axis residual_idx encoding
            becomes invalid because codes from different codebooks are treated as
            adjacent positions inside the same codebook.
            Data/config must guarantee this precondition, e.g. by grouping multiple
            parts into one block. No runtime assert is performed here.
        """
        action_body = self.strip_indicator_prefix(action_str)
        indicator_ids: List[int] = []
        if self.use_indicator_tokens and action_body != action_str:
            sep_token = "<bos_blk>" if self._bos_blk_action_index is not None else "<sep_ind>"
            sep_pos = action_str.find(sep_token)
            ind_prefix = action_str[: sep_pos + len(sep_token)]
            indicator_ids = tokenizer.encode(ind_prefix.strip(), add_special_tokens=False)

        # Order-agnostic split
        parts_strs, detected_order = extract_parts_from_str(action_body, self.part_names)

        # Override the default order with detected_order only in shuffle/indicator
        # mode; otherwise preserve existing behavior determined by lower_first or
        # part_names.
        explicit_order = detected_order if self.part_order_shuffle else None
        decoding_ids = self.set_decoding_order(parts_strs, tokenizer, order=explicit_order)
        action_ids = (
            decoding_ids.tolist() if isinstance(decoding_ids, torch.Tensor) else list(decoding_ids)
        )

        blocks = []
        for i in range(self.num_block_total):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            blk = action_ids[start_idx:end_idx]
            blocks.append(blk)

        input_ids: List[int] = []
        attn_mask: List[float] = []
        labels: List[int] = []

        # Indicator prefix (direct-AR, standard causal attention)
        if indicator_ids:
            input_ids.extend(indicator_ids)
            attn_mask.extend([ACTION_TOKEN_INDEX] * len(indicator_ids))
            labels.extend(indicator_ids)

        for i in range(ar_block_num):
            input_ids.extend(blocks[i])
            attn_mask.extend([ACTION_TOKEN_INDEX] * self.block_size)
            labels.extend(blocks[i])
        blocks = blocks[ar_block_num:]

        if len(blocks) > 0:
            input_ids.extend([self._bos_blk_id] * self.block_size)
            labels.extend([self._bos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX + 0.01] * self.block_size)
            for i, blk in enumerate(blocks):
                input_ids.extend(blk)
                labels.extend(blk)
                attn_mask.extend([ACTION_TOKEN_INDEX + 0.01 * (i + 2)] * len(blk))
            input_ids.extend([self._eos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX])
            labels.extend([self._eos_blk_id] * self.block_size)

        input_str = tokenizer.decode(input_ids)
        label_str = tokenizer.decode(labels)

        input_str = action_template.format(action=input_str, eos=tokenizer.eos_token)
        label_str = action_template.format(action=label_str, eos=tokenizer.eos_token)
        input_ids_full = tokenizer.encode(input_str, add_special_tokens=False)
        labels_full = tokenizer.encode(label_str, add_special_tokens=False)

        start = _find_subsequence(input_ids_full, input_ids)
        if start < 0:
            attn_mask_full = [ACTION_TOKEN_INDEX] * len(input_ids_full)
        else:
            prefix_num = input_ids_full[:start]
            suffix_num = input_ids_full[start + len(input_ids) :]
            attn_mask_full = (
                [ACTION_TOKEN_INDEX] * len(prefix_num)
                + attn_mask
                + [ACTION_TOKEN_INDEX] * len(suffix_num)
            )

        assert len(input_ids_full) == len(labels_full) == len(attn_mask_full)
        return input_ids_full, attn_mask_full, labels_full


# ============================================================================
# Utility functions migrated from vq_wbc_utils.py
# ============================================================================


def extract_parts_from_str(
    s: str,
    tag_names: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """Split a string by part marker tags, supporting arbitrary order.

    Unlike ``_split_optional_tags``, this function does not assume tag order.
    It locates each tag and sorts by position to split correctly.

    Args:
        s: string containing part marker tags
        tag_names: list of part names, e.g. ["left_arm", "right_arm", "lower_body"]

    Returns:
        (parts_dict, detected_order)
        parts_dict: {part_name: content_str}
        detected_order: actual order of tags as they appear in the string
    """
    tags = []
    for name in tag_names:
        tag = f"<{name}_blk>"
        idx = s.find(tag)
        if idx != -1:
            tags.append((idx, name, tag))
    tags.sort(key=lambda x: x[0])

    result: Dict[str, str] = {}
    detected_order: List[str] = []
    for i, (pos, name, tag) in enumerate(tags):
        start = pos + len(tag)
        end = tags[i + 1][0] if i + 1 < len(tags) else len(s)
        content = s[start:end].strip()
        if content:
            result[name] = content
        detected_order.append(name)
    return result, detected_order


def _aggregate_actions(s: str) -> Dict[str, str]:
    """Regex-based aggregation of sign-tagged action tokens."""
    pattern = re.compile(
        r"<((right_arm|left_arm|lower_body)(?:_[a-z]+)?)_\d+>(.*?)(?=<right_arm|<left_arm|<lower_body|$)"
    )
    grouped = {}
    for full_cat, base_cat, acts in pattern.findall(s):
        grouped[base_cat] = grouped.get(base_cat, "") + acts
    return grouped if grouped else {}

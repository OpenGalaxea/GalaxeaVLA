# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import math
from typing import Any, List, Optional, Tuple


ACTION_TOKEN_INDEX = 3


class BARBuilder:
    """Block-wise AR training data formatter, independent of VQActionTokenizer."""

    def __init__(
        self,
        tokenizer: Any,
        block_wise: bool = False,
        block_size: Optional[int] = None,
        bos_blk_id: Optional[int] = None,
        eos_blk_id: Optional[int] = None,
        pad_action_id: Optional[int] = None,
        ar_block_num: int = 0,
        action_template: str = "Action: {action}|",
        num_block: Optional[int] = None,
        serializer: Any = None,
    ):
        self.tokenizer = tokenizer
        self.block_wise = block_wise
        self.block_size = block_size
        self.bos_blk_id = bos_blk_id
        self.eos_blk_id = eos_blk_id
        self.pad_action_id = pad_action_id
        self.ar_block_num = ar_block_num
        self.action_template = action_template
        self.num_block = num_block
        self.serializer = serializer

    def build_io(self, action_str: str) -> Tuple[List[int], List[float], List[int]]:
        """Unified entry point that dispatches by block_wise."""
        if self.block_wise:
            if self.serializer is not None and hasattr(self.serializer, "build_block_io"):
                return self.serializer.build_block_io(
                    action_str,
                    self.ar_block_num,
                    self.tokenizer,
                    self.action_template,
                )
            return self._build_block_io(action_str)
        return self._build_ar_io(action_str)

    def build_io_from_ids(self, action_ids: List[int]) -> Tuple[List[int], List[float], List[int]]:
        """Build IO directly from token ids, avoiding text round-trip.

        Use this when use_extra_tokens=False to prevent BPE re-segmentation
        errors that can corrupt action sequences.
        """
        if self.block_wise:
            return self._build_block_io_from_ids(action_ids)
        return self._build_ar_io_from_ids(action_ids)

    def _build_ar_io_from_ids(self, action_ids: List[int]):
        prefix_ids = self.tokenizer.encode(
            self.action_template.split("{action}")[0], add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            self.action_template.split("{action}")[-1], add_special_tokens=False
        )
        input_ids = prefix_ids + list(action_ids) + suffix_ids
        attn_mask = [ACTION_TOKEN_INDEX] * len(input_ids)
        labels = list(input_ids)
        return input_ids, attn_mask, labels

    def _build_block_io_from_ids(self, action_ids: List[int]):
        if self.num_block is None:
            num_block = int(math.ceil(len(action_ids) / self.block_size))
        else:
            num_block = int(self.num_block)

        blocks: List[List[int]] = []
        for i in range(num_block):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            blk = action_ids[start_idx:end_idx]
            if len(blk) != self.block_size:
                blk.extend([self.pad_action_id] * (self.block_size - len(blk)))
            blocks.append(blk)

        input_ids: List[int] = []
        attn_mask: List[float] = []
        labels: List[int] = []

        for i in range(min(self.ar_block_num, len(blocks))):
            input_ids.extend(blocks[i])
            attn_mask.extend([ACTION_TOKEN_INDEX] * self.block_size)
            labels.extend(blocks[i])
        blocks = blocks[self.ar_block_num :]

        if len(blocks) > 0:
            input_ids.extend([self.bos_blk_id] * self.block_size)
            labels.extend([self.bos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX + 0.01] * self.block_size)

            for i, blk in enumerate(blocks):
                input_ids.extend(blk)
                labels.extend(blk)
                attn_mask.extend([ACTION_TOKEN_INDEX + 0.01 * (i + 2)] * len(blk))

            input_ids.extend([self.eos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX])
            labels.extend([self.eos_blk_id] * self.block_size)

        prefix_ids = self.tokenizer.encode(
            self.action_template.split("{action}")[0], add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            self.action_template.split("{action}")[-1], add_special_tokens=False
        )

        input_ids_full = prefix_ids + input_ids + suffix_ids
        labels_full = prefix_ids + labels + suffix_ids
        attn_mask_full = (
            [ACTION_TOKEN_INDEX] * len(prefix_ids)
            + attn_mask
            + [ACTION_TOKEN_INDEX] * len(suffix_ids)
        )

        assert len(input_ids_full) == len(labels_full) == len(attn_mask_full), (
            f"input_ids: {len(input_ids_full)}, labels: {len(labels_full)}, "
            f"attn_mask: {len(attn_mask_full)} should be equal!"
        )

        return input_ids_full, attn_mask_full, labels_full

    def _build_ar_io(self, action_str: str):
        action_str = self.action_template.format(action=action_str)
        action_tokens = self.tokenizer(action_str)["input_ids"]
        attn_mask = [ACTION_TOKEN_INDEX] * len(action_tokens)
        labels = list(action_tokens)
        input_ids = list(action_tokens)
        return input_ids, attn_mask, labels

    def _build_block_io(self, action_str: str, action_template: Optional[str] = None):
        action_ids = self.tokenizer.encode(action_str, add_special_tokens=False)

        if self.num_block is None:
            num_block = int(math.ceil(len(action_ids) / self.block_size))
        else:
            num_block = int(self.num_block)

        blocks: List[List[int]] = []
        for i in range(num_block):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            blk = action_ids[start_idx:end_idx]
            if len(blk) != self.block_size:
                blk.extend([self.pad_action_id] * (self.block_size - len(blk)))
            blocks.append(blk)

        input_ids: List[int] = []
        attn_mask: List[float] = []
        labels: List[int] = []

        for i in range(min(self.ar_block_num, len(blocks))):
            input_ids.extend(blocks[i])
            attn_mask.extend([ACTION_TOKEN_INDEX] * self.block_size)
            labels.extend(blocks[i])
        blocks = blocks[self.ar_block_num :]

        if len(blocks) > 0:
            input_ids.extend([self.bos_blk_id] * self.block_size)
            labels.extend([self.bos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX + 0.01] * self.block_size)

            for i, blk in enumerate(blocks):
                input_ids.extend(blk)
                labels.extend(blk)
                attn_mask.extend([ACTION_TOKEN_INDEX + 0.01 * (i + 2)] * len(blk))

            input_ids.extend([self.eos_blk_id])
            attn_mask.extend([ACTION_TOKEN_INDEX])
            labels.extend([self.eos_blk_id] * self.block_size)

        input_str = self.tokenizer.decode(input_ids)
        label_str = self.tokenizer.decode(labels)

        if action_template is None:
            action_template = self.action_template
        input_str = action_template.format(action=input_str)
        label_str = action_template.format(action=label_str)

        input_ids_full = self.tokenizer.encode(input_str, add_special_tokens=False)
        labels_full = self.tokenizer.encode(label_str, add_special_tokens=False)

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

        assert len(input_ids_full) == len(labels_full) == len(attn_mask_full), (
            f"input_ids: {len(input_ids_full)}, labels: {len(labels_full)}, "
            f"attn_mask: {len(attn_mask_full)} should be equal!"
        )

        return input_ids_full, attn_mask_full, labels_full


def make_bar_builder(action_tokenizer: Any, tokenizer: Any) -> "BARBuilder":
    """Construct a BARBuilder from an action tokenizer and HF tokenizer.

    Shared factory used by all processor classes to avoid duplication.
    """
    bar_cfg = action_tokenizer.bar_config
    if bar_cfg is not None:
        dynamic_num_block = bool(getattr(action_tokenizer, "dropout_noop_parts", False))
        return BARBuilder(
            tokenizer=tokenizer,
            block_wise=True,
            block_size=bar_cfg["block_size"],
            bos_blk_id=bar_cfg["bos_blk_id"],
            eos_blk_id=bar_cfg["eos_blk_id"],
            pad_action_id=bar_cfg["pad_action_id"],
            ar_block_num=bar_cfg["ar_block_num"],
            action_template=bar_cfg["action_template"],
            num_block=None if dynamic_num_block else bar_cfg.get("num_block"),
            serializer=getattr(action_tokenizer, "serializer", None),
        )
    return BARBuilder(
        tokenizer=tokenizer,
        block_wise=False,
        action_template=action_tokenizer.action_template,
    )


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1

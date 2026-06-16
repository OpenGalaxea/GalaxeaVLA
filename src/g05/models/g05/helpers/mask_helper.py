# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0 AND Apache-2.0
# Copyright (c) 2026 Galaxea
# Copyright 2024 Big Vision Authors.
# Portions are adapted from OpenPI / Google Research big_vision.
# See THIRD_PARTY_NOTICES.md for the upstream Apache-2.0 notice.

"""
MaskHelper: G05 attention mask + position ID builder.

Weight-free pure computation extracted from G05Model while preserving the
interface. Includes VLM block-wise causal masks, Action Expert masks, four
position_ids strategies, and attention mask visualization.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch

from ..io.input_preprocessor import TOKEN_INDEX

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Block-wise causal mask (adapted from Google Research big_vision;
# see THIRD_PARTY_NOTICES.md for the upstream notice)
# ------------------------------------------------------------------


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Block-wise causal mask.

    mask_ar[i]=1 → token i starts a new causal block.
    mask_ar[i]=0 → token i shares its block with the previous token (bidirectional).

    Args:
        input_mask: bool[B, N] — true if valid input, false if padding.
        mask_ar: bool[?B, N] — true where a new causal block starts.
    Returns:
        attn_mask: bool[B, N, N]
    """
    mask_ar = torch.broadcast_to(mask_ar, input_mask.shape)
    cumsum = torch.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return attn_mask & valid_mask


# ------------------------------------------------------------------
# MaskHelper
# ------------------------------------------------------------------


class MaskHelper:
    """G05 attention mask + position ID builder (weight-free pure computation).

    Attributes:
        position_ids_type: one of "lyc" | "pi0fast" | "gaussian"
        action_position_offset: fixed offset, or None for dynamic computation
    """

    def __init__(self, cfg):
        self.position_ids_type = cfg.position_ids_type
        self.action_position_offset = cfg.action_position_offset
        self._training_mode = False

    # ------------------------------------------------------------------
    # VLM: block-wise causal mask + position_ids
    # ------------------------------------------------------------------

    def build_vlm_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        kv_len: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """VLM block-wise causal mask + position_ids.

        Keeps all four position_ids_type modes: lyc, pi0fast, lycv2, gaussian.

        Args:
            input_ids: [B, S]
            attention_mask: [B, S] containing TOKEN_INDEX encoding
            kv_len: KV cache length (0 = prefill)
            dtype: mask dtype
        Returns:
            (causal_mask, position_ids)
            causal_mask: [B, 1, Q, KV]
            position_ids: [B, Q] (0-indexed)
        """
        device = attention_mask.device
        bsz = attention_mask.size(0)
        q_len = input_ids.size(1)
        min_value = torch.finfo(dtype).min

        if kv_len == 0:
            # ============================================================
            # Prefill (kv_len == 0): use the same path as training.
            # ============================================================
            # BAR attention convention (labels from training build_block_io):
            #   <bos_blk> × block_size         -> 3.01  (block #0, bidirectional inside block)
            #   code block 0 × block_size      -> 3.02  (block #1, bidirectional inside block)
            #   code block 1 × block_size      -> 3.03  (block #2, bidirectional inside block)
            #   ...
            #   <eos_blk> × 1                  -> 3     (standard causal AR)
            #   other action tokens (non-BAR)  -> 3     (standard causal AR)
            #
            # The floor==3 and !=3 branch below groups fractional labels into
            # bidirectional block masks.
            # Important invariant: any action token processed by prefill (kv_len=0)
            # must carry correct 3.0x labels from training-side build_block_io. If
            # ar_helper._assign_token_index is used, it only returns integer 3; BAR
            # block recognition fails, intra-block bidirectional attention degrades
            # to standard causal AR, and outputs silently diverge from training.
            #
            # The current inference path (g05_policy.prefill + encode_inference(ar))
            # guarantees that prefill input_ids contain no action tokens and stop at
            # <EOC>. All action tokens are generated in the decode branch (kv_len>0).
            # If future code wants prefill to handle action tokens for re-scoring or
            # speculative decoding, it must use build_block_io to generate a mask with
            # 3.0x labels.
            # ============================================================
            input_mask = attention_mask > 0
            mask_ar = torch.zeros_like(input_mask).float()

            # Block-wise autoregressive for sub-block action tokens
            block_action_mask = (torch.floor(attention_mask) == TOKEN_INDEX.ACTION_TOKEN_INDEX) & (
                attention_mask != TOKEN_INDEX.ACTION_TOKEN_INDEX
            )
            if block_action_mask.any():
                block_vals = attention_mask[block_action_mask]
                _, inverse_indices = torch.unique(block_vals, sorted=True, return_inverse=True)
                mask_ar[block_action_mask] = (inverse_indices + 2).float()
            mask_ar[:, 1:] = mask_ar[:, 1:] != mask_ar[:, :-1]

            # Standard autoregressive tokens
            mask_ar[attention_mask == TOKEN_INDEX.ACTION_TOKEN_INDEX] = 1
            mask_ar[attention_mask == TOKEN_INDEX.COT_TOKEN_INDEX] = 1
            mask_ar[attention_mask == TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX] = 1

            causal_mask_bool = ~make_attn_mask(input_mask, mask_ar).bool()
            causal_mask = causal_mask_bool.long() * min_value
        else:
            # ============================================================
            # KV cache decode (kv_len > 0): inference-only path.
            # ============================================================
            # This branch only checks attention_mask == 0 (padding), ignoring exact
            # values. Therefore integer 3 from _assign_token_index for action tokens
            # is OK in decode, even without 3.0x fractional labels. BAR block
            # bidirectional visibility is implemented implicitly by _infer_single
            # feeding block_size query tokens in one forward pass: the q-to-q area of
            # causal_mask remains 0 (unmasked), so it is naturally bidirectional.
            # See the BAR branch in ar_helper._infer_single.
            # ============================================================
            _causal_mask = (attention_mask == 0) * min_value
            causal_mask = torch.full((bsz, q_len, kv_len + q_len), 0, dtype=dtype, device=device)
            for q in range(q_len):
                causal_mask[:, q, :kv_len] = _causal_mask[:, :kv_len]

        # [B, Q, KV] → [B, 1, Q, KV]
        causal_mask = causal_mask.unsqueeze(1)

        # Position IDs (0-indexed for PaliGemma compatibility)
        position_ids = self._compute_position_ids(attention_mask) - 1

        if kv_len > 0:
            position_ids = position_ids[:, -q_len:]

        return causal_mask, position_ids

    # ------------------------------------------------------------------
    # Position IDs (internal)
    # ------------------------------------------------------------------

    def compute_position_ids(
        self,
        attention_mask: torch.LongTensor,
    ) -> torch.LongTensor:
        """0-indexed VLM position_ids for a given attention_mask.

        Uses the same _compute_position_ids logic as the training-side build_vlm_mask
        kv_len=0 branch, then subtracts 1. Calling this after generate_text can
        recompute state.position_ids and keep train/inference RoPE positions aligned,
        especially for the action position offset in the predict_cot + FM path.
        """
        return self._compute_position_ids(attention_mask) - 1

    def _compute_position_ids(
        self,
        attention_mask: torch.LongTensor,
    ) -> torch.LongTensor:
        """Compute position_ids by position_ids_type, before subtracting 1."""
        if self.position_ids_type == "pi0fast":
            return (attention_mask > 0).masked_fill_((attention_mask == 0), 0).cumsum(-1)
        elif self.position_ids_type == "lyc":
            return attention_mask.cumsum(-1).masked_fill_((attention_mask == 0), 1)
        elif self.position_ids_type == "gaussian":
            if self._is_training:
                from torch.distributions import Normal

                normal_dist = Normal(2.0, 0.5)
                gaussian_samples = normal_dist.sample(attention_mask.shape).to(
                    attention_mask.device
                )
                offset = torch.clamp(gaussian_samples, 1, 3).round().long()
                offset = offset.masked_fill_(
                    (attention_mask == TOKEN_INDEX.IMAGE_TOKEN_INDEX) | (attention_mask == 0), 0
                ).cumsum(-1)
            else:
                offset = (
                    torch.full_like(attention_mask, 2)
                    .masked_fill_(
                        (attention_mask == TOKEN_INDEX.IMAGE_TOKEN_INDEX) | (attention_mask == 0), 0
                    )
                    .cumsum(-1)
                )
            return ((attention_mask > 0).cumsum(-1) + offset).masked_fill_((attention_mask == 0), 1)
        else:
            raise ValueError(f"Invalid position_ids_type: {self.position_ids_type}")

    @property
    def _is_training(self) -> bool:
        """Injected from external model.training, defaulting to False."""
        return self._training_mode

    @_is_training.setter
    def _is_training(self, value: bool):
        self._training_mode = value

    # ------------------------------------------------------------------
    # Action Expert: prefix-attend + self-attend mask
    # ------------------------------------------------------------------

    def build_action_mask(
        self,
        attention_mask_prefix: torch.Tensor,
        action_len: int,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        split_index: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        action_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Action Expert mask + position_ids.

        Action tokens attend to all VLM non-pad tokens + all action tokens.
        Dynamic offset (D2): action_pos = arange(1, H+1) + max(vlm_pos[:split_index]) + 1

        Args:
            attention_mask_prefix: [B, S_prefix] prefix attention mask
            action_len: H (action horizon)
            position_ids_prefix: [B, S] VLM position_ids (0-indexed)
            split_index: prefix end position
            dtype: mask dtype
            action_causal: whether to apply causal masking to action tokens
        Returns:
            (action_mask, action_position_ids)
            action_mask: [B, 1, H, S_prefix + H]
            action_position_ids: [B, H]
        """
        bsz = attention_mask_prefix.size(0)
        device = attention_mask_prefix.device

        # Action attends to all other action tokens
        if action_causal:
            action_self_mask = (
                torch.triu(
                    torch.full(
                        (action_len, action_len), torch.finfo(dtype).min, dtype=dtype, device=device
                    ),
                    diagonal=1,
                )
                .unsqueeze(0)
                .expand(bsz, -1, -1)
            )
        else:
            action_self_mask = torch.zeros(
                (bsz, action_len, action_len),
                dtype=dtype,
                device=device,
            )

        # VLM prefix: mask padding
        _prefix_mask = (attention_mask_prefix == 0) * torch.finfo(dtype).min
        _prefix_mask = _prefix_mask[:, None].expand(-1, action_len, -1)

        # [B, H, S_prefix + H]
        causal_mask = torch.cat([_prefix_mask, action_self_mask], dim=-1)
        # [B, 1, H, S_prefix + H]
        causal_mask = causal_mask.unsqueeze(1)

        # Position IDs: arange(1, H+1) + offset
        action_position_ids = (
            torch.arange(
                1,
                action_len + 1,
                device=device,
            )
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        # Dynamic offset from VLM position_ids
        if self.action_position_offset is not None:
            offset = self.action_position_offset
        elif position_ids_prefix is not None and split_index is not None:
            # position_ids_prefix: [B, S] for paligemma and [3, B, S] for qwen3.5
            offset = position_ids_prefix[..., :split_index].max(-1, keepdim=True)[0]
        else:
            offset = 0

        action_position_ids = action_position_ids + offset

        return causal_mask, action_position_ids

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def visualize(
        mask: torch.Tensor,
        title: str = "",
        token_labels: Optional[List[str]] = None,
        batch_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8),
        show: bool = True,
    ):
        """Render attention mask as a heatmap.

        Args:
            mask: [B, 1, Q, KV] or [B, Q, KV], containing 0 (attend) and min_value (masked)
            title: plot title
            token_labels: token labels for Q and KV axes
            batch_idx: batch sample to render
            save_path: save path, or None to skip saving
            figsize: plot size
            show: whether to call plt.show()
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Squeeze to [Q, KV]
        m = mask[batch_idx].detach().cpu().float()
        if m.dim() == 3:
            m = m[0]  # remove head dim
        # Convert: 0 → 1 (attend), min_value → 0 (masked)
        attend_matrix = (m == 0).float().numpy()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(attend_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

        ax.set_xlabel("Key / Value")
        ax.set_ylabel("Query")
        ax.set_title(title or "Attention Mask (blue=attend, white=masked)")

        if token_labels is not None:
            q_len, kv_len = attend_matrix.shape
            if len(token_labels) == kv_len:
                ax.set_xticks(range(kv_len))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
            if len(token_labels) == q_len:
                ax.set_yticks(range(q_len))
                ax.set_yticklabels(token_labels, fontsize=6)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Mask visualization saved to {save_path}")

        if show:
            plt.show()

        return fig

    @staticmethod
    def visualize_pair(
        vlm_mask: torch.Tensor,
        action_mask: torch.Tensor,
        vlm_title: str = "VLM Mask",
        action_title: str = "Action Expert Mask",
        batch_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (20, 8),
    ):
        """Visualize VLM mask and Action mask side by side.

        Args:
            vlm_mask: [B, 1, Q_vlm, KV_vlm]
            action_mask: [B, 1, H, S_prefix + H]
            other parameters match visualize
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, mask, title in [
            (axes[0], vlm_mask, vlm_title),
            (axes[1], action_mask, action_title),
        ]:
            m = mask[batch_idx].detach().cpu().float()
            if m.dim() == 3:
                m = m[0]
            attend = (m == 0).float().numpy()
            im = ax.imshow(attend, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            ax.set_xlabel("Key / Value")
            ax.set_ylabel("Query")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Mask pair visualization saved to {save_path}")

        return fig


# ------------------------------------------------------------------
# MaskHelperQwen35 — Qwen3.5 MRoPE + causal IMAGE tokens
# ------------------------------------------------------------------


class MaskHelperQwen35(MaskHelper):
    """Qwen3.5 mask helper with MRoPE position_ids and causal IMAGE tokens.

    Overrides:
    - build_vlm_mask: IMAGE tokens use causal attention (not bidirectional);
      position_ids are 3D MRoPE [3, B, S].
    - build_action_mask: position_ids are 3D [3, B, H].
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._image_grid_thw = None
        self._spatial_merge_size = getattr(cfg, "spatial_merge_size", 2)
        self._mrope_position_deltas = None
        # T-axis cursor at the end of the last prefill, shared by text/image/BAR and
        # used by decode continuation. BAR H/W start from the T anchor, so no separate
        # state is needed.
        self._last_T_pos = None
        # Cache the Normal distribution object to avoid rebuilding it in _sample_step
        # on the gaussian train path.
        from torch.distributions import Normal as _Normal

        self._gauss = _Normal(2.0, 0.5)

    def build_vlm_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        kv_len: int = 0,
        dtype: torch.dtype = torch.float32,
        is_action_block: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Qwen3.5: causal IMAGE tokens + 3D MRoPE position_ids.

        Args:
            is_action_block: effective only during decode (kv_len>0). True means the
                caller is in BAR mode (one forward with q_len=block_size action
                tokens), so block-structured positions are needed: same T/H and
                differentiated by W.
        """
        device = attention_mask.device
        bsz = attention_mask.size(0)
        q_len = input_ids.size(1)
        min_value = torch.finfo(dtype).min

        if kv_len == 0:
            input_mask = attention_mask > 0
            mask_ar = torch.zeros_like(input_mask).float()

            # Block-wise autoregressive
            block_action_mask = (torch.floor(attention_mask) == TOKEN_INDEX.ACTION_TOKEN_INDEX) & (
                attention_mask != TOKEN_INDEX.ACTION_TOKEN_INDEX
            )
            if block_action_mask.any():
                block_vals = attention_mask[block_action_mask]
                _, inverse_indices = torch.unique(block_vals, sorted=True, return_inverse=True)
                mask_ar[block_action_mask] = (inverse_indices + 2).float()
            mask_ar[:, 1:] = mask_ar[:, 1:] != mask_ar[:, :-1]

            # Qwen3.5: ALL non-padding tokens are causal (including IMAGE)
            is_non_pad_non_block = (attention_mask != TOKEN_INDEX.PADDING_TOKEN_INDEX) & (
                ~block_action_mask
            )
            mask_ar[is_non_pad_non_block] = 1

            causal_mask_bool = ~make_attn_mask(input_mask, mask_ar).bool()
            causal_mask = causal_mask_bool.long() * min_value
        else:
            _causal_mask = (attention_mask == 0) * min_value
            causal_mask = torch.full((bsz, q_len, kv_len + q_len), 0, dtype=dtype, device=device)
            for q in range(q_len):
                causal_mask[:, q, :kv_len] = _causal_mask[:, :kv_len]

        causal_mask = causal_mask.unsqueeze(1)

        # 3D MRoPE position_ids
        if kv_len == 0:
            position_ids = self._build_mrope_position_ids(attention_mask, device)
        else:
            # Decode: continue from the T position at the end of prefill.
            # The text-like path increments each new token by step. The BAR block
            # path shares T/H across q_len tokens and differentiates only by W,
            # matching the BAR encoding rule used on the prefill side.
            position_ids = self._mrope_decode_position_ids(
                bsz,
                q_len,
                device,
                is_action_block=is_action_block,
            )

        return causal_mask, position_ids

    def build_action_mask(
        self,
        attention_mask_prefix: torch.Tensor,
        action_len: int,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        split_index: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        action_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Override: return 3D MRoPE action position_ids [3, B, H]."""
        causal_mask, action_pos = super().build_action_mask(
            attention_mask_prefix,
            action_len,
            position_ids_prefix=position_ids_prefix,
            split_index=split_index,
            dtype=dtype,
            action_causal=action_causal,
        )
        # [B, H] -> [3, B, H]
        if action_pos.ndim == 2:
            action_pos = action_pos.unsqueeze(0).expand(3, -1, -1).contiguous()
        return causal_mask, action_pos

    # ------------------------------------------------------------------
    # MRoPE position IDs builder
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-segment position computers
    #
    # Shared convention: cursor is the next available 1D slot along the T axis.
    #   - Each segment starts its own (T, H, W) from cursor.
    #   - Return (block_pos[3, seg_len], next_cursor). next_cursor must be strictly
    #     greater than all T values occupied by the segment to avoid collision with
    #     the next segment.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_bar_block_segment(token_type: float) -> bool:
        """BAR block segment: integer part is ACTION_TOKEN_INDEX and fractional part exists.

        Examples are 3.01, 3.02, ..., produced by serialization.build_block_io.
        """
        act = int(TOKEN_INDEX.ACTION_TOKEN_INDEX)
        return int(token_type) == act and token_type != act

    # Each segment computer returns (T_list, H_list, W_list, next_cursor) as Python
    # lists. The outer batch loop concatenates all segments into one list and creates
    # a single torch.tensor on GPU at the end, eliminating small CUDA kernel launches
    # per segment.

    def _text_like_segment_positions(
        self,
        cursor: int,
        seg_len: int,
        steps: List[int],
    ) -> Tuple[List[int], List[int], List[int], int]:
        """Text-like segment: each token occupies one slot, with T=H=W."""
        T: List[int] = []
        p = cursor
        for s in steps:
            T.append(p)
            p += s
        # H and W alias T, sharing the same list content; torch.tensor will copy later.
        return T, T, T, p

    def _bar_block_segment_positions(
        self,
        cursor: int,
        seg_len: int,
        step: int,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """BAR block segment with image-style anchor-relative encoding.

        - T = cursor, shared inside the block and marking the block boundary
        - H = W = cursor + 0..seg_len-1, relative to the T anchor within the block
        - next_cursor = cursor + step

        This is isomorphic to image segments: image uses T=anchor,
        H=anchor+h_grid, W=anchor+w_grid. BAR uses T=anchor and
        H=W=anchor+within_block because BAR has no 2D space. Across blocks, H/W
        advance together with T, so H is always T+within_offset and tightly bound
        to T, unlike the previous independent hw_cursor accumulation.
        """
        T = [cursor] * seg_len
        HW = list(range(cursor, cursor + seg_len))
        return T, HW, HW, cursor + step

    def _image_segment_positions(
        self,
        cursor: int,
        seg_len: int,
        grid_thw,
        spatial_merge_size: int,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """Image segment: T=anchor for the whole segment, H/W = anchor + grid.

        Aligned with _get_vision_position_ids expansion order:
          - position_height = arange(gh).repeat_interleave(gw * gt), outer h
          - position_width  = arange(gw).repeat(gh * gt), inner w
        Therefore, in the nested (h, w) loop, h is outer and w is inner.
        If seg_len is shorter than the token count implied by the grid due to left
        padding truncation, trim to seg_len.
        """
        gt = int(grid_thw[0].item() if isinstance(grid_thw[0], torch.Tensor) else grid_thw[0])
        gh_raw = int(grid_thw[1].item() if isinstance(grid_thw[1], torch.Tensor) else grid_thw[1])
        gw_raw = int(grid_thw[2].item() if isinstance(grid_thw[2], torch.Tensor) else grid_thw[2])
        gh = gh_raw // spatial_merge_size
        gw = gw_raw // spatial_merge_size
        T: List[int] = []
        H: List[int] = []
        W: List[int] = []
        for _ in range(gt):
            for h in range(gh):
                for w in range(gw):
                    T.append(cursor)
                    H.append(cursor + h)
                    W.append(cursor + w)
        if seg_len < len(T):
            T, H, W = T[:seg_len], H[:seg_len], W[:seg_len]
        next_cursor = cursor + max(gh_raw, gw_raw) // spatial_merge_size
        return T, H, W, next_cursor

    def _build_mrope_position_ids(
        self,
        attention_mask: torch.LongTensor,
        device: torch.device,
    ) -> torch.LongTensor:
        """Build 3D MRoPE position_ids [3, B, S].

        For each batch item, split attention_mask into segments with groupby and
        dispatch each segment to the text-like, image, or BAR block position
        computer. The shared cursor advances along the T axis; each segment decides
        how many slots it occupies. See the _*_segment_positions comments.
        """
        import itertools

        bsz, seq_len = attention_mask.shape
        position_ids = torch.zeros(3, bsz, seq_len, dtype=torch.long, device=device)
        image_grid_thw = self._image_grid_thw
        spatial_merge_size = self._spatial_merge_size
        mrope_position_deltas = []

        # ----- One-shot preprocessing: remove repeated GPU->CPU syncs inside loops -----
        # Pull the entire attention_mask to CPU once; previously this did tolist per batch.
        attn_cpu = attention_mask.tolist()  # List[List[float]]
        # Upper bound of steps needed for the whole forward is bsz × seq_len: one
        # sample call and one sync. Training uses Normal; eval/non-gaussian are
        # deterministic and cheap.
        all_steps_t = self._sample_step((bsz, seq_len), device)
        all_steps = all_steps_t.tolist()  # List[List[int]]

        for batch_idx in range(bsz):
            grid_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])
            batch_attn = attn_cpu[batch_idx]
            batch_steps = all_steps[batch_idx]
            step_cursor = 0  # points to the next unused step in batch_steps

            non_padding_mask = attention_mask[batch_idx] != TOKEN_INDEX.PADDING_TOKEN_INDEX
            non_padding_len = sum(1 for v in batch_attn if v != TOKEN_INDEX.PADDING_TOKEN_INDEX)

            # Segment split: keep original float values with BAR fractional parts;
            # int() would collapse 3.01 into 3.
            input_type_group = []
            for key, group in itertools.groupby(enumerate(batch_attn), lambda x: x[1]):
                group = list(group)
                input_type_group.append((float(key), group[0][0], group[-1][0] + 1))

            # Accumulate the entire batch item's (T, H, W) in Python lists, then
            # build one tensor at the end to eliminate small CUDA kernel launches
            # per segment.
            batch_T: List[int] = []
            batch_H: List[int] = []
            batch_W: List[int] = []
            cursor = 0

            for token_type, start_idx, end_idx in input_type_group:
                if token_type == TOKEN_INDEX.PADDING_TOKEN_INDEX:
                    continue
                seg_len = end_idx - start_idx

                if token_type == TOKEN_INDEX.IMAGE_TOKEN_INDEX:
                    grid_thw = next(grid_iter, None)
                    if grid_thw is None:
                        continue
                    T, H, W, cursor = self._image_segment_positions(
                        cursor,
                        seg_len,
                        grid_thw,
                        spatial_merge_size,
                    )
                elif self._is_bar_block_segment(token_type):
                    step = batch_steps[step_cursor]
                    step_cursor += 1
                    T, H, W, cursor = self._bar_block_segment_positions(
                        cursor,
                        seg_len,
                        step,
                    )
                else:
                    steps = batch_steps[step_cursor : step_cursor + seg_len]
                    step_cursor += seg_len
                    T, H, W, cursor = self._text_like_segment_positions(
                        cursor,
                        seg_len,
                        steps,
                    )
                batch_T.extend(T)
                batch_H.extend(H)
                batch_W.extend(W)

            if batch_T:
                # Push all positions for this batch item to GPU in one shot.
                llm_positions = torch.tensor(
                    [batch_T, batch_H, batch_W],
                    dtype=torch.long,
                    device=device,
                )
                position_ids[:, batch_idx, non_padding_mask] = llm_positions
                max_T = max(batch_T)  # pure Python, no sync
                delta = max_T + 1 - non_padding_len
                mrope_position_deltas.append(delta)
            else:
                mrope_position_deltas.append(0)

        self._mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)

        # Cache the last T-axis position for each batch item. Text and image/BAR
        # cursors share the T axis, and decode continues from this point. BAR decode
        # computes H/W directly from the T anchor (image-style), so no separate
        # _last_HW_pos state is needed.
        self._last_T_pos = position_ids[0].max(dim=-1).values  # [B]

        return position_ids

    def _sample_step(
        self,
        shape: Union[Tuple[int, ...], torch.Size],
        device: torch.device,
    ) -> torch.LongTensor:
        """Step sampler shared by text segments, BAR blocks, and decode paths.

        - gaussian + train: N(2, 0.5), clamped to [1, 3] and rounded, using cached Normal
        - gaussian + eval: fixed 2
        - pi0fast / lyc: fixed 1
        """
        if self.position_ids_type == "gaussian":
            if self._is_training:
                return (
                    torch.clamp(
                        self._gauss.sample(shape).to(device),
                        1,
                        3,
                    )
                    .round()
                    .long()
                )
            return torch.full(shape, 2, dtype=torch.long, device=device)
        return torch.ones(shape, dtype=torch.long, device=device)

    def _mrope_decode_position_ids(
        self,
        bsz: int,
        q_len: int,
        device: torch.device,
        is_action_block: bool = False,
    ) -> torch.LongTensor:
        """Continue MRoPE position_ids during decode (kv_len>0).

        Two paths:
        1. text-like (is_action_block=False): each of q_len tokens advances by step,
           with identical values in all three dimensions.
        2. BAR block (is_action_block=True): q_len tokens share T and H, and W
           differentiates within-block positions, following the prefill-side BAR
           encoding rule.

        Returns:
            position_ids: [3, B, q_len]
        """
        if is_action_block:
            # BAR block (image-style):
            #   T = last_T_pos + step, shared inside the block
            #   H = W = T + 0..q_len-1, within-block offsets from the T anchor
            step = self._sample_step((bsz,), device)  # [B]
            T_val = self._last_T_pos[:bsz] + step  # [B]
            W_off = torch.arange(q_len, device=device, dtype=torch.long)
            HW = T_val.unsqueeze(-1) + W_off  # [B, q_len]
            T = T_val.unsqueeze(-1).expand(-1, q_len)  # [B, q_len]
            self._last_T_pos = T_val
            return torch.stack([T, HW, HW], dim=0)  # [3, B, q_len]

        # Text-like path.
        steps = self._sample_step((bsz, q_len), device)
        base = self._last_T_pos[:bsz].unsqueeze(-1)  # [B, 1]
        abs_pos = base + steps.cumsum(-1)  # [B, q_len]
        self._last_T_pos = abs_pos[:, -1]
        return abs_pos.unsqueeze(0).expand(3, -1, -1).contiguous()

    @staticmethod
    def _get_vision_position_ids(
        start_position: int,
        grid_thw,
        spatial_merge_size: int = 1,
        device=None,
    ) -> torch.LongTensor:
        """Compute 3D positional indices for vision tokens. Returns [3, seq_len]."""

        def _val(x):
            return x.item() if isinstance(x, torch.Tensor) else int(x)

        llm_grid_t = _val(grid_thw[0])
        llm_grid_h = _val(grid_thw[1]) // spatial_merge_size
        llm_grid_w = _val(grid_thw[2]) // spatial_merge_size
        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t

        position_width = torch.arange(
            start_position,
            start_position + llm_grid_w,
            device=device,
        ).repeat(llm_grid_h * llm_grid_t)
        position_height = torch.arange(
            start_position,
            start_position + llm_grid_h,
            device=device,
        ).repeat_interleave(llm_grid_w * llm_grid_t)
        position_temporal = torch.full(
            (image_seq_length,),
            start_position,
            device=device,
            dtype=torch.long,
        )

        return torch.stack([position_temporal, position_height, position_width], dim=0)

"""action_accuracy.py

Standalone evaluator for AR-decoded action accuracy and L1 loss.
Decoupled from any policy class — only requires an action_tokenizer instance.

Usage::

    evaluator = ActionAccuracyEvaluator(action_tokenizer)
    metrics = evaluator(
        generated_tokens=decoded_tokens,
        decoded_actions=decoded_actions,
        samples=samples,
        action_gt=gt_actions,
        device=device,
    )
    # metrics == {"action_accuracy": 0.85, "action_l1_loss": 0.012}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)


class ActionAccuracyEvaluator:
    """Compute token-level accuracy and masked L1 loss for AR action predictions.

    Args:
        action_tokenizer: Action tokenizer instance.
        tokenizer: HF tokenizer instance.
        block_wise: Whether BAR is enabled.
        bos_blk_id: BAR begin-of-block token ID (only when BAR enabled).
        eos_blk_id: BAR end-of-block token ID (only when BAR enabled).
    """

    def __init__(
        self, action_tokenizer, tokenizer=None, block_wise=False, bos_blk_id=None, eos_blk_id=None
    ):
        self.action_tokenizer = action_tokenizer
        self._tokenizer = tokenizer
        self._block_wise = block_wise
        self._bos_blk_id = bos_blk_id
        self._eos_blk_id = eos_blk_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        generated_tokens: List[torch.Tensor],
        decoded_actions: List[torch.Tensor],
        samples: List[Dict[str, Any]],
        action_gt: torch.Tensor,
        device: torch.device,
        *,
        absent_keys_per_sample: Optional[List[set[str]]] = None,
    ) -> Dict[str, float]:
        """Evaluate AR action predictions against ground truth.

        Args:
            generated_tokens: Per-sample predicted token id sequences.
            decoded_actions:  Per-sample decoded continuous actions, each ``[H, D]``.
            samples:          Original sample dicts. Each must contain ``"action"``
                              as either a dict (with ``"value"``, ``"action_op_mask"``,
                              ``"action_dim_is_pad"``) or a pretokenized string.
            action_gt:        ``[B, H, D]`` ground-truth continuous actions.
            device:           Compute device.
            absent_keys_per_sample: Optional per-sample sets of absent keys (e.g. dropped
                                    noop parts) whose dims should be excluded from L1.

        Returns:
            ``{"action_accuracy": float, "action_l1_loss": float}``
            ``action_l1_loss`` is ``-1.0`` when no valid sample can be compared.
        """
        pretokenized = self._pretokenize_samples(samples, device=device)
        accuracy = self._token_accuracy(generated_tokens, pretokenized, device)
        l1 = self._masked_l1(
            decoded_actions, pretokenized, action_gt, absent_keys_per_sample=absent_keys_per_sample
        )
        return {"action_accuracy": accuracy, "action_l1_loss": l1}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pretokenize_samples(
        self,
        samples: List[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, Any]]:
        """Tokenize GT actions in *samples*, returning a shallow copy.

        Samples whose ``"action"`` is already a string are left as-is.
        The original action dict is saved under ``"_action_dict_original"``
        for downstream mask retrieval.
        """
        samples_copy = [dict(s) for s in samples]

        need_encode = [
            i
            for i, s in enumerate(samples_copy)
            if "action" in s and not isinstance(s["action"], str)
        ]
        if not need_encode:
            return samples_copy

        action_list = [samples_copy[i]["action"] for i in need_encode]
        action_values = torch.stack([a["value"] for a in action_list], dim=0)
        if device is not None:
            action_values = action_values.to(device)

        encode_kwargs = {}
        if all(a.get("action_op_mask") is not None for a in action_list):
            mask = torch.stack([a["action_op_mask"] for a in action_list], dim=0)
            encode_kwargs["action_op_mask"] = mask.to(device) if device is not None else mask
        if all(a.get("action_dim_is_pad") is not None for a in action_list):
            pad = torch.stack([a["action_dim_is_pad"] for a in action_list], dim=0)
            encode_kwargs["action_dim_is_pad"] = pad.to(device) if device is not None else pad
        parts_meta_list = [a.get("parts_meta") for a in action_list]
        if parts_meta_list and all(pm is not None for pm in parts_meta_list):
            first_parts_meta = parts_meta_list[0]
            if all(pm == first_parts_meta for pm in parts_meta_list[1:]):
                encode_kwargs["input_parts_meta"] = first_parts_meta
            else:
                encode_kwargs["input_parts_meta"] = parts_meta_list

        frequencies = [samples_copy[i].get("frequency") for i in need_encode]
        if frequencies and all(f is not None for f in frequencies):
            encode_kwargs["frequency"] = torch.tensor(
                frequencies, dtype=torch.float32, device=device
            )

        embodiments = [samples_copy[i].get("embodiment") for i in need_encode]
        if embodiments and all(e is not None for e in embodiments):
            encode_kwargs["embodiment"] = embodiments

        pretokenized = self.action_tokenizer(action_values, encode_kwargs=encode_kwargs)
        if self._tokenizer is not None:
            pretokenized = self._tokenizer.batch_decode(pretokenized)
        for batch_pos, idx in enumerate(need_encode):
            samples_copy[idx]["_action_dict_original"] = samples_copy[idx]["action"]
            samples_copy[idx]["action"] = pretokenized[batch_pos]

        return samples_copy

    def _token_accuracy(
        self,
        generated_tokens: List[torch.Tensor],
        samples: List[Dict[str, Any]],
        device: torch.device,
    ) -> float:
        """Token-level accuracy between predicted and GT action tokens."""
        tokenizer = self._tokenizer
        bar = self._block_wise
        is_multi_part = getattr(self.action_tokenizer, "_is_multi_part", False)

        correct = 0.0
        total = 0

        for i, pred in enumerate(generated_tokens):
            gt_action = samples[i].get("action")
            if not isinstance(gt_action, str):
                continue

            gt_str = gt_action
            if is_multi_part and bar:
                # WBC + BAR: pretokenized string uses basic per-part format
                # (<left_arm_blk>codes...<right_arm_blk>codes...) but the model
                # generates in BAR per-block format (<left_arm_blk_0>codes...
                # <left_arm_blk_1>codes...).  Convert GT to BAR format via
                # set_decoding_order so the two are comparable.
                from g05.tokenizer.interface.serialization import (
                    extract_parts_from_str,
                )

                serializer = self.action_tokenizer.serializer
                body = serializer.strip_indicator_prefix(gt_str)
                parts_strs, _ = extract_parts_from_str(body, serializer.part_names)
                bar_ids = serializer.set_decoding_order(parts_strs, tokenizer)
                gt_str = tokenizer.decode(bar_ids.tolist())
            elif bar:
                bos = tokenizer.decode([self._bos_blk_id])
                eos = tokenizer.decode([self._eos_blk_id])
                gt_str = gt_str.replace(bos, "").replace(eos, "")

            gt_tokens = torch.tensor(
                tokenizer.encode(gt_str.strip(), add_special_tokens=False),
                device=device,
            )
            pred_tokens = pred.to(device)
            if pred_tokens.shape[0] >= gt_tokens.shape[0]:
                pred_tokens = pred_tokens[: gt_tokens.shape[0]]
            else:
                pred_tokens = F.pad(
                    pred_tokens, (0, gt_tokens.shape[0] - pred_tokens.shape[0]), value=0
                )

            correct += (pred_tokens == gt_tokens).sum().float().item()
            total += gt_tokens.shape[0]

        return correct / max(total, 1)

    def _masked_l1(
        self,
        decoded_actions: List[torch.Tensor],
        samples: List[Dict[str, Any]],
        action_gt: torch.Tensor,
        *,
        absent_keys_per_sample: Optional[List[set[str]]] = None,
    ) -> float:
        """Masked L1 loss between decoded continuous actions and GT.

        Respects ``action_dim_is_pad`` and ``action_op_mask`` from the
        original action dict (saved as ``"_action_dict_original"``).
        Also excludes dims belonging to absent keys (e.g. dropped noop parts).
        """
        total_l1 = 0.0
        valid_count = 0

        for i, decoded in enumerate(decoded_actions):
            if decoded.shape != action_gt[i].shape:
                logger.warning(
                    "Shape mismatch in AR rollout metric: pred=%s gt=%s",
                    decoded.shape,
                    action_gt[i].shape,
                )
                continue

            pred = decoded.to(action_gt[i])
            gt = action_gt[i]

            action_dict = samples[i].get("_action_dict_original") or samples[i].get("action")
            if isinstance(action_dict, dict):
                dim_is_pad = action_dict.get("action_dim_is_pad")
                op_mask = action_dict.get("action_op_mask")
                parts_meta = action_dict.get("parts_meta")
            else:
                dim_is_pad = None
                op_mask = None
                parts_meta = None

            if dim_is_pad is not None or op_mask is not None:
                valid = torch.ones_like(gt, dtype=torch.bool)
                if dim_is_pad is not None:
                    valid = valid & (~dim_is_pad.to(gt.device).bool()).expand_as(gt)
                if op_mask is not None:
                    valid = valid & op_mask.to(gt.device).bool().unsqueeze(0).expand_as(gt)

                # Exclude absent keys (e.g. gripper dropped by noop_dropout)
                if absent_keys_per_sample is not None and parts_meta is not None:
                    absent = absent_keys_per_sample[i] if i < len(absent_keys_per_sample) else set()
                    if absent:
                        key_order = list(parts_meta.keys())
                        dim_counts = [int(d) for d in parts_meta.values()]
                        offsets = [0]
                        for d in dim_counts[:-1]:
                            offsets.append(offsets[-1] + d)
                        for part, dim_count, off in zip(key_order, dim_counts, offsets):
                            if part in absent:
                                valid[:, off : off + dim_count] = False

                diff = torch.abs(pred - gt)
                n_valid = valid.sum().clamp(min=1)
                cur_l1 = diff.masked_select(valid).sum() / n_valid.float()
            else:
                cur_l1 = F.l1_loss(pred, gt)

            total_l1 += cur_l1.item()
            valid_count += 1

        if valid_count == 0:
            return -1.0
        return total_l1 / valid_count

# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
ARHelper: weight-free autoregressive algorithm helper.

Owns CE loss and the AR decode loop, including block-wise autoregressive decode.
Accesses weights through the model reference (model.vlm.embed/forward/decode).
Does not own any nn.Parameter; precision management is fully handled by Mixture.

Based on GalaxeaJoint.cal_ce_acc + GalaxeaARMixin refactoring.
"""

import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn

from ..model.utils import kv_cache_seq_len
import torch.nn.functional as F

from ..io.input_preprocessor import TOKEN_INDEX
from g05.models.kv_cache import SparseKVCache as _SKV

logger = logging.getLogger(__name__)

# Fallback TOKEN_INDEX when dynamic ranges are not configured.
_FALLBACK_GEN_TOKEN_INDEX = int(TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX)


class ARHelper:
    """Autoregressive algorithm helper.

    Args:
        cfg: model_arch.ar config, containing:
            - ce_weight: CE loss weight
            - vocab_size: VLM vocab size interpolated from the top-level config by Hydra
    """

    def __init__(self, cfg):
        self.ce_weight = cfg.ce_weight
        self.vocab_size = cfg.vocab_size

        # Token range constants (from config, fallback to PaliGemma defaults)
        self.text_token_end = cfg.get("text_token_end", 256000)
        self.loc_token_end = cfg.get("loc_token_end", 257024)

        # Fused Linear CE (liger_kernel): avoids materializing [N_valid, V] logits,
        # reducing peak memory from ~20GB to ~2GB. Log steps need raw logits for
        # ECE/logit_stats, so G05Policy.forward sets _need_full_logits=True before
        # those steps to switch back to the eager path.
        #
        # Must be explicitly declared in config instead of using .get fallback: the
        # feature flag has no default, which avoids silently enabling/disabling it
        # (see feedback_explicit_config). ce_z_loss_scale follows the same rule:
        # 0.0 disables z-loss, Gemma recommends 1e-4, and the current value must be
        # written explicitly in config.
        self.use_fused_ce = cfg.use_fused_ce
        self.ce_z_loss_scale = float(cfg.ce_z_loss_scale)
        # Runtime gate: MetricsCollector.on_forward_start sets this to True before
        # log steps. cal_ce_loss then uses the eager slow path and keeps full logits
        # for ECE/logit_stats in patches.py.
        self._need_full_logits = False

        # BAR (Block-wise Autoregressive): concrete values are synced by G05Policy
        # from action_tokenizer.
        self.block_wise_autoregressive = cfg.block_wise_autoregressive
        self.bos_blk_id = cfg.bos_blk_id
        self.eos_blk_id = cfg.eos_blk_id
        self.block_size = cfg.block_size

        # Cache cal_ce_loss intermediate results for external metric patches to
        # reuse without repeated decode.
        self._last_ce_cache = None

        # AR inference sampling parameters from config, overridable by inference kwargs.
        self.do_sample = cfg.get("do_sample", True)
        self.max_new_tokens = cfg.get("max_new_tokens", 300)
        self.temperature = cfg.get("temperature", 0.7)
        self.top_k = cfg.get("top_k", 128)
        self.top_p = cfg.get("top_p", 0.95)
        self.repetition_penalty = cfg.get("repetition_penalty", 1.2)
        self.no_repeat_ngram_size = cfg.get("no_repeat_ngram_size", 3)

        # Dynamic TOKEN_INDEX assignment for AR inference.
        # Set by G05Policy.set_token_index_ranges() after action_tokenizer is created.
        self._token_index_ranges: Optional[List[dict]] = None

        # EOV token ID — set by G05Policy after tokenizer init.
        # Used by forward_inference to detect EOV for FM trigger.
        self.eov_token_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Token Index assignment (aligned with the legacy assign_position_ids_by_ranges)
    # ------------------------------------------------------------------

    def set_token_index_ranges(
        self,
        action_token_begin_idx: int,
        action_token_end_idx: int,
    ):
        """Configure token ID -> TOKEN_INDEX mapping for AR inference.

        Aligned with legacy GalaxeaJointPolicy.position_ids_config_list:
        - text tokens [0, text_token_end)              -> PRED_TEXT_TOKEN_INDEX = 6
        - loc tokens  [text_token_end, loc_token_end)  -> COT_TOKEN_INDEX = 5
        - action tokens                                -> ACTION_TOKEN_INDEX = 3

        Args:
            action_token_begin_idx: start of action token ID range (inclusive)
            action_token_end_idx: end of action token ID range (exclusive)
        """
        self._token_index_ranges = [
            {
                "lower": 0,
                "upper": self.text_token_end - 1,
                "pos_id": int(TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX),
            },
            {
                "lower": self.text_token_end,
                "upper": self.loc_token_end - 1,
                "pos_id": int(TOKEN_INDEX.COT_TOKEN_INDEX),
            },
            {
                "lower": action_token_begin_idx,
                "upper": action_token_end_idx,
                "pos_id": int(TOKEN_INDEX.ACTION_TOKEN_INDEX),
            },
        ]
        logger.info(
            f"[ARHelper] token_index_ranges configured: "
            f"action=[{action_token_begin_idx}, {action_token_end_idx}]"
        )

    @property
    def action_token_range(self) -> Optional[tuple]:
        """Return the (begin, end) action token ID range, or None if not configured."""
        if self._token_index_ranges is None:
            return None
        for cfg in self._token_index_ranges:
            if cfg["pos_id"] == int(TOKEN_INDEX.ACTION_TOKEN_INDEX):
                return (cfg["lower"], cfg["upper"])
        return None

    def _assign_token_index(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map generated token IDs -> TOKEN_INDEX for attention_mask growth.

        Aligned with legacy GalaxeaARMixin.assign_position_ids_by_ranges.
        Works on any shape tensor (element-wise).

        Invariant: this applies only to the DECODE path.
        For action tokens, this function always returns integer ACTION_TOKEN_INDEX=3.
        It does not generate the 3.0x BAR fractional labels produced by training-side
        build_block_io. This simplification is correct only when action tokens are
        processed exclusively through KV-cache decode (kv_len > 0), because the decode
        branch only checks attention_mask != 0 and ignores exact values. BAR
        intra-block bidirectional attention is implemented implicitly in _infer_single
        by feeding block_size query tokens in one forward pass.

        Do not feed this function's output into the prefill (kv_len=0) branch for
        action tokens. mask_helper's prefill branch relies on 3.0x fractional labels
        to build the block bidirectional mask. Integer 3 from this function would
        degrade BAR into standard causal AR, causing behavior to diverge from training
        without an error. See mask_helper.build_vlm_mask. If action tokens must be
        processed through prefill in the future, use the attention mask returned by
        build_block_io instead.

        Args:
            token_ids: generated token IDs (any shape)
        Returns:
            TOKEN_INDEX values (same shape, long tensor, same device)
        """
        if self._token_index_ranges is None:
            return torch.full_like(token_ids, _FALLBACK_GEN_TOKEN_INDEX)

        result = torch.full_like(token_ids, _FALLBACK_GEN_TOKEN_INDEX)
        for cfg in self._token_index_ranges:
            mask = (token_ids >= cfg["lower"]) & (token_ids <= cfg["upper"])
            result[mask] = cfg["pos_id"]

        # If all tokens fall outside configured ranges (e.g. token id > 261276), keep
        # using the fallback. This should not happen.
        return result

    # ------------------------------------------------------------------
    # CE Loss
    # ------------------------------------------------------------------

    @torch.autocast("cuda", enabled=False)
    def cal_ce_loss(
        self,
        vlm_hidden: torch.Tensor,
        labels: torch.LongTensor,
        model,
    ):
        """Shift + mask + vlm.decode -> CrossEntropy.

        Args:
            vlm_hidden: [B, S, d_vlm]
            labels: [B, S] (-100 masked)
            model: G05Model (for model.vlm.decode)
        Returns:
            (ce_loss, accuracy)
        """
        shift_hidden = vlm_hidden[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        mask = shift_labels != -100
        num_valid = int(mask.sum().item())
        shift_labels = shift_labels.to(shift_hidden.device)

        if num_valid == 0:
            logger.warning("[CE] all labels are -100, returning zero loss")
            self._last_ce_cache = None
            zero = torch.tensor(
                0.0,
                device=vlm_hidden.device,
                dtype=vlm_hidden.dtype,
                requires_grad=True,
            )
            return zero, 0.0

        shift_labels_masked = shift_labels[mask]
        shift_hidden_masked = shift_hidden.flatten(0, 1)[mask]

        # Dual path: FLCE saves memory; eager keeps logits for log-step metrics.
        # MetricsCollector.on_forward_start sets _need_full_logits=True before log
        # steps. Non-log steps (~99%) use FLCE; log steps (~1%) use eager. The
        # worst-case peak matches the previous path, while ~99% of steps drop to ~2GB.
        use_flce = self.use_fused_ce and not self._need_full_logits

        if use_flce:
            # FAST PATH: Liger Fused Linear CE, without materializing [N_valid, V]
            # logits. One kernel returns both per-token loss and per-token correctness
            # (return_token_accuracy), avoiding an extra argmax matmul.
            # Dtype note: FLCE requires input and weight to have the same dtype. Both
            # are bf16 under autocast; FLCE internally accumulates chunks in fp32.
            # Do not manually call .float(), or input/weight dtype mismatch will raise.
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            weight = model.vlm.output_proj.weight  # tied with input_proj.weight
            flce = LigerFusedLinearCrossEntropyLoss(
                reduction="none",
                lse_square_scale=self.ce_z_loss_scale,
                ignore_index=-100,
                return_token_accuracy=True,
            )
            out = flce(weight, shift_hidden_masked, shift_labels_masked)
            token_loss = out.loss  # [N_valid] fp32
            per_token_correct = out.token_accuracy  # [N_valid] 0/1 fp32
            pred = None  # fast path does not materialize logits, so no pred
            shift_logits_for_cache = None
        else:
            # SLOW PATH: original eager path, used on log steps or when use_fused_ce=False.
            shift_logits = model.vlm.decode(shift_hidden_masked)
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            token_loss = F.cross_entropy(shift_logits, shift_labels_masked, reduction="none")
            pred = shift_logits.detach().argmax(dim=-1)
            per_token_correct = (pred == shift_labels_masked).float()
            shift_logits_for_cache = shift_logits.detach()

        ce_loss = token_loss.mean() * self.ce_weight
        accuracy = per_token_correct.sum() / num_valid

        # Split accuracy into action vs cot (text) using token ID ranges
        # Both paths share per_token_correct; pred is not needed for this metric.
        action_accuracy_val = 0.0
        cot_accuracy_val = 0.0
        arange = self.action_token_range
        if arange is not None:
            action_mask = (shift_labels_masked >= arange[0]) & (shift_labels_masked <= arange[1])
            cot_mask = ~action_mask
            n_action = int(action_mask.sum().item())
            n_cot = int(cot_mask.sum().item())
            if n_action > 0:
                action_accuracy_val = float(per_token_correct[action_mask].sum()) / n_action
            if n_cot > 0:
                cot_accuracy_val = float(per_token_correct[cot_mask].sum()) / n_cot

        # Cache per-token intermediates for external metric patches to reuse without
        # repeated decode. In the fast path shift_logits=None; patches._consume_ce_cache_expensive
        # has a fallback early return when cache["shift_logits"] is None. We also cache
        # _per_token_correct directly so _compute_per_sample_ce_acc can skip recomputing
        # pred == label.
        self._last_ce_cache = {
            "token_loss": token_loss.detach(),
            "pred": pred,  # fast path=None, slow path=[N_valid]
            "_per_token_correct": per_token_correct.detach(),  # available in both paths
            "shift_logits": shift_logits_for_cache,  # fast path=None
            "shift_labels_masked": shift_labels_masked,
            "mask": mask,  # 1D bool, used to recover sample ownership
            "batch_size": vlm_hidden.shape[0],
            "seq_len": vlm_hidden.shape[1] - 1,  # seq_len after shifting
            "ce_weight": self.ce_weight,
            "action_accuracy": action_accuracy_val,
            "cot_accuracy": cot_accuracy_val,
        }

        return ce_loss, accuracy.detach()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        model,
        vlm_hidden: torch.Tensor,
        labels: torch.LongTensor,
    ):
        """Single AR training step.

        Args:
            model: G05Model
            vlm_hidden: [B, S, d_vlm]
            labels: [B, S]
        Returns:
            (ce_loss, action_accuracy)
        """
        return self.cal_ce_loss(vlm_hidden, labels, model)

    # ------------------------------------------------------------------
    # Inference — entry point
    # ------------------------------------------------------------------

    def infer(
        self,
        model,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_window: Optional[int] = None,
        ignore_token_ids: Optional[List[int]] = None,
        return_kv_cache: bool = False,
        past_key_values: Optional[List] = None,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """AR inference: last_hidden -> sample -> decode loop.

        The caller must run vlm_prefill() first to provide past_key_values and
        last_hidden. last_hidden is the hidden state that can decode logits for the
        next token: either the last prefill hidden state or the last hidden returned
        by the previous inference_ar call.

        Difference from legacy seed_ids semantics:
          - Old: input_ids was the prefix last token already in KV, then forwarded
                again, duplicating the KV slot and doubling attention weight.
          - New: last_hidden is the hidden state at the end of context; the first
                generated token is sampled directly from it. The prefix last token
                is not forwarded again, so there is no KV duplication or train/test
                mismatch.

        Sampling parameters prefer call-time values, then config defaults.
        BAR mode loops per sample, matching the legacy per-sample decode.

        Args:
            model: G05Model
            last_hidden: [B, d_vlm] context-ending hidden used to sample the first token
            attention_mask: [B, S_prefix] full attention mask matching past_key_values length
            pixel_values: Dict[str, Tensor[B, n_k, C, H_k, W_k]], used only for dtype/device
            past_key_values: VLM KV cache produced by vlm_prefill; required
            max_new_tokens: maximum generated token count (default from config)
            do_sample: whether to sample; False uses argmax greedy decoding
            return_kv_cache: whether to return KV cache / attention_mask / last_hidden
            stop_token_ids: stop generation when any of these token IDs is generated
        Returns:
            dict with:
              "generated_ids": [B, gen_len]
              "last_hidden":   [B, d_vlm]  (when return_kv_cache=True)
              "past_key_values": list      (when return_kv_cache=True)
              "attention_mask": [B, S_new] (when return_kv_cache=True)
        """
        # Prefer call-time values, then config defaults.
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        do_sample = do_sample if do_sample is not None else self.do_sample
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.no_repeat_ngram_size
        )

        # Force argmax when do_sample=False.
        if not do_sample:
            temperature = 0.0

        sampling_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_window=repetition_window,
            ignore_token_ids=ignore_token_ids,
        )

        # Caller must provide past_key_values; prefill is completed by the caller.
        assert past_key_values is not None, (
            "past_key_values is required — caller must do vlm_prefill() first"
        )

        if self.block_wise_autoregressive:
            return self._infer_with_bar(
                model,
                last_hidden,
                attention_mask,
                pixel_values,
                max_new_tokens,
                past_key_values=past_key_values,
                stop_token_ids=stop_token_ids,
                verbose=verbose,
                **sampling_kwargs,
            )
        return self._infer_batched(
            model,
            last_hidden,
            attention_mask,
            pixel_values,
            max_new_tokens,
            return_kv_cache=return_kv_cache,
            past_key_values=past_key_values,
            stop_token_ids=stop_token_ids,
            verbose=verbose,
            **sampling_kwargs,
        )

    # ------------------------------------------------------------------
    # Non-BAR batched inference
    # ------------------------------------------------------------------

    def _infer_batched(
        self,
        model,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        max_new_tokens: int,
        return_kv_cache: bool = False,
        past_key_values: List = None,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        **sampling_kwargs,
    ) -> dict:
        """Batched parallel AR decode. The caller owns prefill.

        Invariants:
          - attention_mask.size(1) always equals past_key_values length and grows with it
          - cur_hidden is always the hidden state that can decode next-token logits:
              * initial = last prefill hidden passed by caller as last_hidden
              * after each forward = last hidden of the newly forwarded token

        Single-step flow, completed atomically inside each loop iteration:
            sample(cur_hidden) → next_token
            → grow attention_mask for the new token
            → forward(next_token, kv_len, grown mask)  →  new hidden + new KV slot
            → cur_hidden = new hidden

        Args:
            last_hidden: [B, d_vlm] initial context-ending hidden
            attention_mask: [B, S] full attention mask matching past_key_values length
            past_key_values: VLM KV cache, provided by caller prefill
            input_ids: [B, 1] seed token
            attention_mask: [B, S] full attention mask
        """
        import time

        first_image = next(iter(pixel_values.values()))
        dtype, device = first_image.dtype, first_image.device
        bsz = last_hidden.size(0)
        eos_id = model.cfg.eos_token_id
        stop_ids = set(stop_token_ids) if stop_token_ids else set()
        if eos_id is not None:
            stop_ids.add(eos_id)

        vlm_kv = past_key_values
        # kv_len is the actual past KV cache length, i.e. token count from prefill.
        kv_len = kv_cache_seq_len(past_key_values)
        cur_hidden = last_hidden  # [B, d_vlm], hidden that predicts the next token
        token_history = None
        generated_ids = []
        pbar = None
        if verbose:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=max_new_tokens,
                    desc="AR decode",
                    unit="tok",
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                logger.warning("tqdm not installed, verbose=True has no effect")
        t_start = time.perf_counter()

        for _ in range(max_new_tokens):
            # 1. Sample next token from current hidden (always [B, 1, d_vlm] into decode)
            cur_hidden_3d = (
                cur_hidden.unsqueeze(1) if cur_hidden.dim() == 2 else cur_hidden[:, -1:, :]
            )
            logits = model.vlm.decode(cur_hidden_3d)  # [B, 1, V]
            next_token = self._sample(
                logits[:, -1, :], prev_tokens=token_history, **sampling_kwargs
            )
            generated_ids.append(next_token)

            # 2. Stop check: the stop token counts as generated but is not committed to KV.
            if all(int(next_token[b]) in stop_ids for b in range(bsz)):
                if pbar is not None:
                    pbar.update(1)
                break

            # 3. Grow attention_mask + token_history for this new token before forward.
            if token_history is None:
                token_history = next_token.unsqueeze(1)
            else:
                token_history = torch.cat(
                    [token_history, next_token.unsqueeze(1)],
                    dim=1,
                )
            _token_idx = self._assign_token_index(next_token).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, _token_idx], dim=1)

            # 4. Forward next_token -> new hidden + append to KV cache.
            cur_token = next_token.unsqueeze(1)  # [B, 1]
            token_embeds = model.vlm.embed(cur_token)
            step_mask, step_pos = model.build_causal_mask_and_position_ids(
                cur_token,
                attention_mask,
                kv_len=kv_len,
                dtype=dtype,
            )
            _cache_kwargs = (
                {"kv_cache": vlm_kv} if isinstance(vlm_kv, _SKV) else {"past_key_values": vlm_kv}
            )
            new_hidden, vlm_kv = model.vlm(
                inputs_embeds=token_embeds,
                attention_mask=step_mask,
                position_ids=step_pos,
                return_kv_cache=True,
                attn_implementation=model.attn_implementation,
                mixture_name="vlm",
                **_cache_kwargs,
            )
            kv_len += 1
            cur_hidden = new_hidden[:, -1, :]  # [B, d_vlm]

            if pbar is not None:
                elapsed = time.perf_counter() - t_start
                n_gen = len(generated_ids)
                tps = n_gen / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix({"tok/s": f"{tps:.1f}"}, refresh=False)
                pbar.update(1)

        if pbar is not None:
            elapsed = time.perf_counter() - t_start
            n_gen = len(generated_ids)
            tps = n_gen / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix({"tok/s": f"{tps:.1f}"})
            pbar.close()

        if not generated_ids:
            result = {"generated_ids": torch.zeros(bsz, 0, dtype=torch.long, device=device)}
        else:
            result = {"generated_ids": torch.stack(generated_ids, dim=1)}
        if return_kv_cache:
            result["past_key_values"] = vlm_kv
            result["attention_mask"] = attention_mask
            # Extract a single hidden: keep [B, d] if not forwarded; otherwise take the last position.
            result["last_hidden"] = cur_hidden if cur_hidden.dim() == 2 else cur_hidden[:, -1, :]
        return result

    # ------------------------------------------------------------------
    # BAR inference: per-sample loop aligned with legacy _forward_infer_discrete
    # ------------------------------------------------------------------

    def _infer_with_bar(
        self,
        model,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        max_new_tokens: int,
        past_key_values: Optional[List] = None,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        **sampling_kwargs,
    ) -> dict:
        """BAR inference: call _infer_single per sample, then pad and stack results."""
        bsz = last_hidden.size(0)
        device = last_hidden.device
        all_tokens: List[List[torch.Tensor]] = []
        max_gen_len = 0

        def _split_pixel_values(pixel_values, idx):
            return {k: v[idx : idx + 1] for k, v in pixel_values.items()}

        for i in range(bsz):
            # Slice precomputed KV per sample.
            if past_key_values is None:
                kv_i = None
            elif isinstance(past_key_values, _SKV):
                kv_i = past_key_values.batch_slice(i)
            else:
                kv_i = [(k[i : i + 1], v[i : i + 1]) for k, v in past_key_values]
            tokens = self._infer_single(
                model,
                last_hidden[i : i + 1],
                attention_mask[i : i + 1],
                _split_pixel_values(pixel_values, i),
                max_new_tokens,
                past_key_values=kv_i,
                stop_token_ids=stop_token_ids,
                verbose=verbose,
                **sampling_kwargs,
            )
            all_tokens.append(tokens)
            max_gen_len = max(max_gen_len, len(tokens))

        if max_gen_len == 0:
            return {
                "generated_ids": torch.zeros(
                    bsz,
                    0,
                    dtype=torch.long,
                    device=device,
                )
            }

        # Pad to same length and stack -> [B, max_gen_len]
        # Use eos_token_id for padding instead of tokenizer pad_token_id, usually 0.
        # Downstream decode_ar calls tokenizer.decode(full_ids) and splits strings. If
        # pad id 0 is used, literal <pad> can appear inside the action string; encoding
        # it back becomes [0, 0, ...] and pollutes _aggregate_actions regex parsing
        # (see processor._clean_special_tokens). eos padding matches the semantics
        # because the sample has ended, and _clean_special_tokens strips literal <eos>.
        eos_id = getattr(model.cfg, "eos_token_id", 0)
        padded = []
        for tokens in all_tokens:
            if tokens:
                cat = torch.stack(tokens)  # [gen_len]
            else:
                cat = torch.zeros(0, dtype=torch.long, device=device)
            if cat.size(0) < max_gen_len:
                cat = F.pad(cat, (0, max_gen_len - cat.size(0)), value=eos_id)
            padded.append(cat)

        return {"generated_ids": torch.stack(padded)}

    # ------------------------------------------------------------------
    # Single-sample decode with BAR state machine
    # (aligned with legacy GalaxeaARMixin._inference_ar_single)
    # ------------------------------------------------------------------

    def _infer_single(
        self,
        model,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        max_new_tokens: int,
        past_key_values: List = None,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        **sampling_kwargs,
    ) -> List[torch.Tensor]:
        """Single-sample AR inference with BAR state machine. Caller owns prefill.

        Args:
            past_key_values: VLM KV cache, provided by caller prefill
            last_hidden: [1, d_vlm] last prefill hidden; if 3D [1,1,d]/[1,k,d], the
                last position is used
            attention_mask: [1, S] full attention mask

        Structure:
          - Sample the first token directly from last_hidden without forwarding the
            prefix last token.
          - If the first token is bos_blk, enter BAR immediately; otherwise continue
            with normal AR.
          - Each later step: forward cur_input -> new hidden -> sample out_len new
            tokens -> update BAR state + cur_input.

        BAR intra-block bidirectional attention implementation, compared with training:
            During training, build_block_io labels each BAR block with 3.01/3.02/...
            fractional tags. mask_helper's prefill branch uses these tags to build
            bidirectional masks inside each block.

            During inference there are no fractional labels: _assign_token_index only
            returns integer 3. BAR bidirectionality is implemented implicitly by
            feeding block_size query tokens in one forward pass:
              - on entering BAR mode, cur_input is filled with [1, block_size] bos_blk
              - later each step uses cur_input = next_tokens.unsqueeze(0), with
                block_size tokens
              - mask_helper's decode branch does not mask among q tokens when
                q_len=block_size, so queries in the same forward pass can see each other
            This is equivalent to the bidirectional mask produced by training prefill
            3.0x fractional labels: different code paths, same attention pattern.

            This equivalence relies on the invariant that action tokens are processed
            only in the decode path (kv_len > 0). Do not put action tokens into
            prefill: _assign_token_index does not emit 3.0x labels, so BAR blocks
            would degrade into standard causal attention and silently diverge. See
            _assign_token_index and mask_helper.build_vlm_mask.
        """
        import time

        first_image = next(iter(pixel_values.values()))
        dtype, device = first_image.dtype, first_image.device
        eos_id = model.cfg.eos_token_id
        stop_ids = set(stop_token_ids) if stop_token_ids else set()
        if eos_id is not None:
            stop_ids.add(eos_id)
        stop_ids = stop_ids or None
        bos_blk = self.bos_blk_id
        eos_blk = self.eos_blk_id
        block_size = self.block_size
        id_dtype = torch.long

        vlm_kv = past_key_values
        # kv_len is the actual past KV cache length, i.e. token count from prefill.
        kv_len = kv_cache_seq_len(past_key_values)
        token_history = None
        generated_tokens: List[torch.Tensor] = []
        in_bar = False
        pbar = None
        if verbose:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=max_new_tokens,
                    desc="AR decode",
                    unit="tok",
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                logger.warning("tqdm not installed, verbose=True has no effect")
        t_start = time.perf_counter()

        # -------- Pre-loop: sample the first token directly from last_hidden --------
        # last_hidden may be [1, d] or [1, k, d] (last hidden from the end of Stage 2).
        if last_hidden.dim() == 3:
            seed_hidden_3d = last_hidden[:, -1:, :]
        else:
            seed_hidden_3d = last_hidden.unsqueeze(1)  # [1, 1, d]
        first_logits = model.vlm.decode(seed_hidden_3d)  # [1, 1, V]
        first_token = self._sample(
            first_logits[:, 0, :], prev_tokens=None, **sampling_kwargs
        )  # [1]
        first_token_scalar = first_token[0]  # 0-D scalar, consistent with loop append format
        generated_tokens.append(first_token_scalar)

        # If the first token is a stop token, return directly; caller state is unchanged.
        if stop_ids and int(first_token_scalar.item()) in stop_ids:
            return generated_tokens

        # Choose cur_input. The first token may be bos_blk, entering BAR and expanding
        # to [1, block_size].
        if self.block_wise_autoregressive and first_token_scalar.item() == bos_blk:
            in_bar = True
            cur_input = torch.full(
                (1, block_size),
                bos_blk,
                device=device,
                dtype=id_dtype,
            )
        else:
            cur_input = first_token.unsqueeze(0)  # [1] → [1, 1]
        token_history = cur_input.clone()
        _idx = self._assign_token_index(cur_input.squeeze(0)).unsqueeze(0)
        attention_mask = torch.cat([attention_mask, _idx], dim=1)

        # -------- Main loop: forward cur_input -> sample out_len -> update state --------
        remaining = max_new_tokens - 1  # first token was generated in the pre-loop
        for _ in range(remaining):
            # Forward
            q_len = cur_input.size(1)
            token_embeds = model.vlm.embed(cur_input)
            step_mask, step_pos = model.build_causal_mask_and_position_ids(
                cur_input,
                attention_mask,
                kv_len=kv_len,
                dtype=dtype,
                is_action_block=in_bar,
            )

            _cache_kwargs = (
                {"kv_cache": vlm_kv} if isinstance(vlm_kv, _SKV) else {"past_key_values": vlm_kv}
            )
            hidden, vlm_kv = model.vlm(
                inputs_embeds=token_embeds,
                attention_mask=step_mask,
                position_ids=step_pos,
                return_kv_cache=True,
                attn_implementation=model.attn_implementation,
                mixture_name="vlm",
                **_cache_kwargs,
            )
            kv_len += q_len

            # Sample
            out_len = q_len if in_bar else 1
            cur_logits = model.vlm.decode(hidden[:, -out_len:, :])
            sampled_list = []
            for p in range(out_len):
                t = self._sample(
                    cur_logits[:, p, :],
                    prev_tokens=token_history,
                    **sampling_kwargs,
                )
                sampled_list.append(t.squeeze(0))
            next_tokens = torch.stack(sampled_list)

            # Stop check (outside BAR)
            if not in_bar and stop_ids and int(next_tokens[0].item()) in stop_ids:
                generated_tokens.append(next_tokens[0])
                break

            # --- BAR state machine ---
            if in_bar:
                if (next_tokens == eos_blk).any():
                    in_bar = False
                    cur_input = torch.tensor([[eos_blk]], device=device, dtype=id_dtype)
                    generated_tokens.append(torch.tensor(eos_blk, device=device, dtype=id_dtype))
                else:
                    for i in range(next_tokens.size(0)):
                        generated_tokens.append(next_tokens[i])
                    cur_input = next_tokens.unsqueeze(0)
            else:
                generated_tokens.append(next_tokens[0])
                if self.block_wise_autoregressive and next_tokens[0].item() == bos_blk:
                    in_bar = True
                    cur_input = torch.full(
                        (1, block_size),
                        bos_blk,
                        device=device,
                        dtype=id_dtype,
                    )
                else:
                    cur_input = next_tokens[0:1].unsqueeze(0)

            # Grow attention_mask + token_history
            _idx = self._assign_token_index(cur_input.squeeze(0)).unsqueeze(0)
            attention_mask = torch.cat([attention_mask, _idx], dim=1)
            flat = cur_input.squeeze(0)
            token_history = torch.cat([token_history, flat.unsqueeze(0)], dim=1)

            if pbar is not None:
                elapsed = time.perf_counter() - t_start
                n_gen = len(generated_tokens)
                tps = n_gen / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix({"tok/s": f"{tps:.1f}"}, refresh=False)
                pbar.update(1)

        if pbar is not None:
            elapsed = time.perf_counter() - t_start
            n_gen = len(generated_tokens)
            tps = n_gen / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix({"tok/s": f"{tps:.1f}"})
            pbar.close()

        return generated_tokens

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        prev_tokens: Optional[torch.Tensor] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_window: Optional[int] = None,
        ignore_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Sample with top-k/top-p + repetition penalty, matching ar_sampling.py args.

        Args:
            logits: [B, V]
            prev_tokens: [B, seq_so_far] or None
        Returns:
            next_token: [B]
        """
        if temperature is not None and temperature <= 0:
            return logits.argmax(dim=-1)

        from .ar_sampling import top_k_top_p_filtering

        sampled = top_k_top_p_filtering(
            logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature if temperature is not None else 1.0,
            prev_tokens=prev_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_window=repetition_window,
            ignore_token_ids=ignore_token_ids,
        )
        return sampled.squeeze(-1)  # [B]

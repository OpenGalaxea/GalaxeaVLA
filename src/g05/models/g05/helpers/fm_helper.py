# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
FMHelper: weight-free Flow Matching algorithm helper.

Owns time sampling, psi_t interpolation, velocity loss, and Euler integration.
Accesses weights through the model reference (model.action_expert.embed/forward/decode).
Does not own any nn.Parameter; precision management is fully handled by Mixture.

Based on GalaxeaJoint + GalaxeaJointPolicy refactoring.
"""

import logging

import torch
import torch.distributions as dist
from typing import Dict
from g05.models.kv_cache import SparseKVCache

logger = logging.getLogger(__name__)


class FMHelper:
    """Flow Matching algorithm helper.

    Args:
        cfg: model_arch.fm config, containing:
            - time_convention: "pi_convention" | "galaxea_convention"
            - flow_sampling: "beta" | "uniform"
            - flow_sig_min: noise floor
            - num_inference_steps: Euler integration step count
            - fm_weight: FM loss weight
            - joint_training: whether gradients flow through VLM KV
            - padding_action_weight: loss weight for padding positions
            - zero_pad_action_target: mode A/B switch
            - num_flow_samples: FM sample multiplier. Default 1; when >1, sample N
              sets of (t, noise) and merge them into an [N*B] batch to reduce
              gradient variance.
            - final_action_clip_value: output clip value
            - horizon_steps: action chunk length H, interpolated from the top-level config by Hydra
            - action_dim: action dimension D, interpolated from the top-level config by Hydra
    """

    def __init__(self, cfg):
        self.time_convention = cfg.time_convention
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.fm_weight = cfg.fm_weight
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.flow_sig_min
        self.padding_action_weight = cfg.padding_action_weight
        self.zero_pad_action_target = cfg.zero_pad_action_target
        self.joint_training = cfg.joint_training
        self.num_flow_samples = getattr(cfg, "num_flow_samples", 1)
        self.action_causal = getattr(cfg, "action_causal", False)
        if self.num_flow_samples > 1:
            logger.info(
                f"[FMHelper] Multi-sample flow matching enabled: "
                f"num_flow_samples={self.num_flow_samples} (AE batch = {self.num_flow_samples}×B)"
            )

        # Correlated noise (per-embodiment selective)
        self.use_correlated_noise = getattr(cfg, "use_correlated_noise", False)
        self.correlation_beta = getattr(cfg, "correlation_beta", 0.5)
        self.correlated_noise_embodiments = list(
            getattr(cfg, "correlated_noise_embodiments", []) or []
        )
        self._action_correlation_cholesky = None
        cholesky_path = getattr(cfg, "action_correlation_cholesky_path", None)
        if self.use_correlated_noise:
            assert self.correlated_noise_embodiments, (
                "use_correlated_noise=True requires non-empty correlated_noise_embodiments; "
                "otherwise all embodiments would be silently corrupted by correlated noise."
            )
            assert cholesky_path, (
                "use_correlated_noise=True requires action_correlation_cholesky_path to be set."
            )
            self._load_cholesky(cholesky_path)

        # Time sampling
        self.flow_sampling = cfg.flow_sampling
        self.flow_t_max = 1 - self.flow_sig_min
        if self.flow_sampling == "beta":
            self.flow_beta_dist = dist.Beta(1.5, 1.0)

    # ------------------------------------------------------------------
    # Time sampling
    # ------------------------------------------------------------------

    def sample_time(self, bsz: int, device: torch.device = None) -> torch.Tensor:
        """Sample FM training time.

        Args:
            bsz: batch size
            device: target device
        Returns:
            t: [bsz] float tensor, range depends on time_convention
        """
        if self.flow_sampling == "uniform":
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        else:  # beta
            z = self.flow_beta_dist.sample((bsz,))
            if self.time_convention == "pi_convention":
                t = 1 - self.flow_t_max * (1 - z)
            else:  # galaxea_convention
                t = self.flow_t_max * (1 - z)

        if device is not None:
            t = t.to(device)
        return t

    # ------------------------------------------------------------------
    # Correlated noise
    # ------------------------------------------------------------------

    def _load_cholesky(self, path: str):
        """Load Cholesky factor and apply beta shrinkage regularization."""
        import numpy as np

        L = torch.from_numpy(np.load(path)).float()
        expected_dim = self.horizon_steps * self.action_dim
        if L.shape[0] != expected_dim:
            logger.warning(
                f"Cholesky shape {L.shape} != expected ({expected_dim},{expected_dim}). "
                f"Expected H={self.horizon_steps} * D={self.action_dim}. Disabling correlated noise."
            )
            self.use_correlated_noise = False
            return
        sigma = L @ L.T
        beta = self.correlation_beta
        sigma_reg = beta * sigma + (1 - beta) * torch.eye(sigma.shape[0])
        self._action_correlation_cholesky = torch.linalg.cholesky(sigma_reg)
        logger.info(
            f"[FMHelper] Loaded correlated noise Cholesky {L.shape}, beta={beta}, "
            f"embodiments={self.correlated_noise_embodiments or 'ALL'}"
        )

    def _sample_noise(
        self,
        actions: torch.Tensor,
        dtype: torch.dtype,
        embodiment_types: list = None,
    ) -> torch.Tensor:
        """Sample noise: correlated for specified embodiments, standard Gaussian for others."""
        B, H, D = actions.shape
        if not self.use_correlated_noise or self._action_correlation_cholesky is None:
            return torch.randn(B, H, D, device=actions.device, dtype=dtype)

        if embodiment_types is not None and self.correlated_noise_embodiments:
            corr_mask = torch.tensor(
                [emb in self.correlated_noise_embodiments for emb in embodiment_types],
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            corr_mask = torch.zeros(B, dtype=torch.bool, device=actions.device)

        if not corr_mask.any():
            return torch.randn(B, H, D, device=actions.device, dtype=dtype)

        noise = torch.randn(B, H, D, device=actions.device, dtype=dtype)
        chol = self._action_correlation_cholesky.to(device=actions.device, dtype=torch.float32)

        if corr_mask.all():
            z = noise.to(torch.float32).reshape(B, H * D)
            return (z @ chol.T).reshape(B, H, D).to(dtype)
        else:
            n_corr = corr_mask.sum().item()
            z_corr = noise[corr_mask].to(torch.float32).reshape(n_corr, H * D)
            noise[corr_mask] = (z_corr @ chol.T).reshape(n_corr, H, D).to(dtype)
            return noise

    # ------------------------------------------------------------------
    # Conditional flow
    # ------------------------------------------------------------------

    def psi_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional flow interpolation.

        x0: noise, x1: action (data), t: [B]
        pi_convention:      psi_t = (1-t)*x1 + t*x0   (t=0 → data, t=1 → noise)
        galaxea_convention: psi_t = t*x1 + (1-t)*x0   (t=0 → noise, t=1 → data)
        """
        t = t[:, None, None]  # (B, 1, 1)
        if self.time_convention == "pi_convention":
            return (1 - t) * x1 + t * x0
        else:
            return t * x1 + (1 - t) * x0

    # ------------------------------------------------------------------
    # FM Loss
    # ------------------------------------------------------------------

    def cal_fm_loss(
        self,
        v_psi: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        action_pad_masks: torch.BoolTensor,
        action_dim_is_pad: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """Velocity MSE loss.

        Args:
            v_psi: predicted velocity [B, H, D]
            x0: noise [B, H, D]
            x1: action (data) [B, H, D]
            action_pad_masks: [B, H] bool
            action_dim_is_pad: [B, D] bool or None
        Returns:
            scalar loss
        """
        if self.time_convention == "pi_convention":
            d_psi = x0 - x1
        else:
            d_psi = x1 - x0

        l2 = (v_psi - d_psi) ** 2

        # Loss weighting
        action_weights = torch.ones_like(l2)
        action_weights[action_pad_masks] = self.padding_action_weight
        if action_dim_is_pad is not None and not self.zero_pad_action_target:
            action_weights.masked_fill_(
                action_dim_is_pad.unsqueeze(1),
                self.padding_action_weight,
            )

        weighted_l2 = action_weights * l2
        weight_sum = torch.clamp(action_weights.sum(), min=1.0)
        return weighted_l2.sum() / weight_sum

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        model,
        vlm_kv_prefix,  # SparseKVCache (Qwen3.5) or List[Tuple[Tensor, Tensor]] (PaliGemma)
        attention_mask_prefix: torch.Tensor,
        position_ids_prefix: torch.Tensor,
        actions: torch.Tensor,
        action_pad_masks: torch.BoolTensor,
        action_dim_is_pad,
        dtype: torch.dtype,
        embodiment_types: list = None,
    ) -> torch.Tensor:
        """Single FM training step.

        Accesses action_expert embed/encode_time/forward/decode through the model
        reference. Time sampling is done internally; callers do not need to handle it.

        Args:
            model: G05Model instance
            vlm_kv_prefix: List[(K, V)] VLM KV cache already sliced to prefix
            attention_mask_prefix: [B, S_prefix] prefix attention mask
            position_ids_prefix: [B, S_prefix] prefix position_ids (0-indexed)
            actions: [B, H, D] ground truth actions
            action_pad_masks: [B, H]
            action_dim_is_pad: [B, D] or None
            dtype: pixel_values dtype
        Returns:
            fm_loss: scalar
        """
        bsz = actions.size(0)
        x1 = actions

        if self.zero_pad_action_target:
            x1 = x1.clone()
            x1[action_pad_masks] = 0.0
            if action_dim_is_pad is not None:
                dim_mask = action_dim_is_pad.unsqueeze(1).expand(-1, x1.size(1), -1)
                x1[dim_mask] = 0.0

        # Build action mask + position_ids (shared across all flow samples)
        prefix_len = attention_mask_prefix.size(1)
        action_mask, action_pos = model.build_action_mask_and_position_ids(
            attention_mask_prefix,
            action_len=actions.shape[1],
            position_ids_prefix=position_ids_prefix,
            split_index=prefix_len,
            dtype=dtype,
            action_causal=self.action_causal,
        )

        # Detach vlm_kv_prefix if not joint training (shared across all flow samples)
        if isinstance(vlm_kv_prefix, SparseKVCache):
            if not self.joint_training:
                vlm_kv_prefix = vlm_kv_prefix.detach()
        else:
            if not self.joint_training:
                vlm_kv_prefix = [(k.detach(), v.detach()) for k, v in vlm_kv_prefix]

        ae_dtype = model.action_expert.layers[0].mlp.gate_proj.weight.dtype
        N = self.num_flow_samples

        # Sample N sets of (t, noise) and stack into [N*B, ...]
        t_list = [self.sample_time(bsz, device=actions.device).to(dtype) for _ in range(N)]
        noise_list = [self._sample_noise(actions, dtype, embodiment_types) for _ in range(N)]

        flat_t = torch.cat(t_list, dim=0)  # [N*B]
        flat_x0 = torch.cat(noise_list, dim=0)  # [N*B, H, D]
        flat_x1 = x1.repeat(N, 1, 1)  # [N*B, H, D]

        flat_psi_t = self.psi_t(flat_x0, flat_x1, flat_t)

        # Mode B: zero padded dims
        if not self.zero_pad_action_target and action_dim_is_pad is not None:
            flat_dim_mask = (
                action_dim_is_pad.repeat(N, 1).unsqueeze(1).expand(-1, flat_psi_t.size(1), -1)
            )
            flat_psi_t = flat_psi_t.clone()
            flat_psi_t[flat_dim_mask] = 0.0

        # AE embed + encode_time
        with torch.autocast(actions.device.type, enabled=False):
            action_embeds = model.action_expert.embed(flat_psi_t)
            time_cond = model.action_expert.encode_time(flat_t)

        if ae_dtype == torch.bfloat16:
            action_embeds = action_embeds.to(torch.bfloat16)

        # Repeat action_mask, action_pos for N*B batch
        flat_action_mask = action_mask.repeat(N, *([1] * (action_mask.ndim - 1)))
        if action_pos.ndim == 3 and action_pos.shape[0] == 3:
            flat_action_pos = action_pos.repeat(1, N, 1)
        else:
            flat_action_pos = action_pos.repeat(N, *([1] * (action_pos.ndim - 1)))

        # Repeat VLM KV cache N times along batch dim
        if isinstance(vlm_kv_prefix, SparseKVCache):
            ae_kv_kwargs_flat = {"kv_cache": vlm_kv_prefix.repeat_batch(N)}
        else:
            if N > 1:

                def _repeat_kv(t, n):
                    return t.repeat(n, *([1] * (t.ndim - 1)))

                flat_kv = [(_repeat_kv(k, N), _repeat_kv(v, N)) for k, v in vlm_kv_prefix]
            else:
                flat_kv = vlm_kv_prefix
            ae_kv_kwargs_flat = {"past_key_values": flat_kv}

        # Single AE forward on N*B batch
        action_hidden = model.action_expert(
            inputs_embeds=action_embeds,
            attention_mask=flat_action_mask,
            position_ids=flat_action_pos,
            time_cond=time_cond,
            attn_implementation=model.attn_implementation,
            mixture_name="action",
            **ae_kv_kwargs_flat,
        )

        with torch.autocast(actions.device.type, enabled=False):
            v_psi = model.action_expert.decode(action_hidden)

        flat_pad_masks = action_pad_masks.repeat(N, 1) if action_pad_masks is not None else None
        flat_dim_is_pad = action_dim_is_pad.repeat(N, 1) if action_dim_is_pad is not None else None

        fm_loss = self.cal_fm_loss(v_psi, flat_x0, flat_x1, flat_pad_masks, flat_dim_is_pad)
        return fm_loss * self.fm_weight

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        model,
        attention_mask: torch.Tensor,
        pixel_values: Dict[str, torch.Tensor],
        past_key_values: list,
        action_dim_is_pad=None,
        position_ids_override=None,
        embodiment_types: list = None,
        **kwargs,
    ) -> torch.Tensor:
        """FM inference: Euler integration. Caller owns prefill.

        Args:
            model: G05Model instance
            attention_mask: [B, S_prefix]
            pixel_values: Dict[str, Tensor[B, n_k, C, H_k, W_k]], used only for dtype/device
            past_key_values: VLM KV cache, provided by caller prefill
            action_dim_is_pad: [B, D] or None
            position_ids_override: position_ids used for action position computation
        Returns:
            action: [B, H, D]
        """
        first_image = next(iter(pixel_values.values()))
        bsz, device, dtype = first_image.size(0), first_image.device, first_image.dtype

        vlm_kv = past_key_values
        position_ids = position_ids_override
        past_key_values = vlm_kv

        # Padded-dim mask
        if not self.zero_pad_action_target and action_dim_is_pad is not None:
            _pad_mask = action_dim_is_pad.bool().unsqueeze(1)
        else:
            _pad_mask = None

        # Initial noise: use Cholesky-transformed noise for embodiments in
        # correlated_noise_embodiments, and standard randn for others, matching
        # original behavior. dummy only passes shape/device; _sample_noise creates
        # fresh randn internally.
        dummy = torch.zeros(bsz, self.horizon_steps, self.action_dim, device=device, dtype=dtype)
        action = self._sample_noise(dummy, dtype, embodiment_types)
        if _pad_mask is not None:
            action.masked_fill_(_pad_mask, 0.0)

        # Build action mask + position_ids
        action_mask, action_pos = model.build_action_mask_and_position_ids(
            attention_mask,
            action_len=self.horizon_steps,
            position_ids_prefix=position_ids,
            split_index=attention_mask.size(1),
            dtype=dtype,
            action_causal=self.action_causal,
        )

        # Build AE KV kwargs once — return_kv_cache=False means prefix_kv is never
        # modified by the AE forward, so the same object is safe to reuse across all steps.
        if hasattr(model, "_build_prefix_action_kv"):
            ae_kv_kwargs = {"kv_cache": model._build_prefix_action_kv(vlm_kv, vlm_kv.num_items())}
        else:
            ae_kv_kwargs = {"past_key_values": past_key_values}

        # Euler integration
        delta_t = 1.0 / self.num_inference_steps
        if self.time_convention == "pi_convention":
            t = torch.ones(bsz, device=device, dtype=dtype)
        else:
            t = torch.zeros(bsz, device=device, dtype=dtype)

        for _ in range(self.num_inference_steps):
            with torch.autocast(device.type, enabled=False):
                action_embeds = model.action_expert.embed(action.float())
                time_cond = model.action_expert.encode_time(t)

            action_hidden = model.action_expert(
                inputs_embeds=action_embeds,
                attention_mask=action_mask,
                position_ids=action_pos,
                time_cond=time_cond,
                attn_implementation=model.attn_implementation,
                mixture_name="action",
                **ae_kv_kwargs,
            )

            action_vel = model.action_expert.decode(action_hidden)

            if self.time_convention == "pi_convention":
                action -= delta_t * action_vel
                t -= delta_t
            else:
                action += delta_t * action_vel
                t += delta_t

            if _pad_mask is not None:
                action.masked_fill_(_pad_mask, 0.0)

        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action, -self.final_action_clip_value, self.final_action_clip_value
            )

        return action

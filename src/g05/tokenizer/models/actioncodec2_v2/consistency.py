# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
Token-space consistency regularization for ActionCodecV2 RVQ training.

Semantically equivalent inputs (time-shifted, amplitude-scaled) should produce the
same token sequence. This module provides:

  ConsistencyAugmenter  — generates positive samples with a simple linear schedule
  compute_consistency_loss — computes per-layer Hamming-weighted residual loss

Schedule design
---------------
All augmentation parameters and layer weights interpolate **linearly** from their
``begin`` values (step=0) to their ``end`` values (step = end_fraction * max_steps).
After that point they stay fixed at ``end``.  No hard phase boundaries.

Augmentation strength
---------------------
  begin: delta_max=1 (time shift ±1 frame), eps=0.02 (amplitude ±2%)
  end:   delta_max=3 (time shift ±3 frames), eps=0.10 (amplitude ±10%)

Augmentation types (per sample, independently chosen each step):
  p=0.5  time shift only
  p=0.4  amplitude scale only
  p=0.1  time shift + amplitude scale
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_lists(a: List[float], b: List[float], t: float) -> List[float]:
    return [_lerp(ai, bi, t) for ai, bi in zip(a, b)]


def _default_weights_begin(n_codebooks: int) -> List[float]:
    """
    Default begin weights: only last 2 layers active.
    For n_codebooks=4 → [0.0, 0.0, 0.5, 1.0]
    """
    K = n_codebooks
    w = [0.0] * K
    if K >= 2:
        w[-2] = 0.5
    w[-1] = 1.0
    return w


def _default_weights_end(n_codebooks: int) -> List[float]:
    """
    Default end weights: gentle ramp across all layers.
    For n_codebooks=4 → [0.1, 0.3, 0.7, 1.0]
    """
    K = n_codebooks
    if K == 1:
        return [1.0]
    if K == 4:
        return [0.1, 0.3, 0.7, 1.0]
    # General: exponential ramp ending at 1.0
    w = [round(0.1 + 0.9 * (k / (K - 1)) ** 1.5, 4) for k in range(K)]
    w[-1] = 1.0
    return w


# ── ConsistencyAugmenter ──────────────────────────────────────────────────────


class ConsistencyAugmenter:
    """
    Generates positive (augmented) views of action sequences for consistency training.

    All parameters vary **linearly** from begin→end over the first
    ``end_fraction`` of training, then stay fixed at end values.

    Time shift
    ----------
    Per-sample δ is sampled uniformly from [-delta_max, delta_max] \\ {0}.
    Boundary handling: boundary-replicated (first/last frame repeated).

      delta > 0: x_pos[:, :T-δ] = x[:, δ:]        (look ahead)
                 x_pos[:, T-δ:] = x[:, -1:] repeat  (pad last frame)
      delta < 0: x_pos[:, -δ:]  = x[:, :T+δ]       (look back)
                 x_pos[:, :-δ]  = x[:, :1] repeat   (pad first frame)

    Amplitude scale
    ---------------
    α ~ Uniform(1-eps, 1+eps), single scalar per sample (not per-dim).

    Implementation
    --------------
    Fully vectorized using torch.gather — no Python loops over batch dimension.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: ActionCodecV2Config (or any object with n_codebooks,
                 consistency_delta_max_begin/end, consistency_eps_begin/end,
                 consistency_layer_weights_begin/end, consistency_schedule_end_fraction)
        """
        K = int(cfg.n_codebooks)
        w_begin = list(cfg.consistency_layer_weights_begin or _default_weights_begin(K))
        w_end = list(cfg.consistency_layer_weights_end or _default_weights_end(K))
        assert len(w_begin) == K and len(w_end) == K, (
            f"consistency_layer_weights must have length n_codebooks={K}"
        )
        self._weights_begin = w_begin
        self._weights_end = w_end
        self._delta_max_begin = int(getattr(cfg, "consistency_delta_max_begin", 1))
        self._delta_max_end = int(getattr(cfg, "consistency_delta_max_end", 3))
        self._eps_begin = float(getattr(cfg, "consistency_eps_begin", 0.02))
        self._eps_end = float(getattr(cfg, "consistency_eps_end", 0.10))
        self._begin_fraction = float(getattr(cfg, "consistency_schedule_begin_fraction", 0.0))
        self._end_fraction = float(getattr(cfg, "consistency_schedule_end_fraction", 0.9))
        self._K = K

    def is_active(self, step: int, max_steps: int) -> bool:
        """Returns True only after begin_fraction of training has elapsed."""
        begin_step = int(self._begin_fraction * max(max_steps, 1))
        return step >= begin_step

    def get_schedule(self, step: int, max_steps: int) -> Tuple[int, float, List[float]]:
        """
        Returns (delta_max, epsilon, layer_weights) for the current training step.

        Linear interpolation from begin to end over [begin_fraction, end_fraction] of training.
        Clamped at begin values before begin_fraction, at end values after end_fraction.

        Args:
            step:      current optimizer step (0-indexed)
            max_steps: total training steps

        Returns:
            delta_max:     int  — max time-shift magnitude in frames
            epsilon:       float — amplitude scale half-range
            layer_weights: List[K] — per-RVQ-level loss weights
        """
        max_steps = max(max_steps, 1)
        begin_step = int(self._begin_fraction * max_steps)
        end_step = max(begin_step + 1, int(self._end_fraction * max_steps))
        t = max(0.0, min((step - begin_step) / (end_step - begin_step), 1.0))

        delta_max = max(
            1, round(_lerp(float(self._delta_max_begin), float(self._delta_max_end), t))
        )
        eps = _lerp(self._eps_begin, self._eps_end, t)
        weights = _lerp_lists(self._weights_begin, self._weights_end, t)

        return delta_max, eps, weights

    def augment(
        self,
        x: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> torch.Tensor:
        """
        Generate positive samples from a batch of flat action tensors.

        Fully vectorized — single torch.gather call for time shift,
        single broadcast multiply for amplitude scale.  No CPU-GPU syncs.

        Args:
            x:         (B, T, D) — flat action tensor (pre-normalization)
            step:      current optimizer step
            max_steps: total training steps (for progressive schedule)

        Returns:
            x_pos:     (B, T, D) — augmented batch, same device/dtype as x
        """
        B, T, D = x.shape
        delta_max, eps, _ = self.get_schedule(step, max_steps)

        # Per-sample augmentation type (3-way categorical)
        rng3 = torch.rand(B, device=x.device)
        use_shift_only = rng3 < 0.5
        use_scale_only = (rng3 >= 0.5) & (rng3 < 0.9)
        use_both = rng3 >= 0.9
        do_shift = use_shift_only | use_both  # p = 0.6
        do_scale = use_scale_only | use_both  # p = 0.5

        # ── Time shift (vectorized) ──────────────────────────────────────────
        # Sample per-sample δ ∈ [-delta_max, delta_max] \ {0}; zero out non-shifted.
        candidates = list(range(-delta_max, 0)) + list(range(1, delta_max + 1))
        rand_idx = torch.randint(0, len(candidates), (B,), device=x.device)
        # deltas[b] = chosen δ for shifted samples, 0 for others (→ identity gather)
        deltas = x.new_tensor(candidates, dtype=torch.long)[rand_idx] * do_shift.long()

        # src_idx[b, t] = clamp(t + δ[b], 0, T-1)
        #   δ>0: look-ahead with last-frame padding
        #   δ<0: look-back with first-frame padding
        #   δ=0: identity (for non-shifted samples)
        t_idx = torch.arange(T, device=x.device, dtype=torch.long)  # (T,)
        src = (t_idx.unsqueeze(0) + deltas.unsqueeze(1)).clamp(0, T - 1)  # (B, T)
        x_pos = x.gather(1, src.unsqueeze(-1).expand(B, T, D))  # (B, T, D)

        # ── Amplitude scale (vectorized) ─────────────────────────────────────
        # α ~ Uniform(1-eps, 1+eps); non-scaled samples get α=1.0 (identity).
        if eps > 0.0:
            alphas = (1.0 - eps) + 2.0 * eps * torch.rand(B, device=x.device)  # (B,)
            scale = 1.0 + (alphas - 1.0) * do_scale.to(x.dtype)  # (B,)
            x_pos = x_pos * scale[:, None, None]

        return x_pos


# ── Consistency loss ──────────────────────────────────────────────────────────


def compute_consistency_loss(
    consist_residuals: List[torch.Tensor],
    level_codes: List[torch.Tensor],
    BN: int,
    layer_weights: List[float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute per-layer token consistency regularization loss with first-divergence masking.

    **First-divergence masking (prefix_match):**
    For each (sample, time position), only the FIRST layer where codes differ is
    penalized.  Later layers are masked out, because their residuals are computed
    relative to different quantization reference points (incommensurable residuals).

    Formally:
        prefix_match_k[b,t] = 1  iff  c_orig[j][b,t] == c_pos[j][b,t]  for all j < k

    So layer k only contributes where all earlier layers agreed AND layer k itself
    differs:
        loss_k = mean(prefix_match_k * delta_k * ||r_orig_k - r_pos_k||_2)

    Natural curriculum: early training pushes layer 0 (most gradient here since
    everything diverges at layer 0); as layer 0 stabilises, layer 1 takes over, etc.

    Grad flow:
      - prefix_match and delta_k are both stop_grad
      - Gradient flows through consist_residuals[k] → encoder output

    Args:
        consist_residuals: List[K] of (2*BN, codebook_dim, T) pre-quantization residuals
            projected to codebook space (after in_proj). Gradient flows through in_proj
            back to the encoder — aligned with VQ decision boundary.
        level_codes: List[K] of (2*BN, T) integer codebook indices.
        BN: number of original samples (= B * N_keys); pos samples are [BN:].
        layer_weights: List[K] of static float weights (relative layer importance).
            With first-divergence masking the natural curriculum is self-emerging, so
            uniform weights [1, 1, 1, 1] work well — no need for time-varying schedules.

    Returns:
        loss:    scalar differentiable tensor (weighted sum over layers)
        metrics: dict with:
            'consist/loss'              weighted total
            'consist/hamming_dist'      expected differing token count per key (0 to n_codebooks*T)
            'consist/tcr_layer_{k}'     per-layer token change rate (unmasked)
            'consist/active_frac_{k}'   fraction of positions active (prefix_match * delta)
            'consist/loss_layer_{k}'    per-layer unweighted loss (with masking)
    """
    K = len(consist_residuals)
    assert len(level_codes) == K and len(layer_weights) == K

    device = consist_residuals[0].device
    total_loss = torch.tensor(0.0, device=device)
    metrics: Dict[str, torch.Tensor] = {}
    hamming = 0.0

    # prefix_match[b, t] = 1 iff all layers 0..k-1 had matching codes at (b, t)
    # Shape: (BN, T) — updated after each layer
    r0 = consist_residuals[0]
    T = r0.shape[-1]
    BN_T = (BN, T)
    prefix_match = torch.ones(BN_T, device=device, dtype=torch.float)

    for k in range(K):
        r = consist_residuals[k]  # (2*BN, D, T)
        c = level_codes[k]  # (2*BN, T)

        r_orig = r[:BN]  # (BN, D, T)
        r_pos = r[BN:]  # (BN, D, T)
        c_orig = c[:BN]  # (BN, T)
        c_pos = c[BN:]  # (BN, T)

        # delta_k: positions where codes differ (stop_grad)
        delta_k = (c_orig != c_pos).float().detach()  # (BN, T)

        # Unmasked TCR (for monitoring — shows true divergence rate)
        tcr = float(delta_k.mean().item())
        hamming += tcr

        # Per-position L2 residual distance.
        # r_orig is stop-grad: only r_pos moves toward r_orig, not the reverse.
        # r_orig is strongly anchored by reconstruction loss (stable inside a Voronoi
        # cell) — the ideal fixed target. r_pos (augmented, no recon loss) is pulled
        # into r_orig's cell, with gradient flowing through encoder(x_pos).
        r_diff = (r_pos - r_orig.detach()).norm(dim=1)  # (BN, T)

        # Active mask: first-divergence gate × token-change mask
        active = prefix_match * delta_k  # (BN, T) — stop_grad by construction
        layer_loss = (active * r_diff).mean()

        w = float(layer_weights[k])
        total_loss = total_loss + w * layer_loss

        active_frac = float(active.mean().item())
        metrics[f"consist/tcr_layer_{k}"] = torch.tensor(tcr)
        metrics[f"consist/active_frac_{k}"] = torch.tensor(active_frac)
        metrics[f"consist/loss_layer_{k}"] = layer_loss.detach()

        # Update prefix_match: positions where layer k differed are permanently masked
        prefix_match = prefix_match * (c_orig == c_pos).float().detach()

    metrics["consist/loss"] = total_loss.detach()
    metrics["consist/hamming_dist"] = torch.tensor(hamming * T)

    return total_loss, metrics

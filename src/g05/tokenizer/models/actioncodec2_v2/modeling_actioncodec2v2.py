# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
ActionCodecV2Model — HuggingFace PreTrainedModel wrapper around the ActionCodecV2 architecture.

Shape flow (defaults):
    input dict: {key: (B, T=32, D_i)}
        ↓ pad/clip each value to max_component_dim=9 → {key: (B, 32, 9)}
        ↓ stack all N keys along batch: (B*N, 32, 9)
        ↓ [optional] Block DCT along time axis
        ↓ Pretransform (independent mode):
              rearrange '(b) (lh h) a → b lh h a', lh=horizon_patch_size=8
              → (B*N, lh=8, h=4, a=9)
        ↓ conv_in  Conv2d(lh=8, C=128, kernel=(1,2))  A: 9 → 8
              → (B*N, 128, 4, 8)
        ↓ ActionEncoder (strides=[[1,1],[2,1],[2,1]])
              H: 4→4→2→1
              → (B*N, latent_dim=128, 1, 8)
        ↓ flatten spatial: (B*N, latent_dim, 8)
        ↓ RVQ: z_q (B*N, latent_dim, 8), codes (B*N, n_codebooks=4, 8)
        ↓ unflatten: (B*N, latent_dim, 1, 8)
        ↓ ActionDecoder (reversed strides)
              H: 1→2→4→4
              → (B*N, 128, 4, 8)
        ↓ conv_out  ConvTranspose2d(C=128, lh=8, kernel=(1,2))  A: 8 → 9
              → (B*N, 8, 4, 9)
        ↓ Reverse pretransform:
              rearrange 'b lh h a → b (lh h) a'   (note: lh fast, h slow)
              → (B*N, 32, 9)
        ↓ [optional] Inverse DCT
        ↓ slice back to D_i
        ↓ unstack batch → {key: (B, 32, D_i)}

    tokens per key = code_h × code_a × n_codebooks = 1 × 8 × 4 = 32
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel

from .configuration_actioncodec2v2 import ActionCodecV2Config
from .consistency import compute_consistency_loss
from .modular_actioncodec2v2 import (
    ActionDecoder,
    ActionEncoder,
    Attention,
    BlockDCT,
    FSQQuantize,
    ResidualVectorQuantize,
)


class ActionCodecV2Model(PreTrainedModel):
    """
    Action tokenizer model based on FasterV2 architecture with generic dict input.

    All keys share the same encoder/decoder/RVQ weights because after pad/clip
    every key has the same shape.  The model does not know about key semantics.

    Args:
        config: ActionCodecV2Config
    """

    config_class = ActionCodecV2Config
    supports_gradient_checkpointing = False

    def __init__(self, config: ActionCodecV2Config):
        super().__init__(config)
        cfg = config

        # ── BlockDCT (no parameters) ────────────────────────────────────────────
        if cfg.use_block_dct:
            self.block_dct = BlockDCT(block_size=cfg.block_dct_block_size)
        else:
            self.block_dct = None

        # ── conv_in: map A axis from max_component_dim → code_a ────────────────
        # Input channels = horizon_patch_size (lh dimension treated as channels)
        # kernel=(1, conv_in_action_kernel) acts on H=1, A axis
        self.conv_in = nn.Conv2d(
            in_channels=cfg.horizon_patch_size,
            out_channels=cfg.encoder_channels,
            kernel_size=(1, cfg.conv_in_action_kernel),
            stride=(1, 1),
            padding=0,
        )

        # ── Encoder ─────────────────────────────────────────────────────────────
        self.encoder = ActionEncoder(
            in_channels=cfg.encoder_channels,
            c_mults=cfg.c_mults,
            strides=cfg.strides,
            transformer_depths=cfg.transformer_depths,
            latent_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            dim_heads=cfg.dim_heads,
            ffn_mult=cfg.ffn_mult,
            use_layer_scale=cfg.use_layer_scale,
            layer_scale_init=cfg.layer_scale_init,
            use_qk_norm=cfg.use_qk_norm,
            rope_base=cfg.rope_base,
        )

        # ── Residual Vector Quantizer ────────────────────────────────────────────
        self.rvq = ResidualVectorQuantize(
            input_dim=cfg.latent_dim,
            n_codebooks=cfg.n_codebooks,
            codebook_size=cfg.codebook_size,
            codebook_dim=cfg.codebook_dim,
            commitment=cfg.commitment_loss_weight,
            decay=cfg.ema_decay,
            threshold_ema_dead=cfg.threshold_ema_dead,
            quantizer_dropout=cfg.quantizer_dropout,
            use_rotation_trick=cfg.use_rotation_trick,
            use_fsq=cfg.use_fsq,
            fsq_levels=cfg.fsq_levels,
        )

        # ── Decoder ─────────────────────────────────────────────────────────────
        self.decoder = ActionDecoder(
            out_channels=cfg.encoder_channels,
            c_mults=cfg.c_mults,
            strides=cfg.strides,
            transformer_depths=cfg.transformer_depths,
            latent_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            dim_heads=cfg.dim_heads,
            ffn_mult=cfg.ffn_mult,
            use_layer_scale=cfg.use_layer_scale,
            layer_scale_init=cfg.layer_scale_init,
            use_qk_norm=cfg.use_qk_norm,
            rope_base=cfg.rope_base,
        )

        # ── conv_out: map A axis back from code_a → max_component_dim ──────────
        # Output channels = horizon_patch_size (to reverse conv_in)
        self.conv_out = nn.ConvTranspose2d(
            in_channels=cfg.encoder_channels,
            out_channels=cfg.horizon_patch_size,
            kernel_size=(1, cfg.conv_in_action_kernel),
            stride=(1, 1),
            padding=0,
        )

        # ── Weight init ─────────────────────────────────────────────────────────
        self._init_weights_custom()
        if cfg.use_lipschitz_encoder:
            self._apply_spectral_norm_encoder()
        self.post_init()
        self._log_init_summary(cfg)

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights_custom(self):
        """Kaiming init for conv_in/conv_out (called before post_init)."""
        for m in [self.conv_in, self.conv_out]:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_spectral_norm_encoder(self):
        """
        Apply spectral_norm to all Linear and Conv2d layers in conv_in + ActionEncoder,
        skipping Attention submodules entirely (to_qkv and to_out are excluded).

        Scope: conv_in, ActionEncoder (DownBlock2D conv layers, GEGLUFFFN w_up/w_down,
               ActionEncoder.out_proj).
        Not applied to: decoder, conv_out, RVQ projections, LayerNorm, Embedding.
        """

        def _walk(module: nn.Module) -> None:
            if isinstance(module, Attention):
                return  # skip entire attention subtree (to_qkv, to_out)
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.utils.spectral_norm(module)
                return  # leaf node, no children to recurse into
            for child in module.children():
                _walk(child)

        _walk(self.conv_in)
        _walk(self.encoder)

    def _init_weights(self, module: nn.Module):
        """HuggingFace post_init hook — handles Linear / LayerNorm / Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Static helpers ─────────────────────────────────────────────────────────

    def _log_init_summary(self, cfg: "ActionCodecV2Config") -> None:
        def _fmt(n: int) -> str:
            return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.1f}K"

        def _p(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        W = 58  # inner width (between ║ chars)

        def row(label: str, value: str) -> str:
            content = f"  {label:<14s}{value}"
            return f"║{content:<{W}s}║"

        def sep(char: str = "═") -> str:
            return f"╠{char * W}╣"

        ch = (
            [cfg.encoder_channels]
            + [cfg.encoder_channels * m for m in cfg.c_mults]
            + [cfg.latent_dim]
        )
        ch_str = " → ".join(str(c) for c in ch)

        codebook_total = cfg.n_codebooks * cfg.codebook_size * cfg.codebook_dim

        lines = [
            f"╔{'═' * W}╗",
            f"║{'ActionCodecV2  —  Init Summary':^{W}s}║",
            sep(),
            row(
                "horizon",
                f"{cfg.horizon}  patch={cfg.horizon_patch_size}  D≤{cfg.max_component_dim}"
                f"  DCT={'on' if cfg.use_block_dct else 'off'}(bs={cfg.block_dct_block_size})",
            ),
            row("channels", ch_str),
            row("strides", str(cfg.strides)),
            row(
                "depths",
                f"{cfg.transformer_depths}  heads={cfg.num_heads}×{cfg.dim_heads}d"
                f"  FFN×{cfg.ffn_mult:.0f}",
            ),
            row(
                "FSQ" if cfg.use_fsq else "RVQ",
                (
                    f"{cfg.n_codebooks}×levels={cfg.fsq_levels}"
                    f"  size={math.prod(cfg.fsq_levels)}  drop={cfg.quantizer_dropout}"
                )
                if cfg.use_fsq
                else (
                    f"{cfg.n_codebooks}×{cfg.codebook_size}×{cfg.codebook_dim}d"
                    f"  ({_fmt(codebook_total)})  drop={cfg.quantizer_dropout}"
                    f"  ema={cfg.ema_decay}"
                ),
            ),
            row(
                "tokens/key",
                f"{cfg.tokens_per_key}  "
                f"(code_h={cfg.code_h} × code_a={cfg.code_a} × {cfg.n_codebooks}cb)",
            ),
            sep(),
            row("conv_in/out", _fmt(_p(self.conv_in) + _p(self.conv_out))),
            row("encoder", _fmt(_p(self.encoder))),
            row("decoder", _fmt(_p(self.decoder))),
            row("rvq", _fmt(_p(self.rvq))),
            row("TOTAL", _fmt(_p(self))),
        ]

        if cfg.consistency_loss_weight > 0:
            lines += [
                sep(),
                row(
                    "consistency",
                    f"λ={cfg.consistency_loss_weight}"
                    f"  δ={cfg.consistency_delta_max_begin}→{cfg.consistency_delta_max_end}"
                    f"  ε={cfg.consistency_eps_begin}→{cfg.consistency_eps_end}",
                ),
                row(
                    "schedule",
                    f"begin={cfg.consistency_schedule_begin_fraction}"
                    f"  end={cfg.consistency_schedule_end_fraction}",
                ),
                row("w_begin", str(cfg.consistency_layer_weights_begin)),
                row("w_end", str(cfg.consistency_layer_weights_end)),
            ]

        lines.append(f"╚{'═' * W}╝")
        self._init_summary_lines = lines
        logger.debug("\n" + "\n".join(lines))

    @staticmethod
    def _pad_or_clip(x: torch.Tensor, max_d: int) -> torch.Tensor:
        """Pad or clip the last dimension to ``max_d``."""
        d = x.shape[-1]
        if d < max_d:
            return F.pad(x, (0, max_d - d))
        elif d > max_d:
            return x[..., :max_d]
        return x

    @staticmethod
    def _pretransform(x: torch.Tensor, horizon_patch_size: int) -> torch.Tensor:
        """
        Independent-mode pretransform.

        (B, T, A) → (B, lh, h, A)
        where lh = horizon_patch_size, h = T / lh.
        lh becomes the channel axis fed into conv_in.
        """
        # T must be divisible by lh (guaranteed by config assertion)
        return rearrange(x, "b (h lh) a -> b lh h a", lh=horizon_patch_size)

    @staticmethod
    def _reverse_pretransform(x: torch.Tensor) -> torch.Tensor:
        """
        Reverse of _pretransform.

        (B, lh, h, A) → (B, h*lh, A)
        """
        return rearrange(x, "b lh h a -> b (h lh) a")

    # ── Forward building blocks ────────────────────────────────────────────────

    def _encode_tensor(self, x: torch.Tensor, return_level_data: bool = False):
        """
        Encode a single padded tensor through the full pipeline up to RVQ.

        Args:
            x:                (B, T, max_component_dim)  — already padded, possibly DCT'd
            return_level_data: if True, return per-level consistency data from RVQ
        Returns (default):
            z_q:        (B, latent_dim, code_h * code_a) — quantized latent (flattened spatial)
            codes:      (B, n_codebooks, code_h * code_a)
            commit_loss: scalar
        Returns (return_level_data=True), additionally:
            consist_residuals: List[K] of (B, D, T) — detached-chain pre-quant residuals
            level_codes:       List[K] of (B, T) — per-level integer codes
        """
        B, T, D = x.shape

        # Pretransform: (B, T, D) → (B, lh, h, D)
        x2d = self._pretransform(x, self.config.horizon_patch_size)
        # x2d: (B, lh, h, D)

        # conv_in: (B, lh, h, D) → (B, C, h, code_a)  [A: D→code_a]
        z = self.conv_in(x2d)

        # Encoder: (B, C, h, code_a) → (B, latent_dim, code_h, code_a)
        z = self.encoder(z)

        # Flatten spatial dims for RVQ: (B, latent_dim, code_h*code_a)
        Bz, Dz, Hc, Ac = z.shape
        z_flat = z.reshape(Bz, Dz, Hc * Ac)

        # RVQ
        if return_level_data:
            z_q, codes, commit_loss, consist_residuals, level_codes = self.rvq(
                z_flat, return_level_data=True
            )
            return z_q, codes, commit_loss, consist_residuals, level_codes

        z_q, codes, commit_loss = self.rvq(z_flat)
        return z_q, codes, commit_loss

    def _decode_zq(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantised latent back to the padded action space.

        Args:
            z_q: (B, latent_dim, code_h * code_a)
        Returns:
            recon: (B, T, max_component_dim)  — before IDCT / before unpad
        """
        cfg = self.config
        code_h = cfg.code_h
        code_a = cfg.code_a

        # Unflatten spatial: (B, latent_dim, code_h, code_a)
        z_2d = z_q.reshape(z_q.shape[0], z_q.shape[1], code_h, code_a)

        # Decoder: (B, latent_dim, code_h, code_a) → (B, C, h, code_a)
        x_dec = self.decoder(z_2d)

        # conv_out: (B, C, h, code_a) → (B, lh, h, max_component_dim)
        x_out = self.conv_out(x_dec)

        # Reverse pretransform: (B, lh, h, D) → (B, T, D)
        recon = self._reverse_pretransform(x_out)
        return recon

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        component_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a dict of action components to RVQ codes.

        Args:
            component_dict: {key: (B, T, D_i)}  — D_i can vary per key

        Returns:
            codes_dict: {key: (B, n_codebooks, code_h * code_a)}
        """
        cfg = self.config
        key_names = list(component_dict.keys())
        N = len(key_names)

        # Collect and pad all components: list of (B, T, max_component_dim)
        B = next(iter(component_dict.values())).shape[0]
        padded = []
        for k in key_names:
            v = component_dict[k].float()
            padded.append(self._pad_or_clip(v, cfg.max_component_dim))

        # Stack along batch dim: (B*N, T, max_component_dim)
        x_batch = torch.cat(padded, dim=0)

        # Block DCT
        if self.block_dct is not None:
            T = x_batch.shape[1]
            x_batch = self.block_dct.dct(x_batch)
            # dct may pad T; store for idct
        else:
            T = None

        # Encode + RVQ
        _, codes, _ = self._encode_tensor(x_batch)
        # codes: (B*N, n_codebooks, code_len)

        # Split back into per-key dicts
        codes_dict = {}
        for i, k in enumerate(key_names):
            codes_dict[k] = codes[i * B : (i + 1) * B]

        return codes_dict

    @torch.no_grad()
    def decode(
        self,
        codes_dict: Dict[str, torch.Tensor],
        d_original: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode RVQ codes back to action components.

        Args:
            codes_dict:  {key: (B, n_codebooks, code_len)}
            d_original:  {key: int}  original D_i per key; if provided, slices
                         the output to the original dimension.

        Returns:
            recon_dict: {key: (B, T, D_i or max_component_dim)}
        """
        cfg = self.config
        key_names = list(codes_dict.keys())
        N = len(key_names)
        B = next(iter(codes_dict.values())).shape[0]

        # Stack codes: (B*N, n_codebooks, code_len)
        codes_batch = torch.cat([codes_dict[k] for k in key_names], dim=0)

        # Reconstruct z_q from codes
        z_q = self.rvq.from_codes(codes_batch)  # (B*N, latent_dim, code_len)

        # Decode
        recon_batch = self._decode_zq(z_q)  # (B*N, T_padded, max_component_dim)

        # Inverse DCT
        if self.block_dct is not None:
            recon_batch = self.block_dct.idct(recon_batch, cfg.horizon)

        recon_batch = recon_batch[:, : cfg.horizon, :]

        # Split per key and optionally unpad
        recon_dict = {}
        for i, k in enumerate(key_names):
            r = recon_batch[i * B : (i + 1) * B]
            if d_original is not None and k in d_original:
                r = r[..., : d_original[k]]
            recon_dict[k] = r

        return recon_dict

    def forward(
        self,
        component_dict: Dict[str, torch.Tensor],
        d_original: Optional[Dict[str, int]] = None,
        x_pos_dict: Optional[Dict[str, torch.Tensor]] = None,
        layer_weights: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Full autoencoder forward pass: encode → RVQ → decode → loss.

        Args:
            component_dict: {key: (B, T, D_i)}
            d_original:     {key: int}  original D_i for reconstruction loss
            x_pos_dict:     optional augmented view {key: (B, T, D_i)} for consistency loss.
                            When provided, orig and pos are concatenated along batch and
                            encoded in a single forward pass.
            layer_weights:  per-RVQ-level weights for the consistency loss (List[K]).

        Returns:
            recon_dict:  {key: (B, T, D_i or max_component_dim)}
            codes_dict:  {key: (B, n_codebooks, code_len)}
            loss_dict:   {"loss": scalar, "commitment_loss": scalar, "reconstruction_loss": scalar}
        """
        cfg = self.config
        key_names = list(component_dict.keys())
        N = len(key_names)
        B = next(iter(component_dict.values())).shape[0]

        # ── 1. Pad/clip all keys and keep padded targets for loss ───────────────
        padded_inputs = []
        for k in key_names:
            v = component_dict[k].float()
            padded_inputs.append(self._pad_or_clip(v, cfg.max_component_dim))

        # Stack: (B*N, T, max_component_dim)
        x_batch = torch.cat(padded_inputs, dim=0)
        # Store actual input T *before* DCT (DCT may zero-pad to next multiple of block_size)
        T_input = x_batch.shape[1]
        x_target = x_batch.clone()  # reconstruction target (before DCT)

        # ── 2. Block DCT (optional) ──────────────────────────────────────────────
        if self.block_dct is not None:
            x_batch = self.block_dct.dct(x_batch)

        # ── 3. Encode + RVQ ──────────────────────────────────────────────────────
        if x_pos_dict is not None:
            # Build x_pos_batch with the same pad/DCT pipeline as x_batch
            padded_pos = []
            for k in key_names:
                v = x_pos_dict[k].float()
                padded_pos.append(self._pad_or_clip(v, cfg.max_component_dim))
            x_pos_batch = torch.cat(padded_pos, dim=0)
            if self.block_dct is not None:
                x_pos_batch = self.block_dct.dct(x_pos_batch)

            # One forward through encoder + RVQ for both orig and pos
            BN = B * N
            x_cat = torch.cat([x_batch, x_pos_batch], dim=0)
            z_q_cat, codes_cat, commit_loss, consist_residuals, level_codes = self._encode_tensor(
                x_cat, return_level_data=True
            )
            z_q = z_q_cat[:BN]
            codes = codes_cat[:BN]
        else:
            z_q, codes, commit_loss = self._encode_tensor(x_batch)

        # ── 4. Decode ────────────────────────────────────────────────────────────
        recon_2d = self._decode_zq(z_q)  # (B*N, T_padded, max_component_dim)

        # ── 5. Inverse DCT ───────────────────────────────────────────────────────
        if self.block_dct is not None:
            recon_2d = self.block_dct.idct(recon_2d, T_input)

        recon_2d = recon_2d[:, :T_input, :]  # trim to actual input length

        # ── 6. Reconstruction loss (MSE in time-domain padded space) ─────────────
        recon_loss = F.mse_loss(recon_2d, x_target)

        total_loss = (
            cfg.reconstruction_loss_weight * recon_loss + cfg.commitment_loss_weight * commit_loss
        )

        loss_dict = {
            "loss": total_loss,
            "reconstruction_loss": recon_loss.detach(),
            "commitment_loss": commit_loss.detach(),
        }

        # ── 6a. Per-key reconstruction loss ──────────────────────────────────────
        for i, k in enumerate(key_names):
            r_k = recon_2d[i * B : (i + 1) * B]
            t_k = x_target[i * B : (i + 1) * B]
            # Compare only on the original (non-padded) dims so padding zeros don't dilute the metric
            d_k = (
                d_original[k]
                if (d_original is not None and k in d_original)
                else cfg.max_component_dim
            )
            loss_dict[f"recon/{k}"] = F.mse_loss(r_k[..., :d_k], t_k[..., :d_k]).detach()

        # ── 6b. Codebook utilisation stats (per RVQ level) ───────────────────────
        for lvl, vq in enumerate(self.rvq.quantizers):
            if isinstance(vq, FSQQuantize):
                # FSQ has no learnable codebook; skip EMA-based stats
                loss_dict[f"codebook/fsq_l{lvl}"] = torch.tensor(1.0)
                continue
            cs = vq.cluster_size.float()
            total = cs.sum()
            if total > 0:
                p = cs / total
                perplexity = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
                utilization = float((cs >= vq.threshold_ema_dead).float().mean())
                dead_frac = float((cs < vq.threshold_ema_dead).float().mean())
            else:
                perplexity, utilization, dead_frac = 1.0, 0.0, 1.0
            loss_dict[f"codebook/perplexity_l{lvl}"] = torch.tensor(perplexity)
            loss_dict[f"codebook/utilization_l{lvl}"] = torch.tensor(utilization)

        # ── 6c. Token consistency loss ────────────────────────────────────────────
        if x_pos_dict is not None and cfg.consistency_loss_weight > 0:
            w = layer_weights if layer_weights is not None else [1.0] * cfg.n_codebooks
            consist_loss, consist_metrics = compute_consistency_loss(
                consist_residuals, level_codes, BN, w
            )
            loss_dict["loss"] = loss_dict["loss"] + cfg.consistency_loss_weight * consist_loss
            loss_dict.update(consist_metrics)

        # ── 7. Split back into per-key dicts ──────────────────────────────────────
        codes_dict: Dict[str, torch.Tensor] = {}
        recon_dict: Dict[str, torch.Tensor] = {}
        for i, k in enumerate(key_names):
            codes_dict[k] = codes[i * B : (i + 1) * B]
            r = recon_2d[i * B : (i + 1) * B]
            if d_original is not None and k in d_original:
                r = r[..., : d_original[k]]
            recon_dict[k] = r

        return recon_dict, codes_dict, loss_dict

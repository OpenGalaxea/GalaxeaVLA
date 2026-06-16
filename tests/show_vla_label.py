#!/usr/bin/env python3
"""VLA label lifecycle visualization.

Given a VLA training task config, generate the exact full label seen by the model
through the real processor + InputPreprocessor, then show:
  0. Config overview
  1. Full VLA label, including input_ids + attention_mask region distribution
  2. Action token details: indicator + part markers + codes + BAR blocks
  3. Action encode/decode roundtrip: encode -> decode -> L1 error
  4. Attention mask visualization: causal mask heatmap

Usage:
    # Basic usage, automatically sets MAX_DATASETS=1
    python tests/show_vla_label.py --task pretrain/bench/ppbench_g05v2

    # Full yaml paths are supported
    python tests/show_vla_label.py --task configs/task/pretrain/bench/ppbench_g05v2.yaml

    # Select sample index
    python tests/show_vla_label.py --task pretrain/bench/ppbench_g05v2 --sample-idx 3

    # Save attention mask plot
    python tests/show_vla_label.py --task pretrain/bench/ppbench_g05v2 --save-dir ./viz_output

    # Extra Hydra overrides
    python tests/show_vla_label.py --task pretrain/bench/ppbench_g05v2 --override model.batch_size=1
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()


# ============================================================================
# Region emoji + color mapping.
# ============================================================================

TOKEN_REGION_INFO = {
    0: ("PAD", "⬛", "\033[90m"),  # gray
    1: ("IMAGE", "🖼️ ", "\033[36m"),  # cyan
    2: ("PROPRIO", "🦾", "\033[35m"),  # purple
    3: ("ACTION", "🎯", "\033[33m"),  # yellow
    4: ("TEXT", "📝", "\033[32m"),  # green
    5: ("COT", "💭", "\033[34m"),  # blue
    6: ("PRED_TEXT", "🔮", "\033[31m"),  # red
}
BAR_BLOCK_INFO = ("BAR_BLK", "🧩", "\033[93m")  # bright yellow
RESET = "\033[0m"


def _region_info(attn_val):
    """Return (name, emoji, color_code)."""
    v_int = int(attn_val)
    v_float = float(attn_val)
    if v_int == 3 and v_float != 3.0:
        return BAR_BLOCK_INFO
    return TOKEN_REGION_INFO.get(v_int, (f"UNK({attn_val})", "❓", "\033[0m"))


def _region_name(attn_val):
    return _region_info(attn_val)[0]


def _bar(value, max_value, width=20, fill="█", empty="░"):
    """Text progress bar."""
    if max_value == 0:
        return empty * width
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


# ============================================================================
# 1. Config loading
# ============================================================================


def _normalize_task_name(task: str) -> str:
    """Normalize different task path formats to a Hydra task name.

    Supports:
        pretrain/bench/ppbench_g05v2                            -> unchanged
        configs/task/pretrain/bench/ppbench_g05v2.yaml          -> pretrain/bench/ppbench_g05v2
        configs/task/pretrain/bench/ppbench_g05v2               -> pretrain/bench/ppbench_g05v2
        /abs/path/configs/task/pretrain/bench/ppbench_g05v2.yaml -> pretrain/bench/ppbench_g05v2
    """
    task = task.removesuffix(".yaml").removesuffix(".yml")
    marker = "configs/task/"
    idx = task.find(marker)
    if idx != -1:
        task = task[idx + len(marker) :]
    return task


def load_config(task: str, overrides: list[str] | None = None):
    """Hydra compose, automatically setting MAX_DATASETS=1 and MAX_EMBODIMENTS=1."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    task = _normalize_task_name(task)

    config_dir = str(PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"task={task}"] + (overrides or []),
        )

    # Delete keys depending on ${hydra:...}; compose API has no HydraConfig.
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = "/tmp/show_vla_label"
    if "logger" in cfg:
        del cfg.logger
    if "hydra" in cfg:
        del cfg.hydra
    OmegaConf.resolve(cfg)

    # Test mode: truncate to 1 dataset, 1 embodiment
    from g05.utils.eval.eval_utils import truncate_datasets, truncate_embodiments

    truncate_embodiments(cfg, 1)
    truncate_datasets(cfg, 1)

    return cfg, task


# ============================================================================
# 2. Data loading
# ============================================================================


def load_data(cfg, device, sample_idx=0):
    """Instantiate dataset + processor and load one batch."""
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.data.processor_utils import build_processors, instantiate_dataset

    print("  📂 Loading dataset...")
    dataset = instantiate_dataset(cfg, is_training_set=True)

    print("  🔧 Building processor...")
    processor = build_processors(cfg)

    stats_path = cfg.get("datastatics_path", None)
    if stats_path and Path(stats_path).exists():
        print(f"  📊 Loading stats: {stats_path}")
        dataset_stats = load_dataset_stats_from_json(Path(stats_path))
    else:
        print("  📊 Computing dataset stats...")
        dataset_stats = dataset.get_dataset_stats(processor)

    processor.set_normalizer_from_stats(dataset_stats)
    dataset.set_processor(processor)
    processor.eval()

    print(f"  🎲 Fetching sample idx={sample_idx}...")
    data = dataset[sample_idx]

    samples = [data["samples"]] if "samples" in data else [data]
    action_gt = data.get("action")
    if action_gt is not None and action_gt.dim() == 2:
        action_gt = action_gt.unsqueeze(0)
    action_dim_is_pad = data.get("action_dim_is_pad")
    if action_dim_is_pad is not None and action_dim_is_pad.dim() == 1:
        action_dim_is_pad = action_dim_is_pad.unsqueeze(0)

    # parts_meta: {"left_arm": 8, "right_arm": 8, ...}, taken from processor.
    parts_meta = data.get("action_parts_meta", None)

    return samples, action_gt, action_dim_is_pad, parts_meta


# ============================================================================
# 3. Build lightweight components
# ============================================================================


def build_components(cfg, device):
    """Build InputPreprocessor + ActionTokenizer without loading model weights."""
    from g05.models.g05.io.input_preprocessor import InputPreprocessor

    model_cfg = cfg.model.model_arch
    hf_path = model_cfg.get("hf_processor_path", None) or model_cfg.pretrained_model_path

    pp_cfg = model_cfg.input_preprocessor
    hf_processor_class = model_cfg.get(
        "hf_processor_class",
        "g05.models.g05.paligemma.processing.PaliGemmaProcessor",
    )
    model_type = "qwen35" if "qwen35" in (hf_processor_class or "").lower() else "paligemma"
    preprocessor = InputPreprocessor(
        hf_processor_path=hf_path,
        hf_processor_class=hf_processor_class,
        action_tokenizer_class=model_cfg.action_tokenizer,
        at_config=model_cfg.AT_CONFIG,
        model_cfg=model_cfg,
        pad_token_id=model_cfg.pad_token_id,
        image_token_index=model_cfg.image_token_index,
        num_image_tokens=model_cfg.vision.num_image_tokens,
        input_action_corruption=pp_cfg.input_action_corruption,
        batchify_action=pp_cfg.batchify_action,
        pred_eov=pp_cfg.get("pred_eov", False),
        pi05_ft_mode=pp_cfg.get("pi05_ft_mode", False),
        proprio_encoder=model_cfg.proprio_encoder,
        model_type=model_type,
    )
    action_tokenizer = preprocessor.action_tokenizer

    action_tokenizer.to(device)

    from g05.models.g05.helpers.mask_helper import MaskHelper

    mask_helper = MaskHelper(model_cfg)

    return preprocessor, action_tokenizer, mask_helper


# ============================================================================
# Section 0: Config overview
# ============================================================================


def show_config_summary(cfg, task_name: str):
    """Print key task/model/tokenizer/data parameters."""
    model_cfg = cfg.model.model_arch
    at_cfg = model_cfg.AT_CONFIG
    is_bar = at_cfg.get("block_wise_autoregressive", False)
    # WBC detection: AT_CONFIG has parts_meta, or vqvae_type contains wbc/Parts.
    vqvae_type = str(at_cfg.get("vqvae_type", ""))
    is_wbc = (
        bool(at_cfg.get("parts_meta"))
        or "wbc" in vqvae_type.lower()
        or "parts" in vqvae_type.lower()
    )
    has_shuffle = at_cfg.get("part_order_shuffle", False)
    has_indicator = at_cfg.get("use_indicator_tokens", False)

    print("\n" + "═" * 70)
    print("🔬 VLA Label Lifecycle Visualization")
    print("═" * 70)

    print(f"\n  📋 Task Config")
    print(f"     Task:           {task_name}")
    print(f"     Batch size:     {cfg.model.batch_size}")
    print(f"     Max tokens:     {model_cfg.max_chunk_token_length}")

    print(f"\n  🤖 Model")
    print(f"     Action dim:     {model_cfg.action_dim}")
    print(f"     Horizon:        {model_cfg.horizon_steps}")
    print(
        f"     Discrete:       {'✅' if model_cfg.discrete_action else '❌'}  |  Continuous: {'✅' if model_cfg.continuous_action else '❌'}"
    )

    print(f"\n  🎰 Action Tokenizer")
    print(f"     Class:          {model_cfg.action_tokenizer.split('.')[-1]}")
    print(
        f"     BAR:            {'✅' if is_bar else '❌'}"
        + (f"  (block_size={at_cfg.get('block_size', '?')})" if is_bar else "")
    )
    print(f"     WBC:            {'✅' if is_wbc else '❌'}")
    if is_wbc:
        print(f"     Part shuffle:   {'🔀 ON' if has_shuffle else '❌ OFF'}")
        print(f"     Indicator:      {'📍 ON' if has_indicator else '❌ OFF'}")
        pm = at_cfg.get("parts_meta")
        if pm:
            print(f"     Parts meta:     {dict(pm) if hasattr(pm, 'items') else pm}")
    if at_cfg.get("vq_config"):
        vq = at_cfg.vq_config
        print(
            f"     Codebook:       {vq.get('codebook_size', '?')}  ×  {vq.get('num_level', '?')} levels"
        )

    if cfg.data.get("embodiment_datasets"):
        emb_names = list(cfg.data.embodiment_datasets.keys())
        print(f"\n  📦 Data")
        print(
            f"     Embodiments:    {', '.join(emb_names[:5])}{'...' if len(emb_names) > 5 else ''}"
        )

    print()


# ============================================================================
# Section 1: Full VLA label
# ============================================================================


def show_vla_label(preprocessor, samples, device, tokenizer):
    """Generate the real label via encode_train() and show region distribution."""
    print("═" * 70)
    print("📊 Section 1: Full VLA Label")
    print("   Actual InputPreprocessor.encode_train() output")
    print("═" * 70)

    input_ids, labels, attention_mask, split_index = preprocessor.encode_train(
        deepcopy(samples),
        device=device,
        training=True,
    )

    ids = input_ids[0].cpu().tolist()
    lbls = labels[0].cpu().tolist()
    attn = attention_mask[0].cpu().float().tolist()

    total_len = len(ids)
    loss_count = sum(1 for l in lbls if l != -100)

    print(f"\n  📏 Total sequence length:  {total_len} tokens")
    print(
        f"  ✂️  Split idx:  {split_index}  (prefix={split_index} | suffix={total_len - split_index})"
    )
    print(f"  📈 Loss tokens:  {loss_count}/{total_len} tokens ({loss_count / total_len:.1%})")

    # Per-region stats + progress bar.
    region_counts = {}
    region_emoji = {}
    region_color = {}
    for v in attn:
        name, emoji, color = _region_info(v)
        region_counts[name] = region_counts.get(name, 0) + 1
        region_emoji[name] = emoji
        region_color[name] = color

    max_count = max(region_counts.values())
    print(f"\n  📊 Region distribution:")
    for region, count in region_counts.items():
        emoji = region_emoji[region]
        color = region_color[region]
        pct = count / total_len * 100
        bar = _bar(count, max_count, width=25)
        print(
            f"    {emoji} {color}{region:10s}{RESET} {count:>5d}t ({pct:5.1f}%)  {color}{bar}{RESET}"
        )

    # Detailed display by region.
    print(f"\n  🗂️  Region details:")
    print(f"  ┌{'─' * 68}┐")

    prev_region = None
    region_start = 0
    for i in range(total_len + 1):
        cur_region = _region_name(attn[i]) if i < total_len else None
        if cur_region != prev_region and prev_region is not None:
            count = i - region_start
            region_ids = ids[region_start:i]
            _, emoji, color = _region_info(attn[region_start])

            # Label status.
            region_labels = lbls[region_start:i]
            if all(l == -100 for l in region_labels):
                loss_icon = "⬜ ignore"
            elif all(l != -100 for l in region_labels):
                loss_icon = "🟩 LOSS  "
            else:
                loss_icon = "🟨 mixed "

            attn_val = attn[region_start]
            attn_desc = "3.xx" if prev_region == "BAR_BLK" else f"{attn_val:.0f}"

            # Header line
            print(
                f"  │ {emoji} {color}{prev_region:8s}{RESET} │ attn={attn_desc:>4s} │ {count:>4d}t │ {loss_icon} │"
            )

            # Text content — full text, wrapped, no truncation
            if prev_region in ("IMAGE", "PAD"):
                print(f"  │   [{count} tokens]")
            else:
                decoded_text = tokenizer.decode(region_ids)
                import textwrap as _tw

                for logical_line in decoded_text.replace("\n", "↵\n").split("\n"):
                    if not logical_line:
                        continue
                    for wline in _tw.wrap(logical_line, width=72) or [logical_line]:
                        print(f"  │   {wline}")

            if cur_region is not None and cur_region != prev_region:
                print(f"  ├{'─' * 68}┤")

            region_start = i
        prev_region = cur_region

    print(f"  └{'─' * 68}┘")

    # Loss legend.
    print(f"\n  💡 Legend: 🟩 LOSS=used for loss  ⬜ ignore=label=-100  🟨 mixed=partially used")
    print()

    return input_ids, labels, attention_mask, split_index


# ============================================================================
# Section 2: Action token details
# ============================================================================


def show_action_details(attention_mask, input_ids, tokenizer, action_tokenizer):
    """Extract the ACTION region, then parse and display it with serialization.py helpers."""
    from g05.tokenizer.interface.serialization import extract_parts_from_str

    print("═" * 70)
    print("🎯 Section 2: Action Token Details")
    print("═" * 70)

    attn = attention_mask[0].cpu().float()
    ids = input_ids[0].cpu()

    # Extract ACTION region (floor(attn)==3).
    action_mask = torch.floor(attn) == 3
    if not action_mask.any():
        print("  ❌ No ACTION region")
        return

    action_indices = action_mask.nonzero(as_tuple=True)[0]
    start, end = action_indices[0].item(), action_indices[-1].item() + 1

    action_ids = ids[start:end].tolist()
    action_attn = attn[start:end].tolist()
    action_text = tokenizer.decode(action_ids)

    # Stats.
    ar_count = sum(1 for v in action_attn if v == 3.0)
    bar_count = sum(1 for v in action_attn if v != 3.0)

    print(f"\n  📍 Position: [{start}, {end})  =  {end - start} tokens")
    print(f"  🎯 Direct-AR:  {ar_count:>4d} tokens  (attn_mask=3)")
    print(f"  🧩 BAR blocks: {bar_count:>4d} tokens  (attn_mask=3.xx)")

    if bar_count > 0:
        block_vals = sorted(set(v for v in action_attn if v != 3.0))
        n_blocks = len(block_vals)
        tokens_per_block = bar_count // max(n_blocks, 1)
        print(f"  📦 {n_blocks} blocks, about {tokens_per_block} tokens per block")

    # Serializer info.
    is_multi = getattr(action_tokenizer, "_is_multi_part", False)
    if is_multi:
        ser = action_tokenizer.serializer
        has_shuffle = getattr(ser, "part_order_shuffle", False)
        has_indicator = getattr(ser, "use_indicator_tokens", False)
        print(f"\n  🔧 Serializer:")
        print(f"     Part order shuffle: {'🔀 ON' if has_shuffle else '❌ OFF'}")
        print(f"     Indicator tokens:   {'📍 ON' if has_indicator else '❌ OFF'}")
        print(f"     Part names:         {ser.part_names}")

    # Raw token text.
    print(f"\n  📜 Raw token text (raw tokenizer.decode output):")
    print(f"  ┌{'─' * 68}┐")
    # Wrap every 80 characters.
    raw = action_text
    line_width = 66
    for i in range(0, len(raw), line_width):
        print(f"  │ {raw[i : i + line_width]:<66s} │")
    print(f"  └{'─' * 68}┘")

    # Structured parsing, reusing serialization.py.
    if is_multi:
        ser = action_tokenizer.serializer
        part_names = ser.part_names

        # Use strip_indicator_prefix to separate indicator.
        body = ser.strip_indicator_prefix(action_text)
        indicator_prefix = (
            action_text[: len(action_text) - len(body)] if body != action_text else ""
        )

        if indicator_prefix:
            # Parse indicator tokens.
            ind_tokens = re.findall(r"<ind_(\w+)>", indicator_prefix)
            print(f"\n  📍 Indicator prefix: {' → '.join(ind_tokens)}")
            print(f"     (predicted part order: {ind_tokens})")

        # Parse with extract_parts_from_str for flat format or _aggregate_actions for BAR format.
        parts_dict, detected_order = extract_parts_from_str(body, part_names)
        if not parts_dict:
            # BAR format (<xxx_blk_0>, <xxx_blk_1>, ...) uses _aggregate_actions.
            from g05.tokenizer.interface.serialization import _aggregate_actions

            parts_dict = _aggregate_actions(body)
            detected_order = [k for k in part_names if k in parts_dict]

        if detected_order:
            print(f"\n  🔀 Detected part order: {' → '.join(detected_order)}")

        PART_EMOJI = {"left_arm": "🦾", "right_arm": "💪", "lower_body": "🦿"}
        code_parts = action_tokenizer.action_tokenizer.code_parts

        print(f"\n  🧩 Per-part codes (parsed by extract_parts_from_str):")
        for part_name in detected_order:
            emoji = PART_EMOJI.get(part_name, "🔧")
            codes_str = parts_dict[part_name].strip()
            # Parse codes from tokenizer text.
            code_ids = tokenizer.encode(codes_str, add_special_tokens=False)
            expected = code_parts.get(part_name, "?")
            print(f"     {emoji} {part_name:15s}: {len(code_ids):>3d} codes (expected {expected})")
            # Show the first few code values.
            action_begin = action_tokenizer.action_token_begin_idx
            raw_codes = [cid - action_begin for cid in code_ids[:8]]
            suffix = f" … +{len(code_ids) - 8}" if len(code_ids) > 8 else ""
            print(f"        codes: [{' '.join(str(c) for c in raw_codes)}{suffix}]")

    print()


# ============================================================================
# Section 3: Action encode/decode roundtrip
# ============================================================================


def show_action_roundtrip(action_tokenizer, action_gt, action_dim_is_pad, device, parts_meta=None):
    """action -> encode -> decode -> compare L1 error."""
    print("═" * 70)
    print("🔄 Section 3: Action Encode/Decode Roundtrip")
    print("═" * 70)

    if action_gt is None:
        print("  ❌ No GT action; skipping roundtrip")
        return

    action_gt = action_gt.to(device)
    B, H, D = action_gt.shape

    # Input stats.
    valid_dims = D
    if action_dim_is_pad is not None:
        valid_dims = int((~action_dim_is_pad[0].bool()).sum().item())

    print(f"\n  📥 Input:")
    print(f"     Shape:       [{B}, {H}, {D}]  (batch, horizon, dim)")
    print(f"     Valid dims:  {valid_dims}/{D}  (padded {D - valid_dims})")
    print(f"     Value range: [{action_gt.min():.3f}, {action_gt.max():.3f}]")

    # Reuse backend encode/decode directly, bypassing the serializer layer.
    backend = action_tokenizer.action_tokenizer

    encode_kw = {}
    if action_dim_is_pad is not None:
        encode_kw["action_dim_is_pad"] = action_dim_is_pad.to(device)

    with torch.no_grad():
        codes_dict = backend.encode(action_gt, **encode_kw)

    if isinstance(codes_dict, dict):
        print(f"\n  📤 VQ encoding (backend.encode):")
        total_codes = 0
        for part_name, codes in codes_dict.items():
            if isinstance(codes, torch.Tensor):
                total_codes += codes.numel()
                print(
                    f"     {part_name:15s}: shape={list(codes.shape)}  range=[{codes.min()}, {codes.max()}]"
                )
        print(f"     Total codes:     {total_codes}")
    else:
        print(f"\n  📤 VQ encoding: shape={list(codes_dict.shape)}  ({codes_dict.numel()} codes)")

    with torch.no_grad():
        decoded = backend.decode(
            codes_dict,
            action_dim_is_pad=encode_kw.get("action_dim_is_pad"),
        )
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    decoded = decoded.to(action_gt.device)  # [B, H, D]

    # Compare.
    abs_diff = torch.abs(action_gt - decoded)

    if action_dim_is_pad is not None:
        valid_mask = ~action_dim_is_pad[0].bool().to(action_gt.device)
        valid_diff = abs_diff[:, :, valid_mask]
    else:
        valid_diff = abs_diff

    l1 = valid_diff.mean().item()
    mse = (valid_diff**2).mean().item()
    max_error = valid_diff.max().item()

    threshold = 0.5
    passed = l1 < threshold

    print(f"\n  📊 Roundtrip error (valid dims):")
    print(f"     ┌─────────────────────────────────────────┐")
    print(f"     │  L1 (mean):    {l1:>10.6f}              │")
    print(f"     │  MSE:          {mse:>10.6f}              │")
    print(f"     │  Max error:    {max_error:>10.6f}              │")
    print(f"     │  Threshold:    L1 < {threshold}                  │")
    print(f"     │  Result:       {'✅ PASS' if passed else '❌ FAIL':>10s}                │")
    print(f"     └─────────────────────────────────────────┘")

    # Per-part breakdown, using parts_meta to map dimensions to body parts.
    dim_l1 = abs_diff.mean(dim=(0, 1))  # [D]

    if parts_meta:
        PART_EMOJI = {
            "left_arm": "🦾",
            "right_arm": "💪",
            "left_gripper": "🤏",
            "right_gripper": "✊",
            "left_ee_pose": "📐",
            "right_ee_pose": "📐",
            "torso": "🧍",
            "torso.velocities": "🔄",
            "chassis": "🚗",
            "chassis.velocities": "🔄",
            "left_hand": "🖐️",
            "right_hand": "🖐️",
            "lower_body": "🦿",
        }
        print(f"\n  🔍 Per-part L1 (grouped by body part):")
        offset = 0
        max_part_l1 = 0
        part_stats = []
        for part_name, part_dim in parts_meta.items():
            if offset + part_dim > D:
                break
            part_diff = abs_diff[:, :, offset : offset + part_dim]
            is_pad = action_dim_is_pad is not None and action_dim_is_pad[0, offset].item()
            part_l1 = part_diff.mean().item()
            part_max = part_diff.max().item()
            part_stats.append((part_name, part_dim, offset, part_l1, part_max, is_pad))
            max_part_l1 = max(max_part_l1, part_l1)
            offset += part_dim

        for part_name, part_dim, start, part_l1, part_max, is_pad in part_stats:
            emoji = PART_EMOJI.get(part_name, "🔧")
            severity = (
                "🟢"
                if part_l1 < 0.01
                else ("🟡" if part_l1 < 0.1 else ("🟠" if part_l1 < 0.5 else "🔴"))
            )
            bar = _bar(part_l1, max(max_part_l1, 1e-6), width=15)
            pad_flag = " ⬜ pad" if is_pad else ""
            print(
                f"     {severity} {emoji} {part_name:20s} (dim {start:>2d}-{start + part_dim - 1:<2d}, {part_dim:>2d}d)"
                f"  L1={part_l1:.6f}  max={part_max:.6f}  {bar}{pad_flag}"
            )
    else:
        # Fallback: per-dim when parts_meta is unavailable.
        n_show = min(10, D)
        top_dims = dim_l1.topk(n_show).indices.tolist()
        print(f"\n  🔍 Per-dim L1 (top-{n_show} worst):")
        for d in top_dims:
            val = dim_l1[d].item()
            pad_flag = (
                " ⬜ padded"
                if (action_dim_is_pad is not None and action_dim_is_pad[0, d].item())
                else ""
            )
            severity = (
                "🟢" if val < 0.01 else ("🟡" if val < 0.1 else ("🟠" if val < 0.5 else "🔴"))
            )
            bar = _bar(val, max_error, width=15)
            print(f"     {severity} dim {d:>3d}: {val:.6f}  {bar}{pad_flag}")

    # Per-timestep L1
    step_l1 = abs_diff.mean(dim=(0, 2))  # [H]
    print(f"\n  ⏱️  Per-timestep L1 (horizon={H}):")
    max_step_l1 = step_l1.max().item()
    for t in range(min(H, 8)):
        val = step_l1[t].item()
        severity = "🟢" if val < 0.1 else ("🟡" if val < 0.3 else "🔴")
        bar = _bar(val, max_step_l1, width=20)
        print(f"     {severity} t={t:>2d}: {val:.4f}  {bar}")
    if H > 8:
        print(f"     ... ({H - 8} more timesteps)")

    print()


# ============================================================================
# Section 4: Attention mask visualization
# ============================================================================


def show_attention_mask(mask_helper, input_ids, attention_mask, save_dir=None):
    """Generate causal mask with MaskHelper and visualize it."""
    from g05.models.g05.helpers.mask_helper import MaskHelper

    print("═" * 70)
    print("👁️  Section 4: Attention Mask")
    print("═" * 70)

    device = input_ids.device
    causal_mask, position_ids = mask_helper.build_vlm_mask(
        input_ids,
        attention_mask,
        kv_len=0,
        dtype=torch.float32,
    )

    S = causal_mask.shape[-1]
    attend = (causal_mask[0, 0] == 0).float()
    attend_ratio = attend.sum().item() / (S * S)

    print(f"\n  📐 Causal mask: [{S} × {S}]  ({S * S:,} entries)")
    print(f"  👁️  Attendable ratio: {attend_ratio:.2%}  ({int(attend.sum().item()):,} / {S * S:,})")
    print(f"  📍 Position IDs: [{position_ids.min().item()}, {position_ids.max().item()}]")

    # Analyze attention patterns by region.
    attn = attention_mask[0].cpu().float()
    regions = []
    prev_region = None
    region_start = 0
    for i in range(S + 1):
        cur_region = _region_name(attn[i].item()) if i < S else None
        if cur_region != prev_region and prev_region is not None:
            regions.append((prev_region, region_start, i))
            region_start = i
        prev_region = cur_region

    if regions:
        print(f"\n  🔗 Inter-region attention pattern (query rows -> key columns):")
        attend_2d = attend.cpu()
        header = "Q \\ K"
        print(f"     {header:12s}", end="")
        for rname, _, _ in regions:
            print(f" {rname[:6]:>6s}", end="")
        print()
        for q_name, q_start, q_end in regions:
            print(f"     {q_name[:12]:12s}", end="")
            for k_name, k_start, k_end in regions:
                region_attend = attend_2d[q_start:q_end, k_start:k_end]
                ratio = region_attend.mean().item()
                # Detect causal: upper triangle all 0 and lower triangle not all 0.
                q_size = q_end - q_start
                k_size = k_end - k_start
                is_causal = False
                if q_size > 1 and k_size > 1 and 0.01 < ratio < 0.99:
                    upper = torch.triu(region_attend, diagonal=1)
                    is_causal = upper.sum().item() == 0 and ratio > 0.01
                if ratio > 0.99:
                    cell = "  ████"
                elif is_causal:
                    cell = "  ◣◣◣◣"
                elif ratio < 0.01:
                    cell = "  ····"
                else:
                    cell = f"  {ratio:.0%}".ljust(6)
                print(cell, end="")
            print()
        print(f"     ████=fully bidirectional  ◣◣◣◣=causal lower triangle  ····=not visible  others=visible ratio")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        token_labels = []
        for i in range(S):
            name = _region_name(attn[i].item())
            token_labels.append(f"{i}:{name[:3]}")

        save_path = str(save_dir / "vlm_causal_mask.png")
        import matplotlib

        matplotlib.use("Agg")
        MaskHelper.visualize(
            causal_mask,
            title="VLM Causal Mask",
            token_labels=token_labels,
            save_path=save_path,
            show=False,
        )
        print(f"\n  💾 Saved: {save_path}")
    else:
        print(f"\n  💡 Use --save-dir to save the attention mask heatmap")

    print()
    print("═" * 70)
    print("🏁 Visualization complete!")
    print("═" * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="🔬 VLA Label lifecycle visualization")
    parser.add_argument(
        "--task",
        required=True,
        help="Hydra task config (e.g. pretrain/bench/ppbench_g05v2 or configs/task/xxx.yaml)",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--save-dir", default=None, help="Directory for saving visualization images")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", nargs="*", default=[], help="Additional Hydra overrides")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 0. Config
    cfg, task_name = load_config(args.task, args.override)
    show_config_summary(cfg, task_name)

    # Data + Components
    samples, action_gt, action_dim_is_pad, parts_meta = load_data(cfg, device, args.sample_idx)
    preprocessor, action_tokenizer, mask_helper = build_components(cfg, device)

    tokenizer = preprocessor.tokenizer

    # 1. Full VLA label.
    input_ids, labels, attention_mask, split_index = show_vla_label(
        preprocessor,
        samples,
        device,
        tokenizer,
    )

    # 2. Action token details.
    show_action_details(attention_mask, input_ids, tokenizer, action_tokenizer)

    # 3. Action Roundtrip
    show_action_roundtrip(action_tokenizer, action_gt, action_dim_is_pad, device, parts_meta)

    # 4. Attention Mask
    show_attention_mask(mask_helper, input_ids, attention_mask, args.save_dir)


if __name__ == "__main__":
    main()

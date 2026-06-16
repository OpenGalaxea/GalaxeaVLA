"""
Consistency test: dataset.__getitem__  vs  serve_policy.build_obs_dict

Verify whether eval_open_loop (offline) and serve_policy (online) feed exactly the
same data into processor.preprocess.

Usage:
    python tests/test_serve_vs_dataset.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        eval_embodiment=galaxea_r1lite

    # Specify sample count and output directory
    python tests/test_serve_vs_dataset.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        --num_samples 5 --output_dir /tmp/serve_vs_dataset \
        eval_embodiment=galaxea_r1lite

Output:
    - terminal: per-field comparison table (keys / shape / dtype / value diff)
    - output_dir/: visualization images
        - sample_{i}_pre_preprocess.png   comparison before preprocess
        - sample_{i}_post_preprocess.png  comparison after preprocess
        - sample_{i}_images.png           image comparison
        - summary.txt                     summary of all differences
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from omegaconf import OmegaConf

from g05.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset
from g05.data_processor.processor.base_processor import BaseProcessor
from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.eval.eval_utils import filter_embodiment, truncate_datasets
from g05.utils.data.normalizer import load_dataset_stats_from_json

from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from scripts.serve_policy import build_obs_dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict: {"a": {"b": tensor}} -> {"a.b": tensor}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _describe(v) -> str:
    """One-line description of a value."""
    if isinstance(v, torch.Tensor):
        return f"Tensor shape={list(v.shape)} dtype={v.dtype}"
    elif isinstance(v, np.ndarray):
        return f"ndarray shape={list(v.shape)} dtype={v.dtype}"
    else:
        return f"{type(v).__name__}: {repr(v)[:80]}"


def _compare_values(a, b, name: str) -> list[str]:
    """Compare two values, return list of difference descriptions."""
    diffs = []
    if type(a) != type(b):
        # tensor vs int, etc
        diffs.append(f"  [{name}] type mismatch: {type(a).__name__} vs {type(b).__name__}")
        return diffs

    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            diffs.append(f"  [{name}] shape mismatch: {list(a.shape)} vs {list(b.shape)}")
            return diffs
        if a.dtype != b.dtype:
            diffs.append(f"  [{name}] dtype mismatch: {a.dtype} vs {b.dtype}")
        if a.is_floating_point() and b.is_floating_point():
            abs_diff = (a.float() - b.float()).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            if max_diff > 0:
                diffs.append(f"  [{name}] value diff: max={max_diff:.6e}, mean={mean_diff:.6e}")
        elif a.dtype == b.dtype:
            if not torch.equal(a, b):
                n_diff = (a != b).sum().item()
                diffs.append(f"  [{name}] {n_diff}/{a.numel()} elements differ")
    elif isinstance(a, (int, float)):
        if a != b:
            diffs.append(f"  [{name}] value: {a} vs {b}")
    elif isinstance(a, str):
        if a != b:
            diffs.append(f"  [{name}] str: '{a[:50]}' vs '{b[:50]}'")
    return diffs


# ──────────────────────────────────────────────────────────────────────
# dataset sample -> simulated raw_obs (what a client would send)
# ──────────────────────────────────────────────────────────────────────


def dataset_sample_to_raw_obs(sample: dict, embodiment_type: str | None = None) -> dict:
    """Convert dataset __getitem__ output to the raw_obs format a client would send.

    Dataset produces:
        images[key]: Tensor [obs_size, C, H, W] uint8
        state[key]:  Tensor [obs_size, D] float32
        task:        str

    Client sends:
        images[key]: ndarray [C, H, W] uint8        (last obs step)
        state[key]:  ndarray [D] float32             (last obs step)
        task:        str
        embodiment_type: str (optional)
    """
    raw_obs = {
        "images": {},
        "state": {},
        "task": sample["task"],
    }
    for k, v in sample["images"].items():
        # Take last obs step, convert to numpy
        raw_obs["images"][k] = v[-1].numpy()  # [C, H, W] uint8

    for k, v in sample["state"].items():
        raw_obs["state"][k] = v[-1].numpy()  # [D] float32

    if embodiment_type:
        raw_obs["embodiment_type"] = embodiment_type

    if "coarse_task" in sample:
        raw_obs["coarse_task"] = sample["coarse_task"]
    if "frequency" in sample:
        raw_obs["frequency"] = float(sample["frequency"])

    return raw_obs


# ──────────────────────────────────────────────────────────────────────
# visualization
# ──────────────────────────────────────────────────────────────────────


def plot_pre_preprocess_comparison(ds_sample: dict, server_dict: dict, output_path: Path):
    """Plot side-by-side comparison of state values before preprocess."""
    ds_flat = _flatten(ds_sample)
    sv_flat = _flatten(server_dict)

    # Collect state keys that exist in both
    state_keys = sorted(
        k
        for k in ds_flat
        if k.startswith("state.")
        and k in sv_flat
        and isinstance(ds_flat[k], torch.Tensor)
        and isinstance(sv_flat[k], torch.Tensor)
    )
    if not state_keys:
        return

    n = len(state_keys)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle(
        "Pre-preprocess: state values comparison\n(dataset vs server build_obs_dict)", fontsize=14
    )

    for i, key in enumerate(state_keys):
        ds_val = ds_flat[key].float().numpy()
        sv_val = sv_flat[key].float().numpy()

        # Dataset value (obs_size, D)
        ax = axes[i, 0]
        ax.set_title(f"Dataset: {key}")
        if ds_val.ndim == 2:
            for t in range(ds_val.shape[0]):
                ax.bar(
                    np.arange(ds_val.shape[1]) + t * 0.3,
                    ds_val[t],
                    width=0.3,
                    alpha=0.7,
                    label=f"t={t}",
                )
        else:
            ax.bar(range(len(ds_val)), ds_val)
        ax.grid(True, alpha=0.3)

        # Server value (num_obs_steps, D)
        ax = axes[i, 1]
        ax.set_title(f"Server: {key}")
        if sv_val.ndim == 2:
            for t in range(sv_val.shape[0]):
                ax.bar(
                    np.arange(sv_val.shape[1]) + t * 0.3,
                    sv_val[t],
                    width=0.3,
                    alpha=0.7,
                    label=f"t={t}",
                )
        else:
            ax.bar(range(len(sv_val)), sv_val)
        ax.grid(True, alpha=0.3)

        # Diff (compare last obs step of both)
        ax = axes[i, 2]
        ax.set_title(f"Diff (last step): {key}")
        ds_last = ds_val[-1] if ds_val.ndim == 2 else ds_val
        sv_last = sv_val[-1] if sv_val.ndim == 2 else sv_val
        min_len = min(len(ds_last), len(sv_last))
        diff = ds_last[:min_len] - sv_last[:min_len]
        colors = ["green" if abs(d) < 1e-6 else "red" for d in diff]
        ax.bar(range(min_len), diff, color=colors)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("difference")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_post_preprocess_comparison(
    ds_preprocessed: dict, sv_preprocessed: dict, output_path: Path
):
    """Plot side-by-side comparison after preprocess."""
    # Compare: proprio, action, action_dim_is_pad, proprio_dim_is_pad
    tensor_keys = sorted(
        k
        for k in ds_preprocessed
        if k in sv_preprocessed
        and isinstance(ds_preprocessed[k], torch.Tensor)
        and isinstance(sv_preprocessed[k], torch.Tensor)
    )
    # Skip pixel_values (too large), focus on proprio/action/masks
    focus_keys = [
        k
        for k in tensor_keys
        if k
        in (
            "proprio",
            "action",
            "action_is_pad",
            "action_dim_is_pad",
            "proprio_is_pad",
            "proprio_dim_is_pad",
        )
    ]
    if not focus_keys:
        return

    n = len(focus_keys)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    fig.suptitle(
        "Post-preprocess comparison\n(dataset preprocess vs server preprocess)", fontsize=14
    )

    for i, key in enumerate(focus_keys):
        ds_val = ds_preprocessed[key].float().flatten().numpy()
        sv_val = sv_preprocessed[key].float().flatten().numpy()

        ax = axes[i, 0]
        ax.set_title(f"Dataset: {key} {list(ds_preprocessed[key].shape)}")
        ax.plot(ds_val, linewidth=0.8)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.set_title(f"Server: {key} {list(sv_preprocessed[key].shape)}")
        ax.plot(sv_val, linewidth=0.8, color="orange")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        ax.set_title(f"Diff: {key}")
        min_len = min(len(ds_val), len(sv_val))
        diff = ds_val[:min_len] - sv_val[:min_len]
        ax.plot(diff, linewidth=0.8, color="red")
        ax.axhline(y=0, color="black", linewidth=0.5)
        max_abs = np.abs(diff).max()
        ax.set_title(f"Diff: {key} (max={max_abs:.6e})")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_image_comparison(ds_sample: dict, server_dict: dict, output_path: Path):
    """Plot image comparison between dataset and server."""
    ds_images = ds_sample.get("images", {})
    sv_images = server_dict.get("images", {})
    common_keys = sorted(set(ds_images.keys()) & set(sv_images.keys()))
    if not common_keys:
        return

    n = len(common_keys)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle("Image comparison (last obs step)\nDataset vs Server", fontsize=14)

    for i, key in enumerate(common_keys):
        ds_img = ds_images[key]
        sv_img = sv_images[key]

        # Dataset: [obs_size, C, H, W] uint8 tensor -> take last step
        if isinstance(ds_img, torch.Tensor):
            ds_np = ds_img[-1].permute(1, 2, 0).numpy()  # [H, W, C]
        else:
            ds_np = ds_img

        # Server: [num_obs_steps, C, H, W] tensor -> take last step
        if isinstance(sv_img, torch.Tensor):
            sv_np = sv_img[-1].permute(1, 2, 0).numpy()
        else:
            sv_np = sv_img

        ax = axes[i, 0]
        ax.imshow(ds_np)
        ax.set_title(
            f"Dataset: {key}\nshape={list(ds_images[key].shape)}, dtype={ds_images[key].dtype}"
        )
        ax.axis("off")

        ax = axes[i, 1]
        ax.imshow(sv_np)
        ax.set_title(
            f"Server: {key}\nshape={list(sv_images[key].shape)}, dtype={sv_images[key].dtype}"
        )
        ax.axis("off")

        ax = axes[i, 2]
        diff = np.abs(ds_np.astype(float) - sv_np.astype(float))
        ax.imshow(diff.astype(np.uint8), cmap="hot")
        ax.set_title(f"Abs diff: {key}\nmax={diff.max():.1f}, mean={diff.mean():.4f}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_pixel_values_comparison(ds_pv: torch.Tensor, sv_pv: torch.Tensor, output_path: Path):
    """Plot pixel_values (after preprocess transforms) comparison."""
    n = min(ds_pv.shape[0], sv_pv.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle("pixel_values after preprocess (normalized)\nDataset vs Server", fontsize=14)

    for i in range(n):
        # Denormalize for visualization: rough inverse of ImageNet normalization
        def denorm(t):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = t * std + mean
            return t.clamp(0, 1).permute(1, 2, 0).numpy()

        ds_np = denorm(ds_pv[i])
        sv_np = denorm(sv_pv[i])

        axes[i, 0].imshow(ds_np)
        axes[i, 0].set_title(f"Dataset cam[{i}]")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sv_np)
        axes[i, 1].set_title(f"Server cam[{i}]")
        axes[i, 1].axis("off")

        diff = np.abs(ds_np - sv_np)
        axes[i, 2].imshow(diff / (diff.max() + 1e-8))
        axes[i, 2].set_title(f"Diff (max={diff.max():.6f})")
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# core comparison
# ──────────────────────────────────────────────────────────────────────


def compare_dicts(ds_dict: dict, sv_dict: dict, label: str) -> list[str]:
    """Compare two dicts field-by-field, return list of difference lines."""
    lines = [f"\n{'=' * 70}", f"  {label}", f"{'=' * 70}"]

    ds_flat = _flatten(ds_dict)
    sv_flat = _flatten(sv_dict)
    all_keys = sorted(set(ds_flat.keys()) | set(sv_flat.keys()))

    ds_only = sorted(set(ds_flat.keys()) - set(sv_flat.keys()))
    sv_only = sorted(set(sv_flat.keys()) - set(ds_flat.keys()))
    common = sorted(set(ds_flat.keys()) & set(sv_flat.keys()))

    if ds_only:
        lines.append(f"\n  Keys ONLY in dataset ({len(ds_only)}):")
        for k in ds_only:
            lines.append(f"    {k}: {_describe(ds_flat[k])}")

    if sv_only:
        lines.append(f"\n  Keys ONLY in server ({len(sv_only)}):")
        for k in sv_only:
            lines.append(f"    {k}: {_describe(sv_flat[k])}")

    lines.append(f"\n  Common keys ({len(common)}):")
    n_match = 0
    n_diff = 0
    for k in common:
        diffs = _compare_values(ds_flat[k], sv_flat[k], k)
        if diffs:
            n_diff += 1
            for d in diffs:
                lines.append(d)
        else:
            n_match += 1
            lines.append(f"  [{k}] MATCH  {_describe(ds_flat[k])}")

    lines.append(
        f"\n  Summary: {n_match} match, {n_diff} differ, "
        f"{len(ds_only)} dataset-only, {len(sv_only)} server-only"
    )
    return lines


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare dataset.__getitem__ vs serve_policy.build_obs_dict"
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to compare")
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=1,
        help="Max dataset_dirs to load per group (default: 1, for fast testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for plots (default: <run_dir>/test_serve_vs_dataset)",
    )
    args, remaining = parser.parse_known_args()
    overrides = [r for r in remaining if "=" in r]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ---- load config ----
    run_dir = find_run_dir(args.ckpt_path)
    print(f"Found run dir: {run_dir}")
    cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    eval_embodiment = cfg.get("eval_embodiment", None)
    is_mixture = "embodiment_datasets" in cfg.data
    if is_mixture:
        if not eval_embodiment:
            # Auto-select first embodiment to avoid loading all datasets
            first_emb = next(iter(cfg.data.embodiment_datasets))
            eval_embodiment = first_emb
            OmegaConf.set_struct(cfg, False)
            cfg.eval_embodiment = eval_embodiment
            print(f"Auto-selected embodiment: {eval_embodiment}")
        filter_embodiment(cfg, eval_embodiment)

    # Truncate dataset_dirs for fast loading (default: 1 dir)
    truncate_datasets(cfg, args.max_datasets)

    # ---- load processor (no model needed) ----
    dataset_stats = load_dataset_stats_from_json(run_dir / "dataset_stats.json")

    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()
    action_horizon = int(cfg.data.action_size)
    if isinstance(processor, MixtureProcessor):
        for sub_processor in processor.processors.values():
            sub_processor.action_horizon = action_horizon
    else:
        processor.action_horizon = action_horizon

    # ---- load dataset (without auto-preprocess) ----
    dataset_eval = instantiate_dataset(cfg, is_training_set=False)
    if isinstance(dataset_eval, MixtureLerobotDataset):
        emb = dataset_eval.embodiments[0]
        ds = dataset_eval.datasets[0]
        ds.processor = None  # Bypass set_processor to avoid None.eval()
        emb_processor = processor[emb] if isinstance(processor, MixtureProcessor) else processor
        print(f"Using embodiment: {emb}")
    else:
        dataset_eval.processor = None  # Bypass set_processor to avoid None.eval()
        ds = dataset_eval
        emb_processor = processor
        emb = eval_embodiment

    # ---- output dir ----
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(run_dir / "test_serve_vs_dataset")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    all_report_lines = []
    all_report_lines.append(f"Checkpoint: {args.ckpt_path}")
    all_report_lines.append(f"Embodiment: {emb}")
    all_report_lines.append(f"Num samples: {args.num_samples}")
    all_report_lines.append(f"Dataset size: {len(ds)}")

    # ---- compare samples ----
    num_samples = min(args.num_samples, len(ds))
    # Pick evenly spaced indices
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    for sample_i, idx in enumerate(indices):
        print(f"\n{'#' * 70}")
        print(f"  Sample {sample_i} (dataset idx={idx})")
        print(f"{'#' * 70}")

        # ---- Path A: dataset __getitem__ (what eval_open_loop uses) ----
        ds_sample = ds[int(idx)]  # raw sample, no preprocess

        # ---- Path B: simulate client -> build_obs_dict ----
        raw_obs = dataset_sample_to_raw_obs(ds_sample, embodiment_type=emb)
        server_dict = build_obs_dict(raw_obs, processor)

        # ====== Compare BEFORE preprocess ======
        report = compare_dicts(ds_sample, server_dict, f"Sample {sample_i}: BEFORE preprocess")
        for line in report:
            print(line)
        all_report_lines.extend(report)

        # Visualize images
        try:
            plot_image_comparison(
                ds_sample, server_dict, output_dir / f"sample_{sample_i}_images.png"
            )
            print(f"  -> Saved {output_dir / f'sample_{sample_i}_images.png'}")
        except Exception as e:
            print(f"  -> Image plot failed: {e}")

        # Visualize state values
        try:
            plot_pre_preprocess_comparison(
                ds_sample, server_dict, output_dir / f"sample_{sample_i}_pre_preprocess.png"
            )
            print(f"  -> Saved {output_dir / f'sample_{sample_i}_pre_preprocess.png'}")
        except Exception as e:
            print(f"  -> Pre-preprocess plot failed: {e}")

        # ====== Run through preprocess ======
        ds_for_preprocess = deepcopy(ds_sample)
        sv_for_preprocess = deepcopy(server_dict)

        # Use the sub-processor for the specific embodiment
        try:
            ds_preprocessed = emb_processor.preprocess(ds_for_preprocess)
            sv_preprocessed = emb_processor.preprocess(sv_for_preprocess)
        except Exception as e:
            msg = f"  preprocess failed: {e}"
            print(msg)
            all_report_lines.append(msg)
            continue

        # ====== Compare AFTER preprocess ======
        report = compare_dicts(
            ds_preprocessed, sv_preprocessed, f"Sample {sample_i}: AFTER preprocess"
        )
        for line in report:
            print(line)
        all_report_lines.extend(report)

        # Visualize post-preprocess tensors
        try:
            plot_post_preprocess_comparison(
                ds_preprocessed,
                sv_preprocessed,
                output_dir / f"sample_{sample_i}_post_preprocess.png",
            )
            print(f"  -> Saved {output_dir / f'sample_{sample_i}_post_preprocess.png'}")
        except Exception as e:
            print(f"  -> Post-preprocess plot failed: {e}")

        # Visualize pixel_values comparison
        if "pixel_values" in ds_preprocessed and "pixel_values" in sv_preprocessed:
            try:
                plot_pixel_values_comparison(
                    ds_preprocessed["pixel_values"],
                    sv_preprocessed["pixel_values"],
                    output_dir / f"sample_{sample_i}_pixel_values.png",
                )
                print(f"  -> Saved {output_dir / f'sample_{sample_i}_pixel_values.png'}")
            except Exception as e:
                print(f"  -> Pixel values plot failed: {e}")

    # ---- Save summary ----
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(all_report_lines))
    print(f"\nSummary saved to: {summary_path}")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

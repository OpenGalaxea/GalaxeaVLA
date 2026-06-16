#!/usr/bin/env python3
"""Checkpoint export tool: copy training artifacts into a unified backup directory.

Includes git provenance information.

Usage:
    python tools/export_checkpoint.py [source_dir] [--target-base DIR] [--name NAME]
                                      [--checkpoint CKPT] [--strip] [-y]

    --strip   remove training state such as optimizer / scheduler / ema, keeping only model weights
    --output  directly specify output file path; implies --strip and skips directory structure
              e.g. --output /path/to/model_22000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401 — enable arrow-key/editing support for input()
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_EXP_DIR = os.environ.get("DEFAULT_EXP_DIR", "outputs/checkpoints")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str):
    print(msg, flush=True)


def format_size(n: int) -> str:
    """Bytes → human-readable string (KB / MB / GB)."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    else:
        return f"{n / 1024**3:.1f} GB"


def copy_with_progress(src: Path, dst: Path, desc: str | None = None):
    """Copy a file with progress display for large files (>100 MB)."""
    size = src.stat().st_size
    if size < 100 * 1024 * 1024:
        shutil.copy2(src, dst)
        return

    label = desc or src.name
    copied = 0
    chunk = 64 * 1024 * 1024  # 64 MB
    start = time.time()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            copied += len(buf)
            pct = copied / size * 100
            elapsed = time.time() - start
            speed = copied / elapsed if elapsed > 0 else 0
            speed_str = format_size(int(speed)) + "/s"
            print(
                f"\r  Copying... {label}: {format_size(copied)}/{format_size(size)}"
                f" ({pct:.1f}%) [{speed_str}]",
                end="",
                flush=True,
            )
    # preserve metadata
    shutil.copystat(src, dst)
    print(flush=True)


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------


def validate_source_dir(path: Path) -> dict:
    """Validate source directory and return info dict.

    Returns:
        {
            "root": Path,
            "hydra_dir": Path,
            "checkpoints": [Path, ...],   # sorted by step number
            "last_pt": Path | None,        # symlink target if exists
            "action_tokenizer": Path | None,
            "dataset_stats": Path | None,
        }
    """
    if not path.is_dir():
        raise SystemExit(f"Error: source directory does not exist: {path}")

    hydra_dir = path / ".hydra"
    if not hydra_dir.is_dir():
        raise SystemExit(f"Error: missing .hydra/ directory: {path}")

    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Error: missing checkpoints/ directory: {path}")

    checkpoints = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda p: _extract_step(p.name),
    )
    if not checkpoints:
        raise SystemExit("Error: no .pt files found in checkpoints/")

    last_pt = path / "last.pt"
    last_target = None
    if last_pt.is_symlink():
        last_target = last_pt.resolve()

    action_tok = path / "action_tokenizer.pt"
    dataset_stats = path / "dataset_stats.json"

    return {
        "root": path.resolve(),
        "hydra_dir": hydra_dir,
        "checkpoints": checkpoints,
        "last_target": last_target,
        "action_tokenizer": action_tok if action_tok.exists() else None,
        "dataset_stats": dataset_stats if dataset_stats.exists() else None,
    }


def _extract_step(name: str) -> int:
    """Extract step number from filename like 'step_2000.pt'."""
    stem = Path(name).stem
    parts = stem.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return 0


# ---------------------------------------------------------------------------
# Git info from wandb metadata
# ---------------------------------------------------------------------------


def find_wandb_metadata(path: Path) -> Path | None:
    """Find the latest wandb-metadata.json under wandb/."""
    candidates = list(path.glob("wandb/wandb/run-*/files/wandb-metadata.json"))
    if not candidates:
        return None
    # pick the latest by run directory name (run-YYYYMMDD_HHMMSS-...)
    return sorted(candidates)[-1]


def extract_git_info(metadata_path: Path | None) -> dict:
    """Extract git commit / remote / branch from wandb metadata.

    Falls back to 'unknown' for missing fields.
    """
    info = {"git_commit": "unknown", "git_branch": "unknown", "git_remote": "unknown"}

    if metadata_path is None or not metadata_path.exists():
        return info

    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return info

    git_block = meta.get("git", {})
    commit = git_block.get("commit", "unknown")
    remote = git_block.get("remote", "unknown")
    info["git_commit"] = commit
    info["git_remote"] = remote

    # Try to find branch containing this commit
    if commit != "unknown":
        try:
            result = subprocess.run(
                ["git", "branch", "-a", "--contains", commit],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    if "HEAD" in line:
                        continue
                    if line:
                        info["git_branch"] = line
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return info


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------


def prompt_source_dir() -> Path:
    """Interactively ask for source directory path."""
    log("\n--- Step 1: Source directory ---")
    while True:
        raw = input("Enter source directory path: ").strip()
        if raw:
            p = Path(raw)
            if p.is_dir():
                return p
            log(f"  Directory does not exist: {raw}")
        else:
            log("  Path cannot be empty")


def prompt_target(default_base: str, source_path: Path | None = None) -> tuple[str, str]:
    """Interactively ask for target base + folder name.

    Returns (base_dir, folder_name).
    """
    base = default_base

    # Default name: join the last two source path components with '-', e.g.
    # r1lite_g05_pp_pretrain-pp_KI_pretrain_4k.
    default_name = ""
    if source_path is not None:
        parts = source_path.resolve().parts
        if len(parts) >= 2:
            default_name = f"{parts[-2]}-{parts[-1]}"

    log(f"\n--- Step 2: Target configuration ---")
    log(f"Default target directory: {base}")
    log("Enter !base to change the default directory, or enter a folder name directly")

    prompt_msg = f"Folder name [{default_name}]: " if default_name else "Folder name (required): "
    while True:
        raw = input(prompt_msg).strip()
        if raw == "!base":
            new_base = input(f"New target directory [{base}]: ").strip()
            if new_base:
                base = new_base
                log(f"Target directory updated to: {base}")
            continue
        if raw:
            return base, raw
        if default_name:
            return base, default_name
        log("  Name cannot be empty")


def prompt_checkpoint_selection(info: dict) -> Path:
    """List available checkpoints and let user pick one.

    Returns the selected checkpoint Path.
    """
    log("\n--- Step 3: Select checkpoint ---")
    log("Available checkpoints:")

    checkpoints = info["checkpoints"]
    last_target = info["last_target"]

    for i, ckpt in enumerate(checkpoints, 1):
        size = format_size(ckpt.stat().st_size)
        log(f"  [{i}] {ckpt.name}\t({size})")

    # Add last.pt option if it's a valid symlink
    last_idx = None
    if last_target and last_target.exists():
        last_idx = len(checkpoints) + 1
        size = format_size(last_target.stat().st_size)
        log(f"  [{last_idx}] last.pt → {last_target.name}\t(symlink)")

    default = last_idx if last_idx else len(checkpoints)
    while True:
        raw = input(f"Select [default={default}]: ").strip()
        if not raw:
            idx = default
        else:
            try:
                idx = int(raw)
            except ValueError:
                log("  Please enter a number")
                continue

        if 1 <= idx <= len(checkpoints):
            return checkpoints[idx - 1]
        if last_idx and idx == last_idx:
            return last_target
        log(f"  Invalid selection, please enter 1-{last_idx or len(checkpoints)}")


def prompt_strip() -> bool:
    """Ask whether to strip optimizer/scheduler/ema states."""
    log("\n--- Step 4: Strip checkpoint ---")
    log("Remove optimizer/scheduler/ema training state and keep only model weights")
    while True:
        raw = input("Strip checkpoint? [y/N]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no", ""):
            return False


def prompt_confirm(manifest: list[tuple[str, int]], git_info: dict, target: Path) -> bool:
    """Print export manifest and ask for confirmation.

    manifest: list of (display_name, size_bytes)
    """
    log("\n========== Export Manifest ==========")
    log(f"{'File':<40s} {'Size':>12s}")

    total = 0
    for name, size in manifest:
        log(f"{name:<40s} {format_size(size):>12s}")
        total += size

    log(f"{'':─<40s} {'':─>12s}")
    log(f"{'Total':<40s} {format_size(total):>12s}")

    log(f"\nGit commit: {git_info['git_commit']}")
    log(f"Git branch: {git_info['git_branch']}")
    log(f"Git remote: {git_info['git_remote']}")
    log(f"\nTarget path: {target}")

    while True:
        raw = input("\nConfirm export? [Y/n]: ").strip().lower()
        if raw in ("y", "yes", ""):
            return True
        if raw in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# Strip training state
# ---------------------------------------------------------------------------

# Keep only these keys and drop everything else.
_MODEL_KEYS = {"model_state_dict"}


def strip_checkpoint(src: Path, dst: Path) -> tuple[int, int]:
    """Load checkpoint, keep only model weights, save to dst.

    Returns (original_size, stripped_size).
    """
    import torch

    original_size = src.stat().st_size
    log(f"  Loading checkpoint: {src.name} ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise SystemExit(
            f"Error: invalid checkpoint format, expected dict, got {type(ckpt).__name__}"
        )

    kept = {k: v for k, v in ckpt.items() if k in _MODEL_KEYS}
    removed_keys = sorted(set(ckpt.keys()) - _MODEL_KEYS)
    if removed_keys:
        log(f"  Removing keys: {', '.join(removed_keys)}")
    else:
        log("  No keys need to be removed")

    if "model_state_dict" not in kept:
        raise SystemExit("Error: checkpoint does not contain model_state_dict")

    log(f"  Saving stripped checkpoint ...")
    torch.save(kept, dst)
    stripped_size = dst.stat().st_size
    log(
        f"  Reduced size: {format_size(original_size)} -> {format_size(stripped_size)}"
        f" (saved {format_size(original_size - stripped_size)})"
    )

    del ckpt, kept
    return original_size, stripped_size


# ---------------------------------------------------------------------------
# Stage & finalize
# ---------------------------------------------------------------------------


def stage_files(
    info: dict,
    selected_ckpt: Path,
    staging_dir: Path,
    git_info: dict,
    strip: bool = False,
) -> list[tuple[str, int]]:
    """Copy small files to staging dir and build manifest.

    Returns manifest: list of (display_name, size_bytes).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    # .hydra/
    hydra_dst = staging_dir / ".hydra"
    shutil.copytree(info["hydra_dir"], hydra_dst)
    for f in sorted(hydra_dst.rglob("*")):
        if f.is_file():
            rel = f".hydra/{f.relative_to(hydra_dst)}"
            manifest.append((rel, f.stat().st_size))

    # action_tokenizer.pt
    if info["action_tokenizer"]:
        src = info["action_tokenizer"]
        shutil.copy2(src, staging_dir / "action_tokenizer.pt")
        manifest.append(("action_tokenizer.pt", src.stat().st_size))

    # dataset_stats.json
    if info["dataset_stats"]:
        src = info["dataset_stats"]
        shutil.copy2(src, staging_dir / "dataset_stats.json")
        manifest.append(("dataset_stats.json", src.stat().st_size))

    # checkpoint: place under checkpoints/ to keep downstream .parent.parent path convention.
    ckpt_staging = staging_dir / "checkpoints"
    ckpt_staging.mkdir(exist_ok=True)
    if strip:
        stripped_path = ckpt_staging / "model_state_dict.pt"
        _, stripped_size = strip_checkpoint(selected_ckpt, stripped_path)
        ckpt_display = f"checkpoints/model_state_dict.pt (← {selected_ckpt.name}, stripped)"
        manifest.append((ckpt_display, stripped_size))
    else:
        ckpt_display = f"checkpoints/model_state_dict.pt (← {selected_ckpt.name})"
        manifest.append((ckpt_display, selected_ckpt.stat().st_size))

    # export_meta.json
    now = subprocess.run(
        ["date", "+%Y-%m-%d %H:%M"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {
        "source": str(info["root"]),
        "exported_at": now,
        "original_checkpoint": selected_ckpt.name,
        "stripped": strip,
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "git_remote": git_info["git_remote"],
    }
    meta_path = staging_dir / "export_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    manifest.append(("export_meta.json", meta_path.stat().st_size))

    return manifest


def finalize(staging_dir: Path, selected_ckpt: Path, target_dir: Path, strip: bool = False):
    """Move staged files + copy large checkpoint to target directory."""
    if target_dir.exists():
        log(f"\nTarget directory already exists: {target_dir}")
        raw = input("Overwrite? [y/N]: ").strip().lower()
        if raw not in ("y", "yes"):
            raise SystemExit("Cancelled")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Move small files from staging (skip checkpoints/, handled below with progress)
    for item in staging_dir.iterdir():
        if item.name == "checkpoints":
            continue
        dst = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.move(str(item), str(dst))

    # Copy checkpoint with progress; strip and non-strip paths both go through here.
    log(f"\n--- Step 5: Copy to target ---")
    ckpt_dst = target_dir / "checkpoints"
    ckpt_dst.mkdir(exist_ok=True)
    if strip:
        ckpt_src = staging_dir / "checkpoints" / "model_state_dict.pt"
        copy_with_progress(
            ckpt_src, ckpt_dst / "model_state_dict.pt", desc="model_state_dict.pt (stripped)"
        )
    else:
        copy_with_progress(selected_ckpt, ckpt_dst / "model_state_dict.pt", desc=selected_ckpt.name)

    log(f"Export completed: {target_dir}")

    # If invoked by a wrapper (e.g. tools/list_runs.py) that wants to collect
    # the resulting target paths, append to the file pointed to by this env var.
    result_file = os.environ.get("EXPORT_RESULT_FILE")
    if result_file:
        try:
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(str(target_dir) + "\n")
        except OSError as e:
            log(f"Warning: failed to write EXPORT_RESULT_FILE: {e}")


def cleanup(staging_dir: Path):
    """Remove staging directory if it exists."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint to the unified backup directory",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Source directory (optional; prompted interactively when omitted)",
    )
    parser.add_argument(
        "--target-base",
        default=DEFAULT_EXP_DIR,
        help=f"Target root directory (default: {DEFAULT_EXP_DIR})",
    )
    parser.add_argument(
        "--name",
        help="Target folder name (optional; prompted interactively when omitted)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint file name to use directly, for example step_16000.pt",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Remove optimizer/scheduler/ema training state and keep only model weights",
    )
    parser.add_argument(
        "--output",
        help="Write directly to an output .pt path (implies --strip and skips prompts)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = parser.parse_args()

    # Fast path: --output strips directly to the specified file.
    if args.output:
        src_dir = Path(args.source) if args.source else None
        if src_dir is None:
            raise SystemExit("Error: --output mode requires a source directory")
        info = validate_source_dir(src_dir)
        if args.checkpoint:
            selected = info["root"] / "checkpoints" / args.checkpoint
            if not selected.exists():
                raise SystemExit(f"Error: checkpoint does not exist: {selected}")
        else:
            selected = info["checkpoints"][-1]
        dst = Path(args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        strip_checkpoint(selected, dst)
        return

    # Step 1: Source directory
    if args.source:
        source = Path(args.source)
    else:
        source = prompt_source_dir()

    info = validate_source_dir(source)
    log(f"Source directory: {info['root']}")
    log(f"  checkpoints: {len(info['checkpoints'])}")

    # Step 2: Target
    if args.name:
        target_base, target_name = args.target_base, args.name
    else:
        target_base, target_name = prompt_target(args.target_base, source_path=info["root"])

    target_dir = Path(target_base) / target_name

    # Step 3: Checkpoint selection
    if args.checkpoint:
        ckpt_path = info["root"] / "checkpoints" / args.checkpoint
        if not ckpt_path.exists():
            raise SystemExit(f"Error: checkpoint does not exist: {ckpt_path}")
        selected = ckpt_path
    else:
        selected = prompt_checkpoint_selection(info)

    log(f"\nSelected: {selected.name} ({format_size(selected.stat().st_size)})")

    # Step 4: Strip decision
    if args.strip:
        do_strip = True
    elif args.yes:
        do_strip = False  # non-interactive mode defaults to no strip; requires explicit --strip
    else:
        do_strip = prompt_strip()

    # Step 5: Stage small files + confirm
    staging_dir = Path(f"./tmp/export_{target_name}_{os.getpid()}")
    try:
        # Git info
        wandb_meta = find_wandb_metadata(info["root"])
        if wandb_meta is None:
            log("Warning: wandb-metadata.json not found; git info will be marked as unknown")
        git_info = extract_git_info(wandb_meta)

        manifest = stage_files(info, selected, staging_dir, git_info, strip=do_strip)

        if args.yes:
            confirmed = True
        else:
            confirmed = prompt_confirm(manifest, git_info, target_dir)

        if not confirmed:
            log("Cancelled")
            return

        # Step 6: Finalize
        finalize(staging_dir, selected, target_dir, strip=do_strip)

    except KeyboardInterrupt:
        log("\nInterrupted")
    finally:
        cleanup(staging_dir)


if __name__ == "__main__":
    main()

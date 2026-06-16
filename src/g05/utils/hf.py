"""Helpers for resolving Hugging Face model identifiers to local files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

QWEN35_2B_BASE_REPO_ID = "Qwen/Qwen3.5-2B-Base"

_MODEL_REPO_ALIASES = {
    "qwen/qwen3.5-2b-base": QWEN35_2B_BASE_REPO_ID,
    "qwen/qwen3.5-2b": "Qwen/Qwen3.5-2B",
}

_LOCAL_PREFIXES = {
    "checkpoint",
    "checkpoints",
    "ckpt",
    "ckpts",
    "exp",
    "experiments",
    "output",
    "outputs",
    "run",
    "runs",
}


def normalize_hf_model_id(model_path: str) -> str:
    """Return the canonical HF repo id for known aliases."""
    return _MODEL_REPO_ALIASES.get(model_path.lower(), model_path)


def resolve_hf_model_path(
    model_path: str | os.PathLike[str],
    *,
    allow_patterns: Sequence[str] | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a local directory or HF model repo id to a local directory path.

    Existing local directories are returned unchanged. HF model repo ids are
    downloaded through ``snapshot_download`` and resolved to the cache snapshot
    directory. Relative paths with common local prefixes such as
    ``checkpoints/...`` are treated as local paths.
    """
    model_path_str = os.fspath(model_path)
    expanded_path = Path(os.path.expandvars(os.path.expanduser(model_path_str)))
    if expanded_path.is_dir():
        return str(expanded_path)

    if _looks_like_local_path(model_path_str):
        return str(expanded_path)

    repo_id = normalize_hf_model_id(model_path_str)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "huggingface-hub is required to download pretrained model files. "
            f"Install project dependencies or provide a local directory for {repo_id!r}."
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        local_files_only=local_files_only,
        revision=revision,
        token=token,
        cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
    )


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    first = value.split("/", 1)[0]
    return first in _LOCAL_PREFIXES

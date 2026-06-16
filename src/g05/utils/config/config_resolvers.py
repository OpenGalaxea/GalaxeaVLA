import math
from datetime import datetime
from pathlib import Path
from typing import Any, List, Callable, Optional
from omegaconf import DictConfig, OmegaConf


def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)


def _oc_merge(base: Any, override: Any) -> Any:
    """
    Deep-merge two configs. Used in debug YAML files to override specific
    fields (e.g. dataset_groups) while inheriting the rest from the source.

    Usage in YAML:
        ${oc.merge:${oc.load:source.yaml,key},{field: value}}
    """
    if not isinstance(base, DictConfig):
        base = OmegaConf.create(base)
    if not isinstance(override, DictConfig):
        override = OmegaConf.create(override)
    merged = OmegaConf.merge(base, override)
    return merged


def _oc_load(path: str, key: Optional[str] = None) -> Any:
    """
    Load a YAML/JSON config and optionally select a key.
    Uses Hydra's to_absolute_path to honor original working dir.
    """
    try:
        from hydra.utils import to_absolute_path
    except ImportError:
        to_absolute_path = None  # should not happen in normal Hydra runs
    load_path = Path(path)
    if not load_path.is_absolute() and to_absolute_path is not None:
        load_path = Path(to_absolute_path(path))
    cfg = OmegaConf.load(load_path)
    if key is None or key == "":
        return cfg
    return OmegaConf.select(cfg, key)


def sum_shapes(shape_meta_list):
    if not shape_meta_list:
        return 0
    total = sum(int(item["shape"]) for item in shape_meta_list if item["key"] is not None)
    return total


def max_action_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            action_cfg = dataset_cfg.shape_meta.action
            current_dim = sum_shapes(action_cfg)

            if current_dim > max_dim:
                max_dim = current_dim

    return max_dim


def max_state_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            state_cfg = dataset_cfg.shape_meta.state
            current_dim = sum_shapes(state_cfg)
            if current_dim > max_dim:
                max_dim = current_dim

    return max_dim


def _get_obs_image_steps(obs_size) -> int:
    """Extract image steps from obs_size (int scalar).

    Used to compute cond_steps and num_input_images in config interpolations.
    """
    return int(obs_size)


_hydra_runtime_values = {}


def _set_hydra_runtime(key: str, value: str) -> None:
    """Set a Hydra runtime value for resolution."""
    _hydra_runtime_values[key] = value


def _hydra_resolver(key: str) -> str:
    """Mock Hydra resolver for non-Hydra environments.

    Supports:
        - hydra:runtime.output_dir -> returns hydra.runtime.output_dir
        - hydra:runtime.choices.task -> returns hydra.runtime.choices.task

    Note: The hydra.runtime node must be set in the config before resolution.
    """
    return f"${{hydra.{key}}}"


def register_default_resolvers() -> None:
    """
    Register all resolvers commonly used across entrypoints.
    Safe to call multiple times.
    """
    _register("oc.load", _oc_load)
    _register("oc.merge", _oc_merge)
    _register(
        "eval", eval
    )  # allows arbitrary python code execution in configs using the ${eval:''} resolver
    _register("split", lambda s, idx: s.split("/")[int(idx)])  # split string
    _register("max", lambda x: max(x))
    _register("round_up", math.ceil)
    _register("round_down", math.floor)
    _register("sum_shapes", sum_shapes)
    _register("max_action_dim", max_action_dim)
    _register("max_state_dim", max_state_dim)
    _register("obs_image_steps", _get_obs_image_steps)
    _register("now", lambda pattern: datetime.now().strftime(pattern))
    _register("oc.env", lambda var, default=None: __import__("os").environ.get(var, default))
    _register("hydra", _hydra_resolver)

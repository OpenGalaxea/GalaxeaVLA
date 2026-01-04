import math
from pathlib import Path
from typing import Any, List, Callable, Optional
from omegaconf import OmegaConf


def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)

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

def register_default_resolvers() -> None:
    """
    Register all resolvers commonly used across entrypoints.
    Safe to call multiple times.
    """
    _register("oc.load", _oc_load)
    _register("eval", eval) # allows arbitrary python code execution in configs using the ${eval:''} resolver
    _register("split", lambda s, idx: s.split('/')[int(idx)]) # split string
    _register("max", lambda x: max(x))
    _register("round_up", math.ceil)
    _register("round_down", math.floor)
"""msgpack codec with numpy support, shared by serve.py and policy ws clients.

Wire format (matches OpenPI's hook so the two are interoperable):

    ndarray  → {"__ndarray__": True, "data": <bytes>, "dtype": str, "shape": tuple}
    np.generic → {"__npgeneric__": True, "data": <python scalar>, "dtype": str}

Decoder side recursively turns bytes keys into str keys, so consumers can use
the natural ``obj["key"]`` access regardless of the writer's locale.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


_UNSUPPORTED_KINDS = ("V", "O", "c")  # void, object, complex


def _pack(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_KINDS:
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _bytes_to_str_keys(obj: Any) -> Any:
    """Recursively decode bytes-typed dict keys to str."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj: dict[Any, Any]) -> Any:
    obj = _bytes_to_str_keys(obj)
    if "__ndarray__" in obj:
        return np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=tuple(obj["shape"]),
        )
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)

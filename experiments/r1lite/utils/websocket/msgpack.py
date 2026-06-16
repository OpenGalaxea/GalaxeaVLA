import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _get(obj, key):
    if key in obj:
        return obj[key]
    str_key = key.decode() if isinstance(key, bytes) else key
    if str_key in obj:
        return obj[str_key]
    bytes_key = str_key.encode() if isinstance(str_key, str) else str_key
    if bytes_key in obj:
        return obj[bytes_key]
    return None


def unpack_array(obj):
    if _get(obj, b"__ndarray__") is not None:
        return np.ndarray(
            buffer=_get(obj, b"data"),
            dtype=np.dtype(_get(obj, b"dtype")),
            shape=_get(obj, b"shape"),
        )
    if _get(obj, b"__npgeneric__") is not None:
        return np.dtype(_get(obj, b"dtype")).type(_get(obj, b"data"))
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


if __name__ == "__main__":
    # round-trip test for numpy arrays
    arr = np.random.rand(3, 224, 224).astype(np.float32)
    packed = packb({"image": arr, "label": "test"})
    unpacked = unpackb(packed)
    assert np.allclose(unpacked["image"], arr), "ndarray round-trip failed"
    assert unpacked["label"] == "test"

    # scalar test
    scalar = np.float32(3.14)
    packed_s = packb({"val": scalar})
    unpacked_s = unpackb(packed_s)
    assert np.isclose(unpacked_s["val"], scalar), "scalar round-trip failed"

    print("msgpack round-trip smoke test passed.")

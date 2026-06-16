"""Tests for utils/websocket/msgpack.py – numpy ↔ msgpack round-trips."""

import numpy as np
import pytest

from utils.websocket.msgpack import Packer, unpackb

_packer = Packer()


def packb(obj):
    return _packer.pack(obj)


class TestNdarrayRoundTrip:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8])
    def test_various_dtypes(self, dtype):
        arr = np.arange(6, dtype=dtype).reshape(2, 3)
        result = unpackb(packb(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == arr.dtype

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = unpackb(packb(arr))
        np.testing.assert_array_equal(result, arr)

    def test_scalar_round_trip(self):
        val = np.float32(3.14)
        result = unpackb(packb(val))
        assert isinstance(result, np.floating)
        np.testing.assert_almost_equal(float(result), float(val), decimal=5)

    def test_mixed_dict(self):
        data = {
            "image": np.zeros((2, 3), dtype=np.uint8),
            "label": "hello",
            "score": np.float64(0.95),
        }
        blob = packb(data)
        out = unpackb(blob)
        # msgpack decodes str keys as str (raw=False default in modern msgpack)
        img_key = "image" if "image" in out else b"image"
        np.testing.assert_array_equal(out[img_key], data["image"])
        label_key = "label" if "label" in out else b"label"
        expected_label = "hello" if isinstance(out[label_key], str) else b"hello"
        assert out[label_key] == expected_label
        score_key = "score" if "score" in out else b"score"
        np.testing.assert_almost_equal(float(out[score_key]), 0.95, decimal=5)

    def test_unsupported_dtype_raises(self):
        arr = np.array([1 + 2j, 3 + 4j])  # complex → dtype.kind == 'c'
        with pytest.raises(ValueError, match="Unsupported dtype"):
            packb(arr)

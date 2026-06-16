"""Tests for core/inference/factory.py – create_inference_engine."""

from unittest.mock import patch, MagicMock

import pytest


class TestCreateInferenceEngine:
    def _make_config(self):
        return {"websocket": {"host": "127.0.0.1", "port": 9999}}

    @patch("core.inference.factory.WebSocketClientEngine")
    def test_returns_websocket_engine(self, MockWSEngine):
        from core.inference.factory import create_inference_engine

        config = self._make_config()
        instance = MagicMock()
        MockWSEngine.return_value = instance

        result = create_inference_engine(config)
        MockWSEngine.assert_called_once_with(config)
        assert result is instance

    def test_missing_websocket_config_raises(self):
        from core.inference.factory import create_inference_engine

        with pytest.raises(KeyError):
            create_inference_engine({})

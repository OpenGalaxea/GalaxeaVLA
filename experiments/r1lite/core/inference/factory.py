from typing import Dict, Any
from loguru import logger

from core.inference.inference_engine import InferenceEngine
from core.inference.websocket_engine import WebSocketClientEngine


def create_inference_engine(config: Dict[str, Any]) -> InferenceEngine:
    logger.info("Creating WebSocket inference engine")
    return WebSocketClientEngine(config)


if __name__ == "__main__":
    config = {"websocket": {"host": "localhost", "port": 8080}}
    engine = create_inference_engine(config)
    print(f"Created engine: {type(engine).__name__}, uri={engine._uri}")
    print("InferenceEngine factory smoke test passed.")

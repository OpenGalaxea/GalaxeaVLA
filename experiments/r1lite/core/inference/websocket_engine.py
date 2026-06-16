import logging
import time
from typing import Dict, Tuple, Any

from typing_extensions import override
import websockets.sync.client

from core.inference.inference_engine import InferenceEngine
from utils.websocket.msgpack import Packer, unpackb
import os


class WebSocketClientEngine(InferenceEngine):
    def __init__(self, config: Dict[str, Any]):
        host = config["websocket"]["host"]
        port = config["websocket"]["port"]

        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()

    def connect(self):
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        old_proxy_settings = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy'),
                'all_proxy': os.environ.get('all_proxy')
        }

        while True:
            try:
                if 'http_proxy' in os.environ:
                    del os.environ['http_proxy']
                if 'https_proxy' in os.environ:
                    del os.environ['https_proxy']
                if 'all_proxy' in os.environ:
                    del os.environ['all_proxy']
                headers = {"Authorization": f"Api-Key "}
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                    ping_interval=None,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)
            finally:
                for key, value in old_proxy_settings.items():
                    if value is not None:
                        os.environ[key] = value
                    elif key in os.environ:
                        del os.environ[key]

    @override
    def predict_action(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = unpackb(response)
        if "error" in result:
            err = result["error"]
            raise RuntimeError(f"Inference server error (code={err.get('code')}): {err.get('message')}")
        return result


if __name__ == "__main__":
    # Test URI construction.
    config1 = {"websocket": {"host": "localhost", "port": 8080}}
    engine1 = WebSocketClientEngine(config1)
    assert engine1._uri == "ws://localhost:8080", f"got {engine1._uri}"

    config2 = {"websocket": {"host": "ws://10.0.0.1", "port": 9090}}
    engine2 = WebSocketClientEngine(config2)
    assert engine2._uri == "ws://10.0.0.1:9090", f"got {engine2._uri}"

    config3 = {"websocket": {"host": "ws://10.0.0.1", "port": None}}
    engine3 = WebSocketClientEngine(config3)
    assert engine3._uri == "ws://10.0.0.1", f"got {engine3._uri}"

    print("WebSocketClientEngine URI smoke test passed.")

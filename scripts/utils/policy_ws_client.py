from __future__ import annotations

import asyncio
from typing import Any

from g05.utils.websocket import packb, unpackb

import numpy as np


class PolicyWebSocketClient:
    def __init__(self, uri: str, timeout_s: float = 30.0):
        self.uri = uri
        self.timeout_s = timeout_s
        self._ws = None
        self.metadata: dict[str, Any] | None = None

    async def connect(self) -> dict[str, Any]:
        import websockets

        connect_kwargs = {
            "max_size": None,
            "ping_interval": 30,
            "ping_timeout": 120,
            "proxy": None,
        }
        try:
            self._ws = await websockets.connect(self.uri, **connect_kwargs)
        except TypeError as exc:
            if "proxy" not in str(exc):
                raise
            connect_kwargs.pop("proxy")
            self._ws = await websockets.connect(self.uri, **connect_kwargs)
        handshake = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        self.metadata = unpackb(handshake)
        return self.metadata

    async def infer(
        self, raw_obs: dict[str, Any], current_step: int | None = None
    ) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket client is not connected")
        if current_step is not None:
            raw_obs = {**raw_obs, "current_step": current_step}
        await asyncio.wait_for(self._ws.send(packb(raw_obs)), timeout=self.timeout_s)
        response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        return unpackb(response)

    async def reset(self) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket client is not connected")
        await asyncio.wait_for(self._ws.send(packb({"__reset__": True})), timeout=self.timeout_s)
        response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        resp = unpackb(response)
        if not isinstance(resp, dict) or not resp.get("__reset__"):
            raise RuntimeError(f"Unexpected reset response: {resp}")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> "PolicyWebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class MultiPolicyWSClient:
    def __init__(self, uri: str, num_connections: int, timeout_s: float = 30.0):
        self.uri = uri
        self.num_connections = num_connections
        self.timeout_s = timeout_s
        self.clients: list[PolicyWebSocketClient] = [
            PolicyWebSocketClient(uri, timeout_s) for _ in range(num_connections)
        ]
        self.need_obs: list[bool] = [True] * num_connections
        self.metadata: list[dict[str, Any]] = []
        self.action_chunks: list[dict | None] = [None] * num_connections
        self.chunk_steps: list[int] = [0] * num_connections
        self.action_steps: int = 0
        self.ensemble: bool = False

    async def connect_all(self) -> list[dict[str, Any]]:
        results = await asyncio.gather(*(c.connect() for c in self.clients))
        self.metadata = list(results)
        self.need_obs = [True] * self.num_connections
        self.action_steps = self.metadata[0]["action_steps"]
        self.ensemble = self.metadata[0].get("ensemble", False)
        self.action_chunks = [None] * self.num_connections
        self.chunk_steps = [0] * self.num_connections
        return self.metadata

    def needs_recompute(self, idx: int) -> bool:
        if self.action_chunks[idx] is None:
            return True
        if self.chunk_steps[idx] >= self.action_steps:
            return True
        chunk = self.action_chunks[idx]
        for arr in chunk.values():
            if (
                isinstance(arr, np.ndarray)
                and arr.ndim >= 1
                and self.chunk_steps[idx] >= arr.shape[0]
            ):
                return True
        return False

    def get_cached_action(self, idx: int) -> dict | None:
        if self.needs_recompute(idx):
            return None
        chunk = self.action_chunks[idx]
        if not chunk:
            return None
        step = self.chunk_steps[idx]
        single = {}
        for part, arr in chunk.items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and step < arr.shape[0]:
                single[part] = arr[step]
            else:
                single[part] = arr[0] if isinstance(arr, np.ndarray) and arr.ndim >= 1 else arr
        self.chunk_steps[idx] += 1
        return single

    def peek_remaining(self, idx: int) -> int:
        if self.action_chunks[idx] is None:
            return 0
        return self.action_steps - self.chunk_steps[idx]

    def clear_chunk(self, idx: int) -> None:
        self.action_chunks[idx] = None
        self.chunk_steps[idx] = 0
        self.need_obs[idx] = True

    async def infer_batch(
        self,
        indices: list[int],
        raw_obs_list: list[dict],
        current_steps: list[int | None] | None = None,
    ) -> None:
        if current_steps is None:
            current_steps = [None] * len(indices)
        # No explicit reset needed — server resets its cache automatically
        # when it receives a non-empty obs (implicit reset on recompute).
        coros = [
            self.clients[i].infer(raw_obs_list[j], current_steps[j]) for j, i in enumerate(indices)
        ]
        responses = await asyncio.gather(*coros)
        for j, i in enumerate(indices):
            resp = responses[j]
            chunk = resp.get("action_chunk")
            if chunk is not None:
                self.action_chunks[i] = chunk
                self.chunk_steps[i] = 0
                self.need_obs[i] = False
            else:
                action = resp.get("action", {})
                chunk = {}
                for part, val in action.items():
                    if isinstance(val, np.ndarray):
                        chunk[part] = val[np.newaxis, ...]
                    else:
                        chunk[part] = np.array([val])
                self.action_chunks[i] = chunk
                self.chunk_steps[i] = 0
                self.need_obs[i] = resp.get("need_obs", True)

    async def infer_all(
        self, raw_obs_list: list[dict | None], active_mask: list[bool] = None
    ) -> list[dict | None]:
        if active_mask is None:
            active_mask = [True] * self.num_connections
        coros = []
        indices = []
        for i in range(self.num_connections):
            if not active_mask[i]:
                continue
            if self.need_obs[i]:
                if raw_obs_list[i] is None:
                    raise ValueError(
                        f"Env {i} needs obs (need_obs=True) but raw_obs_list[{i}] is None"
                    )
                coros.append(self.clients[i].infer(raw_obs_list[i]))
            else:
                coros.append(self.clients[i].infer({}))
            indices.append(i)

        if not coros:
            return [None] * self.num_connections

        responses = await asyncio.gather(*coros)
        for j, resp in enumerate(responses):
            self.need_obs[indices[j]] = resp.get("need_obs", True)

        full_responses: list[dict | None] = [None] * self.num_connections
        for j, resp in enumerate(responses):
            full_responses[indices[j]] = resp
        return full_responses

    async def reset_connection(self, idx: int) -> None:
        await self.clients[idx].reset()
        self.clear_chunk(idx)

    async def reset_all(self) -> None:
        await asyncio.gather(*(c.reset() for c in self.clients))
        self.need_obs = [True] * self.num_connections
        self.action_chunks = [None] * self.num_connections
        self.chunk_steps = [0] * self.num_connections

    async def close_all(self) -> None:
        await asyncio.gather(*(c.close() for c in self.clients))

    async def __aenter__(self) -> "MultiPolicyWSClient":
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close_all()

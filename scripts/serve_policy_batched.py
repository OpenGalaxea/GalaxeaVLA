# ruff: noqa: E402
"""WebSocket policy server with dynamic batch inference.

Extends serve_policy.py with a DynamicBatcher that collects requests from
multiple clients and dispatches them as micro-batches to the GPU, improving
throughput when multiple robots connect simultaneously.

Batching triggers:
  - len(batch) >= max_batch_size  (default 5)
  - elapsed >= max_wait_ms       (default 1000ms)

ChunkedPolicyWrapper cache hits bypass the batcher entirely — only recomputes
(cached chunk exhausted) go through the queue.

Usage:
    python scripts/serve_policy_batched.py \\
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \\
        --host 0.0.0.0 --port 8765 \\
        --max_batch_size 5 --max_wait_ms 1000.0 \\
        eval_embodiment=galaxea_r1lite
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import dataclasses
import functools
import logging
import time
from collections import deque

import numpy as np
import torch
import websockets

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.common.pytorch_utils import dict_apply
from g05.utils.websocket import packb, unpackb

from scripts.serve_policy import (
    build_obs_dict,
    _parse_task_and_plan,
    _serialize_error,
    setup,
)
from experiments.libero.libero_eval_utils import LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD

logger = logging.getLogger(__name__)


# ──────── Data classes for batch coordination ────────


@dataclasses.dataclass
class PendingRequest:
    raw_obs: dict
    future: asyncio.Future
    enqueued_at: float


@dataclasses.dataclass
class RequestOutcome:
    request: PendingRequest
    action: dict | None = None
    error: Exception | None = None


class QueueFullError(RuntimeError):
    pass


# ──────── DynamicBatcher ────────


class DynamicBatcher:
    """Collects inference requests and dispatches them as micro-batches.

    A single asyncio task (_worker_loop) consumes from an asyncio.Queue.
    Requests accumulate until either:
      - len(pending) >= max_batch_size, or
      - elapsed time since first request >= max_wait_s

    Then the batch is sent to a ThreadPoolExecutor for GPU inference,
    and each client's future is resolved with the result.
    """

    def __init__(
        self,
        policy,
        processor,
        *,
        max_batch_size: int = 5,
        max_wait_ms: float = 1000.0,
        max_queue_size: int = 256,
        device: str = "cuda",
    ):
        self.inferencer = PolicyInferencer(policy, processor, device=device)
        self.processor = processor
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.max_queue_size = max(1, int(max_queue_size))
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue(maxsize=self.max_queue_size)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="policy-batcher"
        )
        self.worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)
        while not self.queue.empty():
            req = self.queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("Dynamic batcher stopped"))

    def submit(self, raw_obs: dict) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        req = PendingRequest(raw_obs=raw_obs, future=future, enqueued_at=time.monotonic())
        try:
            self.queue.put_nowait(req)
        except asyncio.QueueFull as exc:
            future.cancel()
            raise QueueFullError("Inference queue is full") from exc
        return future

    async def _worker_loop(self) -> None:
        while True:
            try:
                req = await self.queue.get()
                pending = [req]

                # Drain any requests already waiting in the queue (arrived
                # while previous batch was being processed on GPU).
                while not self.queue.empty() and len(pending) < self.max_batch_size:
                    pending.append(self.queue.get_nowait())

                # If we haven't reached max_batch_size yet, wait up to
                # max_wait_s from NOW for more requests to arrive.
                if len(pending) < self.max_batch_size:
                    deadline = time.monotonic() + self.max_wait_s
                    while len(pending) < self.max_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            pending.append(
                                await asyncio.wait_for(self.queue.get(), timeout=remaining)
                            )
                        except asyncio.TimeoutError:
                            break

                logger.info(
                    "Dispatching batch: %d requests collected (max=%d, waited=%.0fms)",
                    len(pending),
                    self.max_batch_size,
                    (time.monotonic() - pending[0].enqueued_at) * 1000.0,
                )

                loop = asyncio.get_running_loop()
                outcomes = await loop.run_in_executor(self.executor, self._process_batch, pending)

                for outcome in outcomes:
                    if outcome.request.future.done():
                        continue
                    if outcome.error is not None:
                        outcome.request.future.set_exception(outcome.error)
                    else:
                        outcome.request.future.set_result(outcome.action)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("DynamicBatcher worker loop error")
                for p in pending:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("Batcher worker loop error"))
                await asyncio.sleep(0.1)

    def _process_batch(self, requests: list[PendingRequest]) -> list[RequestOutcome]:
        outcomes: list[RequestOutcome] = []
        valid_requests: list[PendingRequest] = []
        valid_obs_dicts: list[dict] = []

        for req in requests:
            try:
                _parse_task_and_plan(req.raw_obs)
                obs_dict = build_obs_dict(req.raw_obs, self.processor)
            except Exception as exc:
                outcomes.append(RequestOutcome(request=req, error=exc))
                continue
            valid_requests.append(req)
            valid_obs_dicts.append(obs_dict)

        if not valid_obs_dicts:
            return outcomes

        try:
            actions, timing = self.inferencer.infer_with_timing(valid_obs_dicts)
        except Exception as exc:
            logger.exception("Dynamic batch inference failed")
            outcomes.extend(RequestOutcome(request=r, error=exc) for r in valid_requests)
            return outcomes

        for req, action in zip(valid_requests, actions):
            outcomes.append(RequestOutcome(request=req, action=action))

        elapsed_ms = (time.monotonic() - min(r.enqueued_at for r in requests)) * 1000.0
        logger.info(
            "Batch forward: batch_size=%d, waited=%.1fms, infer=%.1fms",
            len(valid_obs_dicts),
            (min(r.enqueued_at for r in requests) - requests[0].enqueued_at) * 1000.0
            if len(requests) > 1
            else 0.0,
            elapsed_ms,
        )
        timing_str = ", ".join(f"{k}={v:.1f}" for k, v in sorted(timing.items()))
        logger.info("Batch timing breakdown: %s", timing_str)
        return outcomes


# ──────── ChunkedPolicyWrapper (batched variant) ────────


class ChunkedPolicyWrapper:
    """Wraps batch inference with chunk caching and step-by-step serving.

    Decides when to call the model (recompute) vs serve from cache.
    One instance per client connection.

    Cache hits bypass the batcher entirely. Only recomputes (need_obs == True)
    go through batcher.submit(), allowing cross-client batching of the
    expensive GPU forward pass.
    """

    def __init__(
        self,
        batcher: DynamicBatcher,
        processor,
        *,
        action_steps: int = 16,
        request_timeout_s: float = 30.0,
        default_gripper_value: float = 1.0,
    ):
        self.batcher = batcher
        self.processor = processor
        self.action_steps = action_steps
        self.request_timeout_s = request_timeout_s
        self._cached_chunk = None
        self._chunk_step = 0
        self._cot_text = None
        self._last_raw_chunk: dict | None = None
        self._default_gripper_value = default_gripper_value
        self._gripper_keys = self._resolve_gripper_keys()

    @property
    def need_obs(self) -> bool:
        if self._cached_chunk is None:
            return True
        if self._chunk_step >= self.action_steps:
            return True
        for arr in self._cached_chunk.values():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and self._chunk_step >= arr.shape[0]:
                return True
        return False

    @property
    def full_chunk(self) -> dict | None:
        return self._last_raw_chunk

    def _resolve_gripper_keys(self) -> dict[str, int]:
        result: dict[str, int] = {}
        procs = {}
        if hasattr(self.processor, "processors"):
            procs = self.processor.processors
        else:
            procs = {"_": self.processor}
        for _name, proc in procs.items():
            merger = getattr(proc, "action_state_merger", None)
            meta = getattr(merger, "max_action_shape_meta", None)
            if meta and isinstance(meta, dict):
                for k, dim in meta.items():
                    if "gripper" in k:
                        result[k] = dim
        return result

    def _current_gripper_action_value(self, raw_obs: dict, key: str) -> float:
        state = raw_obs.get("state", {}) if isinstance(raw_obs, dict) else {}
        if key not in state:
            return float(self._default_gripper_value)
        scalar = float(np.asarray(state[key], dtype=np.float32).reshape(-1)[0])
        return 1.0 if scalar >= LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD else 0.0

    def _mark_absent_gripper(self, action: dict, absent_keys: set[str], raw_obs: dict) -> dict:
        absent_grippers = absent_keys & set(self._gripper_keys.keys())
        if not absent_grippers:
            return action
        sample_val = next(v for k, v in action.items() if not str(k).startswith("_"))
        is_tensor = isinstance(sample_val, torch.Tensor)
        if is_tensor:
            horizon = sample_val.shape[1]
            device = sample_val.device
        else:
            horizon = sample_val.shape[0]
        for gk in absent_grippers:
            gdim = self._gripper_keys[gk]
            fill_value = self._current_gripper_action_value(raw_obs, gk)
            if is_tensor:
                action[gk] = torch.full(
                    (1, horizon, gdim),
                    fill_value,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                action[gk] = np.full(
                    (horizon, gdim),
                    fill_value,
                    dtype=np.float32,
                )
        return action

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        self._last_raw_chunk = None
        if self.need_obs:
            if not raw_obs:
                raise ValueError("need obs for recompute but got empty")
            t0 = time.monotonic()
            future = self.batcher.submit(raw_obs)
            try:
                action = await asyncio.wait_for(future, timeout=self.request_timeout_s)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Inference request timed out after {self.request_timeout_s}s")
            self._cot_text = action.pop("_cot_text", None)
            absent_keys = action.pop("_absent_keys", set())
            action = self._mark_absent_gripper(action, absent_keys, raw_obs)
            self._cached_chunk = dict_apply(
                action,
                lambda x: x[0].numpy() if isinstance(x, torch.Tensor) else x,
            )
            self._chunk_step = 0
            self._last_raw_chunk = self._cached_chunk
            logger.debug(
                "Recompute: %.1fms (next %d from cache)",
                (time.monotonic() - t0) * 1000,
                self.action_steps - 1,
            )

        single_step = {}
        for part, arr in self._cached_chunk.items():
            if arr.ndim >= 1 and self._chunk_step < arr.shape[0]:
                single_step[part] = arr[self._chunk_step]
            else:
                single_step[part] = arr[0] if arr.ndim >= 1 else arr
        self._chunk_step += 1
        return single_step, self._cot_text

    def reset(self):
        self._cached_chunk = None
        self._chunk_step = 0


class EnsembleChunkedPolicyWrapper:
    """Chunked serving with temporal ensemble of overlapping predictions.

    Instead of discarding predictions beyond action_steps, stores all
    overlapping predictions in a per-timestep deque and averages them
    when serving each step.
    """

    def __init__(
        self,
        batcher: DynamicBatcher,
        processor,
        *,
        action_steps: int = 16,
        request_timeout_s: float = 30.0,
        default_gripper_value: float = 1.0,
    ):
        self.batcher = batcher
        self.processor = processor
        self.action_steps = action_steps
        self.request_timeout_s = request_timeout_s
        self._global_step = 0
        self._steps_since_recompute = 0
        self._last_recompute_step = 0
        self._ensemble_buf: dict[str, dict[int, deque]] = {}
        self._last_raw_chunk: dict | None = None
        self._cot_text = None
        self._default_gripper_value = default_gripper_value
        self._gripper_keys = self._resolve_gripper_keys()

    def set_step(self, step: int) -> None:
        self._global_step = step
        self._steps_since_recompute = step - self._last_recompute_step

    def _resolve_gripper_keys(self) -> dict[str, int]:
        result: dict[str, int] = {}
        procs = {}
        if hasattr(self.processor, "processors"):
            procs = self.processor.processors
        else:
            procs = {"_": self.processor}
        for _name, proc in procs.items():
            merger = getattr(proc, "action_state_merger", None)
            meta = getattr(merger, "max_action_shape_meta", None)
            if meta and isinstance(meta, dict):
                for k, dim in meta.items():
                    if "gripper" in k:
                        result[k] = dim
        return result

    def _current_gripper_action_value(self, raw_obs: dict, key: str) -> float:
        state = raw_obs.get("state", {}) if isinstance(raw_obs, dict) else {}
        if key not in state:
            return float(self._default_gripper_value)
        scalar = float(np.asarray(state[key], dtype=np.float32).reshape(-1)[0])
        return 1.0 if scalar >= LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD else 0.0

    def _mark_absent_gripper(self, action: dict, absent_keys: set[str], raw_obs: dict) -> dict:
        absent_grippers = absent_keys & set(self._gripper_keys.keys())
        if not absent_grippers:
            return action
        sample_val = next(v for k, v in action.items() if not str(k).startswith("_"))
        is_tensor = isinstance(sample_val, torch.Tensor)
        if is_tensor:
            horizon = sample_val.shape[1]
            device = sample_val.device
        else:
            horizon = sample_val.shape[0]
        for gk in absent_grippers:
            gdim = self._gripper_keys[gk]
            fill_value = self._current_gripper_action_value(raw_obs, gk)
            if is_tensor:
                action[gk] = torch.full(
                    (1, horizon, gdim),
                    fill_value,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                action[gk] = np.full(
                    (horizon, gdim),
                    fill_value,
                    dtype=np.float32,
                )
        return action

    @property
    def need_obs(self) -> bool:
        if not self._ensemble_buf:
            return True
        if self._steps_since_recompute >= self.action_steps:
            return True
        for ts_map in self._ensemble_buf.values():
            if self._global_step not in ts_map or len(ts_map[self._global_step]) == 0:
                return True
        return False

    @property
    def full_chunk(self) -> dict | None:
        return self._last_raw_chunk

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        self._last_raw_chunk = None
        if self.need_obs:
            if not raw_obs:
                raise ValueError("need obs for recompute but got empty")
            t0 = time.monotonic()
            future = self.batcher.submit(raw_obs)
            try:
                action = await asyncio.wait_for(future, timeout=self.request_timeout_s)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Inference request timed out after {self.request_timeout_s}s")
            self._cot_text = action.pop("_cot_text", None)
            absent_keys = action.pop("_absent_keys", set())
            action = self._mark_absent_gripper(action, absent_keys, raw_obs)
            chunk = dict_apply(
                action,
                lambda x: x[0].numpy() if isinstance(x, torch.Tensor) else x,
            )
            self._last_raw_chunk = chunk
            start = self._global_step
            chunk_len = None
            for part, arr in chunk.items():
                if part.startswith("_"):
                    continue
                if chunk_len is None:
                    chunk_len = arr.shape[0]
                if part not in self._ensemble_buf:
                    self._ensemble_buf[part] = {}
                for t in range(arr.shape[0]):
                    abs_t = start + t
                    if abs_t not in self._ensemble_buf[part]:
                        self._ensemble_buf[part][abs_t] = deque()
                    self._ensemble_buf[part][abs_t].append(arr[t])
            logger.info(
                "Ensemble recompute: %.1fms (chunk covers steps %d..%d)",
                (time.monotonic() - t0) * 1000,
                start,
                start + chunk_len - 1,
            )
            self._steps_since_recompute = 0
            self._last_recompute_step = self._global_step

        single_step = {}
        for part, ts_map in self._ensemble_buf.items():
            preds = ts_map.get(self._global_step)
            if preds is None or len(preds) == 0:
                raise ValueError(f"No predictions for {part} at step {self._global_step}")
            stacked = np.stack(list(preds))
            single_step[part] = stacked.mean(axis=0)

        for part in self._ensemble_buf:
            self._ensemble_buf[part].pop(self._global_step, None)
        self._ensemble_buf = {p: ts for p, ts in self._ensemble_buf.items() if ts}

        self._global_step += 1
        self._steps_since_recompute += 1
        return single_step, self._cot_text

    def reset(self):
        self._ensemble_buf = {}
        self._global_step = 0
        self._steps_since_recompute = 0
        self._last_recompute_step = 0
        self._cot_text = None


# ──────── WebSocket handler ────────


async def handler(
    ws,
    batcher: DynamicBatcher,
    processor,
    *,
    action_steps: int = 16,
    request_timeout_ms: float = 120000.0,
    ensemble: bool = False,
):
    client = ws.remote_address
    logger.info(f"Client connected: {client} (action_steps={action_steps})")

    await ws.send(packb({"action_steps": action_steps, "ensemble": ensemble}))
    if ensemble:
        policy = EnsembleChunkedPolicyWrapper(
            batcher,
            processor,
            action_steps=action_steps,
            request_timeout_s=request_timeout_ms / 1000.0,
        )
    else:
        policy = ChunkedPolicyWrapper(
            batcher,
            processor,
            action_steps=action_steps,
            request_timeout_s=request_timeout_ms / 1000.0,
        )

    try:
        async for msg in ws:
            t0 = time.monotonic()
            try:
                raw_obs = unpackb(msg)
                if isinstance(raw_obs, dict) and raw_obs.get("__reset__"):
                    policy.reset()
                    await ws.send(packb({"__reset__": True, "need_obs": True}))
                    continue
                current_step = (
                    raw_obs.pop("current_step", None) if isinstance(raw_obs, dict) else None
                )
                if current_step is not None and hasattr(policy, "set_step"):
                    policy.set_step(int(current_step))
                # Non-empty obs means client wants a recompute — reset server
                # cache so get_action always goes through the batcher.
                # This avoids needing a separate reset round-trip.
                if raw_obs and not isinstance(policy, EnsembleChunkedPolicyWrapper):
                    policy.reset()
                action, cot_text = await policy.get_action(raw_obs)
                resp = {"action": action, "need_obs": policy.need_obs}
                chunk = policy.full_chunk
                if chunk is not None:
                    resp["action_chunk"] = chunk
                if cot_text is not None:
                    resp["cot_text"] = cot_text
                await ws.send(packb(resp))
            except QueueFullError as exc:
                await ws.send(
                    packb({**_serialize_error(503, str(exc)), "need_obs": policy.need_obs})
                )
            except TimeoutError as exc:
                await ws.send(
                    packb({**_serialize_error(504, str(exc)), "need_obs": policy.need_obs})
                )
            except ValueError as exc:
                await ws.send(
                    packb({**_serialize_error(400, str(exc)), "need_obs": policy.need_obs})
                )
            except Exception as exc:
                logger.exception("Inference request failed")
                await ws.send(
                    packb({**_serialize_error(500, str(exc)), "need_obs": policy.need_obs})
                )
            dt = (time.monotonic() - t0) * 1000
            logger.debug(f"Step: {dt:.1f}ms")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client}")


# ──────── Server ────────


async def serve(
    policy,
    processor,
    host: str,
    port: int,
    *,
    device: str = "cuda",
    action_steps: int = 16,
    max_batch_size: int = 5,
    max_wait_ms: float = 1000.0,
    max_queue_size: int = 256,
    request_timeout_ms: float = 120000.0,
    ensemble: bool = False,
):
    batcher = DynamicBatcher(
        policy,
        processor,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_queue_size=max_queue_size,
        device=device,
    )
    batcher.start()

    bound_handler = functools.partial(
        handler,
        batcher=batcher,
        processor=processor,
        action_steps=action_steps,
        request_timeout_ms=request_timeout_ms,
        ensemble=ensemble,
    )
    try:
        async with websockets.serve(
            bound_handler, host, port, max_size=None, ping_interval=30, ping_timeout=300
        ):
            mode = "RTC" if action_steps == 1 else f"chunk({action_steps})"
            logger.info(
                "Batched policy server listening on ws://%s:%d "
                "(mode=%s, device=%s, max_batch=%d, max_wait=%.0fms, ensemble=%s)",
                host,
                port,
                mode,
                device,
                max_batch_size,
                max_wait_ms,
                "on" if ensemble else "off",
            )
            await asyncio.Future()
    finally:
        await batcher.close()


# ──────── main ────────


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket policy server with dynamic batch inference"
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda", help="CUDA device for model inference")
    parser.add_argument(
        "--action_steps",
        type=int,
        default=16,
        help="Steps to serve per inference. >1=chunk mode (default: 16), 1=RTC",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=40,
        help="Batch size threshold — dispatch when reached (default: 40)",
    )
    parser.add_argument(
        "--max_wait_ms",
        type=float,
        default=2000.0,
        help="Max wait time in ms before dispatching partial batch (default: 2000)",
    )
    parser.add_argument(
        "--max_queue_size",
        type=int,
        default=256,
        help="Max pending requests in queue (backpressure, default: 256)",
    )
    parser.add_argument(
        "--request_timeout_ms",
        type=float,
        default=120000.0,
        help="Per-request future timeout in ms (default: 120000)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Enable temporal ensemble of overlapping chunk predictions",
    )
    args, remaining = parser.parse_known_args()

    overrides = [r for r in remaining if "=" in r]

    run_dir = find_run_dir(args.ckpt_path)
    print(f"Found run dir: {run_dir}")

    cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    eval_embodiment = cfg.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in cfg.data:
        filter_embodiment(cfg, eval_embodiment)

    from g05.utils.logging.logging_config import setup_logging

    setup_logging(log_level=logging.INFO, is_main_process=True)

    policy, processor = setup(cfg, device=args.device)
    logger.info("Model and processor loaded on %s.", args.device)

    asyncio.run(
        serve(
            policy,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_steps=args.action_steps,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            max_queue_size=args.max_queue_size,
            request_timeout_ms=args.request_timeout_ms,
            ensemble=args.ensemble,
        )
    )


if __name__ == "__main__":
    main()

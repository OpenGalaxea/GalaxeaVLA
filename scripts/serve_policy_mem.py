"""WebSocket policy server for MEM multi-frame inference with history buffers.

Based on serve_policy.py, with per-camera frame buffers (deque) for MEM video
encoders where obs_size > 1. Single-frame models are also supported with
deque maxlen=1, equivalent to the old expand behavior.

Usage:
    # Load config from the training run_dir.
    python scripts/serve_policy_mem.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        --action_steps 15 \
        eval_embodiment=galaxea_r1lite

    # Manually specify a task config when the run_dir has no .hydra/config.yaml.
    python scripts/serve_policy_mem.py \
        --ckpt_path /path/to/model_state_dict.pt \
        --task_config configs/task/pretrain/10k_pretrain_full_AC2_2cb_qwen35_2b_mem.yaml \
        --action_steps 15 \
        eval_embodiment=galaxea_r1lite
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import logging
import time

from g05.utils.websocket import packb, unpackb
import numpy as np
import torch

from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.common.pytorch_utils import dict_apply
from g05.data_processor.processor.mixture_processor import MixtureProcessor

from g05.utils.data.processor_utils import build_processors
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.checkpoint.ckpt_utils import (
    find_run_dir,
    load_config_from_run_dir,
    load_config_from_task_yaml,
)
from g05.models.g05.inferencer import (
    PolicyInferencer,
    resolve_processor,
)

import websockets

logger = logging.getLogger(__name__)


def _resolve_stats_path(cfg) -> Path:
    """Resolve dataset stats path for both `last.pt` and `checkpoints/step_x.pt`."""
    run_dir = cfg.get("run_dir", None)
    if run_dir:
        candidate = Path(str(run_dir)) / "dataset_stats.json"
        if candidate.exists():
            return candidate

    ckpt_path = Path(cfg.ckpt_path)
    candidates = [
        ckpt_path.parent / "dataset_stats.json",
        ckpt_path.parent.parent / "dataset_stats.json",
    ]
    if cfg.get("datastatics_path", None):
        candidates.append(Path(str(cfg.datastatics_path)))

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"dataset_stats.json not found (tried: {checked})")


# ──────── setup: model + processor initialization ────────


def setup(cfg, device: str = "cuda"):
    """Load policy and processor, aligned with eval_open_loop.py:362-413."""
    model = load_model_from_checkpoint(
        cfg.model.model_arch,
        cfg.ckpt_path,
        device=device,
        extra_prefixes=["normalizer."],
        eval_mode=False,
    )

    if cfg.model.get("model_weights_to_bf16", True):
        model = model.to(torch.bfloat16)

    model.apply_fp32_params()

    policy = model.eval()
    if hasattr(policy, "action_tokenizer"):
        policy.action_tokenizer.to(device)

    if cfg.model.get("use_torch_compile", False):
        logger.info("Compiling model with torch.compile (first inference will be slow)...")
        policy = torch.compile(policy, mode="max-autotune")

    stats_path = _resolve_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(stats_path)

    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    from g05.data_processor.transforms.action_filter import BaseActionFilter

    def _neutralize_action_filter(p):
        safe_filter = BaseActionFilter()
        safe_filter.set_shape_meta(p.shape_meta)
        p.action_filter = safe_filter

    if isinstance(processor, MixtureProcessor):
        for emb_name in processor.processors:
            _neutralize_action_filter(processor.processors[emb_name])
    else:
        _neutralize_action_filter(processor)

    action_horizon = int(cfg.data.action_size)
    if isinstance(processor, MixtureProcessor):
        for emb_name, sub_processor in processor.processors.items():
            sub_processor.action_horizon = action_horizon
    else:
        processor.action_horizon = action_horizon

    return policy, processor


# ──────── input validation ────────


def _parse_task_and_plan(raw_obs: dict) -> None:
    """Parse a [PLAN] marker from task and split it into task + plan."""
    task_str = raw_obs.get("task", "")
    if "[PLAN]" in task_str:
        parts = task_str.split("[PLAN]", 1)
        raw_obs["task"] = parts[0].strip()
        raw_obs["plan"] = parts[1].strip()


def _validate_obs(raw_obs: dict, processor) -> None:
    """Validate that client raw_obs satisfies the raw_shape protocol."""
    p = resolve_processor(processor, raw_obs)
    sm = p.shape_meta

    for required in ("images", "state", "task"):
        if required not in raw_obs:
            raise ValueError(f"raw_obs is missing required field '{required}'")

    if (
        isinstance(processor, MixtureProcessor)
        and len(processor.processors) > 1
        and "embodiment_type" not in raw_obs
    ):
        raise ValueError("Mixture mode requires raw_obs to contain an 'embodiment_type' field")

    if not isinstance(raw_obs["task"], str):
        raise ValueError(f"raw_obs['task'] must be str, got {type(raw_obs['task'])}")

    if "frequency" in raw_obs and not isinstance(raw_obs["frequency"], (int, float, np.number)):
        raise ValueError(
            f"raw_obs['frequency'] must be numeric, got {type(raw_obs['frequency'])}"
        )

    if "coarse_task" in raw_obs and not isinstance(raw_obs["coarse_task"], str):
        raise ValueError(
            f"raw_obs['coarse_task'] must be str, got {type(raw_obs['coarse_task'])}"
        )

    if "plan" in raw_obs and not isinstance(raw_obs["plan"], str):
        raise ValueError(f"raw_obs['plan'] must be str, got {type(raw_obs['plan'])}")

    expected_img_keys = {m["key"] for m in sm["images"]}
    actual_img_keys = set(raw_obs["images"].keys())
    missing = expected_img_keys - actual_img_keys
    if missing:
        raise ValueError(f"images is missing keys defined by shape_meta: {missing}")
    extra = actual_img_keys - expected_img_keys
    if extra:
        logger.warning(f"images contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["images"]:
        img = raw_obs["images"][m["key"]]
        if not isinstance(img, np.ndarray):
            raise ValueError(f"images['{m['key']}'] must be np.ndarray, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"images['{m['key']}'] must be [C,H,W] (3D), got shape={img.shape}")
        if img.dtype != np.uint8:
            logger.warning(f"images['{m['key']}'] dtype={img.dtype}, expected uint8")

    expected_state_keys = {m["key"] for m in sm["state"]}
    actual_state_keys = set(raw_obs["state"].keys())
    missing = expected_state_keys - actual_state_keys
    if missing:
        raise ValueError(f"state is missing keys defined by shape_meta: {missing}")
    extra = actual_state_keys - expected_state_keys
    if extra:
        logger.warning(f"state contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["state"]:
        s = raw_obs["state"][m["key"]]
        if not isinstance(s, np.ndarray):
            raise ValueError(f"state['{m['key']}'] must be np.ndarray, got {type(s)}")
        if s.ndim != 1:
            raise ValueError(f"state['{m['key']}'] must be [D] (1D), got shape={s.shape}")
        expected_dim = m["raw_shape"]
        if s.shape[0] != expected_dim:
            raise ValueError(
                f"state['{m['key']}'] dim={s.shape[0]}, expected raw_shape {expected_dim}"
            )


# ──────── obs -> preprocess input construction ────────


def build_obs_dict(raw_obs: dict, processor, frame_buffers=None, state_buffers=None) -> dict:
    """Build the processor.preprocess input from client raw observations.

    Data format is aligned with base_lerobot_dataset.py:304-323 __getitem__ output.
    Client input always uses raw_shape; shape appears only after processor-internal
    transforms.

    When frame_buffers/state_buffers are provided in multi-frame MEM mode, take
    K historical frames from the buffers. Otherwise, fall back to single-frame
    expand, preserving old obs_size=1 behavior.
    """
    _validate_obs(raw_obs, processor)

    p = resolve_processor(processor, raw_obs)
    num_obs_steps = p.num_obs_steps
    action_horizon = getattr(p, "action_horizon", num_obs_steps)

    expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
    expected_state_keys = {m["key"] for m in p.shape_meta["state"]}

    if frame_buffers and state_buffers:
        images = {
            k: torch.stack(list(frame_buffers[k])) for k in expected_img_keys if k in frame_buffers
        }
        state = {
            k: torch.stack(list(state_buffers[k]))
            for k in expected_state_keys
            if k in state_buffers
        }
    else:
        images = {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1, -1, -1)
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        }
        state = {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1)
            .float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        }

    data = {
        "images": images,
        "state": state,
        "task": raw_obs["task"],
        "action": {
            m["key"]: torch.zeros(action_horizon, m["raw_shape"]) for m in p.shape_meta["action"]
        },
        "action_is_pad": torch.ones(action_horizon, dtype=torch.bool),
        "state_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "image_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "idx": 0,
    }
    if "coarse_task" in raw_obs:
        data["coarse_task"] = raw_obs["coarse_task"]
    if "plan" in raw_obs:
        data["plan"] = raw_obs["plan"]
    frequency = raw_obs.get("frequency", 15)
    data["frequency"] = float(frequency)
    if isinstance(processor, MixtureProcessor):
        data["embodiment"] = raw_obs.get("embodiment_type") or next(iter(processor.processors))
    return data


# ──────── WebSocket handler ────────


def _serialize_error(code: int, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


class ChunkedPolicyWrapper:
    """Wraps model inference with chunk caching, step-by-step serving, and frame buffers.

    Maintains per-camera frame buffers (deque) for multi-frame models (MEM
    video encoder, obs_size > 1). When obs_size == 1, buffers hold exactly
    one frame, equivalent to the old expand() behavior.

    One instance per client connection.
    """

    def __init__(self, inferencer, processor, *, action_steps: int = 16):
        self.inferencer = inferencer
        self.processor = processor
        self.action_steps = action_steps
        self._cached_chunk = None
        self._chunk_step = 0
        self._cot_text = None

        # Frame buffer for multi-frame models (MEM video encoder)
        if isinstance(processor, MixtureProcessor):
            p = next(iter(processor.processors.values()))
        else:
            p = processor
        self._num_obs_steps = p.num_obs_steps
        self._cam_keys = [m["key"] for m in p.shape_meta["images"]]
        self._state_keys = [m["key"] for m in p.shape_meta["state"]]
        self._frame_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._num_obs_steps) for k in self._cam_keys
        }
        self._state_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._num_obs_steps) for k in self._state_keys
        }
        self._buffers_initialized = False
        if self._num_obs_steps > 1:
            logger.info(
                "Frame buffer enabled: obs_size=%d, cameras=%s",
                self._num_obs_steps,
                self._cam_keys,
            )

    @property
    def need_obs(self) -> bool:
        """Whether the next call to get_action needs a real observation."""
        return self._cached_chunk is None or self._chunk_step >= self.action_steps

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        """Return single-step action ``{part: ndarray[dim]}``.

        On recompute: updates frame buffers, builds multi-frame obs, runs inference.
        """
        if self.need_obs:
            if not raw_obs:
                raise ValueError("need obs for recompute but got empty")
            t0 = time.monotonic()
            _parse_task_and_plan(raw_obs)
            t1 = time.monotonic()
            self._update_buffers(raw_obs)
            t2 = time.monotonic()
            obs_dict = build_obs_dict(
                raw_obs,
                self.processor,
                frame_buffers=self._frame_buffers,
                state_buffers=self._state_buffers,
            )
            t3 = time.monotonic()
            actions = await asyncio.to_thread(self.inferencer.infer, [obs_dict])
            t4 = time.monotonic()
            action = actions[0]
            self._cot_text = action.pop("_cot_text", None)
            self._cached_chunk = dict_apply(
                action, lambda x: x[0].numpy() if isinstance(x, torch.Tensor) else x
            )
            t5 = time.monotonic()
            self._chunk_step = 0
            logger.info(
                "Recompute: %.1fms total | buffers=%.1fms build_obs=%.1fms infer=%.1fms post=%.1fms (next %d from cache)",
                (t5 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t4 - t3) * 1000,
                (t5 - t4) * 1000,
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
        """Invalidate cache and clear frame buffers (episode boundary)."""
        self._cached_chunk = None
        self._chunk_step = 0
        for buf in self._frame_buffers.values():
            buf.clear()
        for buf in self._state_buffers.values():
            buf.clear()
        self._buffers_initialized = False

    def _update_buffers(self, raw_obs: dict) -> None:
        """Append current observation to frame/state buffers.

        On first call after reset, fills all slots with the first observation
        (clamp-to-start, matching training's episode boundary behavior).
        """
        p = resolve_processor(self.processor, raw_obs)
        expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
        expected_state_keys = {m["key"] for m in p.shape_meta["state"]}

        img_tensors = {
            k: torch.from_numpy(np.array(v, copy=True))
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        }
        state_tensors = {
            k: torch.from_numpy(np.array(v, copy=True)).float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        }

        if not self._buffers_initialized:
            for _ in range(self._num_obs_steps):
                for k, t in img_tensors.items():
                    self._frame_buffers[k].append(t.clone())
                for k, t in state_tensors.items():
                    self._state_buffers[k].append(t.clone())
            self._buffers_initialized = True
        else:
            for k, t in img_tensors.items():
                self._frame_buffers[k].append(t)
            for k, t in state_tensors.items():
                self._state_buffers[k].append(t)


async def handler(ws, inferencer, processor, action_steps=16, visualize: bool = False):
    """Handle a single WebSocket connection."""
    client = ws.remote_address
    logger.info(f"Client connected: {client} (action_steps={action_steps})")

    await ws.send(packb({"action_steps": action_steps}))
    policy = ChunkedPolicyWrapper(inferencer, processor, action_steps=action_steps)

    visualizer = None
    if visualize:
        from scripts.utils.mem_live_viz import MemLiveVisualizer

        visualizer = MemLiveVisualizer(window_prefix=f"mem[{client[0]}:{client[1]}]")

    try:
        async for msg in ws:
            t0 = time.monotonic()
            try:
                raw_obs = unpackb(msg)
                if isinstance(raw_obs, dict) and raw_obs.get("__reset__"):
                    policy.reset()
                    if visualizer is not None:
                        visualizer.reset()
                    await ws.send(packb({"__reset__": True}))
                    logger.info("Episode reset from client %s", client)
                    continue
                was_fresh = policy.need_obs
                action, cot_text = await policy.get_action(raw_obs)
                resp = {"action": action, "need_obs": policy.need_obs}
                if cot_text is not None:
                    resp["cot_text"] = cot_text
                await ws.send(packb(resp))
                if visualizer is not None:
                    visualizer.update(
                        images=raw_obs.get("images") if isinstance(raw_obs, dict) else None,
                        cot_text=cot_text,
                        is_fresh=was_fresh,
                        chunk_step=policy._chunk_step,
                        action_steps=action_steps,
                    )
            except ValueError as exc:
                await ws.send(packb(_serialize_error(400, str(exc))))
            except Exception as exc:
                logger.exception("Inference request failed")
                await ws.send(packb(_serialize_error(500, str(exc))))
            dt = (time.monotonic() - t0) * 1000
            logger.debug(f"{'Infer' if policy._chunk_step == 1 else 'Cache'}: {dt:.1f}ms")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client}")
    finally:
        if visualizer is not None:
            visualizer.close()


async def serve(
    handler_fn,
    policy,
    processor,
    host: str,
    port: int,
    *,
    device: str = "cuda",
    action_steps: int = 16,
    visualize: bool = False,
):
    """Start the WebSocket server."""
    inferencer = PolicyInferencer(policy, processor, device=device)
    bound_handler = functools.partial(
        handler_fn,
        inferencer=inferencer,
        processor=processor,
        action_steps=action_steps,
        visualize=visualize,
    )
    async with websockets.serve(bound_handler, host, port, max_size=None):
        mode = "RTC" if action_steps == 1 else f"chunk({action_steps})"
        logger.info(
            "Policy server listening on ws://%s:%d (mode=%s, device=%s)",
            host,
            port,
            mode,
            device,
        )
        await asyncio.Future()


# ──────── main ────────


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket policy server — MEM multi-frame variant"
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--task_config",
        default=None,
        help="Override task config YAML (when run dir has no .hydra/config.yaml)",
    )
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
        "--visualize",
        action="store_true",
        help="Open cv2 window showing head/wrist frames + COT/affordance overlay (needs DISPLAY; "
        "first connected client gets the window, others run normally).",
    )
    args, remaining = parser.parse_known_args()

    overrides = [r for r in remaining if "=" in r]

    # --- Config loading. ---
    if args.task_config:
        cfg = load_config_from_task_yaml(args.task_config, args.ckpt_path, overrides)
        print(f"Loaded config from task YAML: {args.task_config}")
    else:
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
            handler,
            policy,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_steps=args.action_steps,
            visualize=args.visualize,
        )
    )


if __name__ == "__main__":
    main()

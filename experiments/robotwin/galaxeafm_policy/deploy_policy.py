import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g05.models.g05.inferencer import PolicyInferencer
from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.data_processor.transforms.action_filter import BaseActionFilter
from g05.utils.checkpoint.checkpoint_utils import load_state_dict_safely
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.processor_utils import build_processors

logger = logging.getLogger(__name__)

CAMERA_KEY_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

def _meta_raw_dim(meta: Dict[str, Any]) -> int:
    raw_shape = meta["raw_shape"]
    if isinstance(raw_shape, int):
        return raw_shape
    raise ValueError(f"Expected integer raw_shape for RoboTwin meta, got: {raw_shape}")


def _flat_dim_from_meta(meta_list: list[Dict[str, Any]]) -> int:
    max_end = 0
    for meta in meta_list:
        start = int(meta["start_index"])
        end = start + _meta_raw_dim(meta)
        max_end = max(max_end, end)

    occupied = np.zeros(max_end, dtype=bool)
    for meta in meta_list:
        start = int(meta["start_index"])
        end = start + _meta_raw_dim(meta)
        if occupied[start:end].any():
            raise ValueError(f"Overlapping RoboTwin meta slice: {meta}")
        occupied[start:end] = True
    if not occupied.all():
        missing = np.where(~occupied)[0].tolist()
        raise ValueError(f"RoboTwin meta slices do not cover flat vector indices: {missing}")
    return int(max_end)


def _split_flat_vector(
    vector: np.ndarray,
    meta_list: list[Dict[str, Any]],
    *,
    repeat_steps: int,
) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {}
    for meta in meta_list:
        key = str(meta["key"])
        start = int(meta["start_index"])
        dim = _meta_raw_dim(meta)
        part = vector[start : start + dim]
        payload[key] = torch.from_numpy(part).unsqueeze(0).expand(repeat_steps, -1).float()
    return payload


def _zero_action_by_meta(
    meta_list: list[Dict[str, Any]],
    *,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    return {
        str(meta["key"]): torch.zeros(horizon, _meta_raw_dim(meta), dtype=torch.float32)
        for meta in meta_list
    }


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _register_hydra_builtin_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "now",
        lambda pattern, _tz="": time.strftime(pattern),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "oc.env",
        lambda key, default=None: (
            os.environ[key]
            if key in os.environ
            else default
            if default is not None
            else (_ for _ in ()).throw(KeyError(f"Env var '{key}' not set"))
        ),
        replace=True,
    )


def _register_project_oc_load_resolver() -> None:
    def _oc_load_from_project(path: str, key: Optional[str] = None) -> Any:
        load_path = Path(path)
        if not load_path.is_absolute():
            load_path = (PROJECT_ROOT / load_path).resolve()
        cfg = OmegaConf.load(load_path)
        if key is None or key == "":
            return cfg
        return OmegaConf.select(cfg, key)

    OmegaConf.register_new_resolver("oc.load", _oc_load_from_project, replace=True)


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        rel_text = relative.as_posix()
    else:
        if _is_none_like(sim_cfg_name):
            rel_text = "sim_robotwin.yaml"
        else:
            rel_text = str(sim_cfg_name).strip()

    if rel_text.endswith(".yaml"):
        rel_text = rel_text[:-5]
    if rel_text == "":
        raise ValueError("Resolved sim config name is empty.")
    return rel_text


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    register_default_resolvers()
    _register_hydra_builtin_resolvers()
    _register_project_oc_load_resolver()

    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # `oc.load:configs/...` in our configs relies on project-root-relative paths.
    # RoboTwin launches this module with cwd at `<robotwin_root>`, so we must pin
    # cwd to PROJECT_ROOT during composition.
    prev_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(config_name=config_name, overrides=overrides)
    finally:
        os.chdir(prev_cwd)

    # Do not global-resolve here: sim config may contain `${hydra:...}` fields
    # (e.g. EVALUATION.output_dir), but this process is not launched by Hydra.
    # We only read a small subset of non-hydra-dependent fields below.
    return cfg


def _apply_checkpoint_model_config(cfg: DictConfig, checkpoint_file: Path) -> DictConfig:
    checkpoint_run_dir = checkpoint_file.parent.parent
    config_path = checkpoint_run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint Hydra config not found: {config_path}")

    checkpoint_cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(checkpoint_cfg, False)
    if checkpoint_cfg.get("model") is None:
        raise KeyError(f"Checkpoint config missing `model`: {config_path}")
    if checkpoint_cfg.get("tokenizer") is None:
        raise KeyError(f"Checkpoint config missing `tokenizer`: {config_path}")

    # RoboTwin runtime config owns eval/env fields, while the checkpoint owns the
    # model architecture and tokenizer settings needed to instantiate weights.
    OmegaConf.set_struct(cfg, False)
    cfg.model = OmegaConf.create(OmegaConf.to_container(checkpoint_cfg.model, resolve=False))
    cfg.tokenizer = OmegaConf.create(OmegaConf.to_container(checkpoint_cfg.tokenizer, resolve=False))

    # Keep checkpoint configs explicit: legacy private-package targets should be
    # fixed in the bundle config, not silently rewritten at runtime.
    model_config_text = str(OmegaConf.to_container(cfg.model, resolve=False))
    tokenizer_config_text = str(OmegaConf.to_container(cfg.tokenizer, resolve=False))
    if "galaxea_fm." in model_config_text or "galaxea_fm." in tokenizer_config_text:
        raise ValueError(
            f"Checkpoint model/tokenizer config still references legacy `galaxea_fm.*`: {config_path}"
        )

    hf_processor_path = Path(str(cfg.model.model_arch.hf_processor_path)).expanduser()
    if not hf_processor_path.is_absolute():
        hf_processor_path = PROJECT_ROOT / hf_processor_path
    if not (hf_processor_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"HF processor tokenizer.json not found: {hf_processor_path}")

    processor_hf_path = Path(
        str(cfg.model.processor.tokenizer_params.pretrained_model_name_or_path)
    ).expanduser()
    if not processor_hf_path.is_absolute():
        processor_hf_path = PROJECT_ROOT / processor_hf_path
    if not (processor_hf_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Processor tokenizer.json not found: {processor_hf_path}")

    action_tokenizer_path = Path(str(cfg.tokenizer.vq_config.ckpt_dir)).expanduser()
    if not action_tokenizer_path.is_absolute():
        action_tokenizer_path = PROJECT_ROOT / action_tokenizer_path
    if not action_tokenizer_path.is_file():
        raise FileNotFoundError(f"Action tokenizer checkpoint not found: {action_tokenizer_path}")

    model_tokenizer_vq_config = OmegaConf.select(cfg, "model.tokenizer.vq_config")
    if model_tokenizer_vq_config is None:
        raise KeyError("Checkpoint model config missing `model.tokenizer.vq_config`.")

    logger.info("Loaded RoboTwin model config from checkpoint: %s", config_path)
    logger.info("Using HF processor: %s", hf_processor_path)
    logger.info("Using action tokenizer: %s", action_tokenizer_path)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _as_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image HWC with 3 channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


class GalaxeaFMRobotWinPolicy:
    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        seed: Optional[int],
    ) -> None:
        if sigma_shift is not None:
            raise ValueError("`sigma_shift` is not supported by current GalaxeaFM inference path.")
        if float(text_cfg_scale) != 1.0:
            raise ValueError("`text_cfg_scale` is not supported; only 1.0 is allowed.")
        if str(negative_prompt) != "":
            raise ValueError("`negative_prompt` is not supported; only empty string is allowed.")
        if str(rand_device) != "cpu":
            raise ValueError("`rand_device` is not supported; only 'cpu' is allowed.")
        if bool(tiled):
            raise ValueError("`tiled` is not supported; only False is allowed.")
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        model = instantiate(cfg.model.model_arch)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing `model_state_dict`.")
        model = load_state_dict_safely(
            model,
            checkpoint["model_state_dict"],
            extra_prefixes=["normalizer."],
        )
        del checkpoint

        if model_dtype in {torch.bfloat16, torch.float16}:
            model = model.to(model_dtype)
        if hasattr(model, "apply_fp32_params"):
            model.apply_fp32_params()
        self.policy = model.to(device).eval()
        if hasattr(self.policy, "action_tokenizer"):
            self.policy.action_tokenizer.to(device)

        processor = build_processors(cfg)
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        processor.set_normalizer_from_stats(dataset_stats)
        processor.eval()

        self._embodiment_key: Optional[str] = None
        if isinstance(processor, MixtureProcessor):
            if "robotwin" in processor.processors:
                self._embodiment_key = "robotwin"
                self._sub_processor = processor["robotwin"]
            elif len(processor.processors) == 1:
                self._embodiment_key = next(iter(processor.processors.keys()))
                self._sub_processor = processor[self._embodiment_key]
            else:
                raise ValueError(
                    "MixtureProcessor has multiple embodiments but no `robotwin` key."
                )
        else:
            self._sub_processor = processor

        self.processor = processor
        self.inferencer = PolicyInferencer(self.policy, processor, device=device)

        self.image_meta = self._sub_processor.shape_meta["images"]
        self.state_meta = self._sub_processor.shape_meta["state"]
        self.action_meta = self._sub_processor.shape_meta["action"]
        self.state_dim = _flat_dim_from_meta(self.state_meta)
        self.action_dim = _flat_dim_from_meta(self.action_meta)

        image_meta_by_key = {meta["key"]: meta for meta in self.image_meta}
        for target_key in CAMERA_KEY_MAP.values():
            if target_key not in image_meta_by_key:
                raise ValueError(
                    f"Missing image key in processor.shape_meta['images']: {target_key}"
                )
        self.image_meta_by_key = image_meta_by_key
        self._logged_image_shapes: set[str] = set()

        self.num_obs_steps = int(self._sub_processor.num_obs_steps)
        if self.num_obs_steps <= 0:
            raise ValueError(f"num_obs_steps must be positive, got {self.num_obs_steps}")

        model_horizon = int(cfg.data.action_size)
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if action_horizon > model_horizon:
            raise ValueError(
                f"action_horizon ({action_horizon}) exceeds model horizon ({model_horizon})."
            )
        self.model_action_horizon = model_horizon
        self.exec_action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, self.exec_action_horizon)))

        fm_helper = getattr(self.policy.model, "fm_helper", None)
        if fm_helper is None:
            raise ValueError("Current policy model does not expose `fm_helper`.")
        fm_helper.num_inference_steps = int(num_inference_steps)

        if isinstance(self.processor, MixtureProcessor):
            neutral_filter = BaseActionFilter()
            neutral_filter.set_shape_meta(self._sub_processor.shape_meta)
            self._sub_processor.action_filter = neutral_filter
            self._sub_processor.action_horizon = model_horizon
        else:
            neutral_filter = BaseActionFilter()
            neutral_filter.set_shape_meta(self.processor.shape_meta)
            self.processor.action_filter = neutral_filter
            self.processor.action_horizon = model_horizon

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self.timing_enabled = bool(timing_enabled)
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized GalaxeaFMRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.exec_action_horizon,
            self.replan_steps,
        )

    def _build_obs_dict(self, observation: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        if "observation" not in observation:
            raise KeyError("Missing `observation` in RoboTwin observation payload.")
        if "joint_action" not in observation or "vector" not in observation["joint_action"]:
            raise KeyError("Missing `joint_action.vector` in RoboTwin observation payload.")

        obs_data = observation["observation"]
        images: dict[str, torch.Tensor] = {}
        for source_key, target_key in CAMERA_KEY_MAP.items():
            if source_key not in obs_data or "rgb" not in obs_data[source_key]:
                raise KeyError(f"Missing `{source_key}.rgb` in RoboTwin observation payload.")
            chw = _as_chw_uint8(obs_data[source_key]["rgb"])
            if chw.shape[0] != 3:
                raise ValueError(
                    f"Image channel mismatch for {target_key}: got {tuple(chw.shape)}, "
                    "expected CHW with 3 channels."
                )
            if target_key not in self._logged_image_shapes:
                logger.info("RoboTwin image shape detected: %s=%s", target_key, tuple(chw.shape))
                self._logged_image_shapes.add(target_key)
            images[target_key] = (
                torch.from_numpy(chw)
                .unsqueeze(0)
                .expand(self.num_obs_steps, -1, -1, -1)
            )

        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: got {state_vector.shape[0]}, expected {self.state_dim}"
            )
        state = _split_flat_vector(
            state_vector,
            self.state_meta,
            repeat_steps=self.num_obs_steps,
        )
        action = _zero_action_by_meta(self.action_meta, horizon=self.model_action_horizon)
        payload: Dict[str, Any] = {
            "images": images,
            "state": state,
            "task": str(instruction),
            "action": action,
            "action_is_pad": torch.ones(self.model_action_horizon, dtype=torch.bool),
            "state_is_pad": torch.zeros(self.num_obs_steps, dtype=torch.bool),
            "image_is_pad": torch.zeros(self.num_obs_steps, dtype=torch.bool),
            "idx": 0,
            "frequency": float(15.0),
        }
        if self._embodiment_key is not None:
            payload["embodiment"] = self._embodiment_key
        return payload

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        obs_dict = self._build_obs_dict(observation=observation, instruction=instruction)

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        pred = self.inferencer.infer([obs_dict])[0]
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        expected_keys = [str(meta["key"]) for meta in self.action_meta]
        action_keys = [key for key in pred.keys() if not key.startswith("_")]
        extra_keys = sorted(set(action_keys) - set(expected_keys))
        missing_keys = [key for key in expected_keys if key not in pred]
        if extra_keys or missing_keys:
            raise ValueError(
                f"RoboTwin action key mismatch: missing={missing_keys}, extra={extra_keys}, "
                f"got={sorted(action_keys)}"
            )

        action_chunk: np.ndarray | None = None
        for meta in self.action_meta:
            key = str(meta["key"])
            value = pred[key]
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            if arr.ndim == 3:
                if arr.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1 for action `{key}`, got {arr.shape}")
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected action `{key}` tensor [T,D] or [1,T,D], got shape {tuple(arr.shape)}"
                )

            dim = _meta_raw_dim(meta)
            if arr.shape[1] != dim:
                raise ValueError(
                    f"Action dim mismatch for `{key}`: got {arr.shape[1]}, expected {dim}"
                )
            if action_chunk is None:
                action_chunk = np.zeros((arr.shape[0], self.action_dim), dtype=np.float32)
            elif action_chunk.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"Action horizon mismatch for `{key}`: got {arr.shape[0]}, "
                    f"expected {action_chunk.shape[0]}"
                )
            start = int(meta["start_index"])
            action_chunk[:, start : start + dim] = arr.astype(np.float32)

        if action_chunk is None:
            raise RuntimeError("No action generated for RoboTwin evaluation.")
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        max_horizon = min(self.exec_action_horizon, action_chunk.shape[0])
        n_exec = min(self.replan_steps, max_horizon)
        if n_exec <= 0:
            raise RuntimeError("No action generated for current step.")
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty (replan step)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            raise RuntimeError("Pending action queue is empty after inference.")

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def get_model(usr_args: Dict[str, Any]):
    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")
    checkpoint_path = str(Path(str(checkpoint_path)).expanduser().resolve())
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    cfg = _apply_checkpoint_model_config(cfg, checkpoint_file)

    device = str(usr_args.get("device") or cfg.EVALUATION.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or "bf16")
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        action_horizon = int(cfg.data.action_size)
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.replan_steps)

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.text_cfg_scale))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.negative_prompt))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.rand_device))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.tiled))
    timing_enabled = _parse_bool(usr_args.get("timing_enabled", cfg.EVALUATION.timing_enabled))
    seed = _parse_optional_int(usr_args.get("seed"))

    prev_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        policy = GalaxeaFMRobotWinPolicy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        seed=seed,
    )
    finally:
        os.chdir(prev_cwd)
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()

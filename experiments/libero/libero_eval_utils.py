from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from pathlib import Path

import imageio
import numpy as np
import yaml

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_GRIPPER_HOLD_SENTINEL = -100.0
LIBERO_GRIPPER_HOLD_ATOL = 1e-4
LIBERO_GRIPPER_OPEN_COMMAND = -1.0
LIBERO_GRIPPER_CLOSE_COMMAND = 1.0
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015


def _find_libero_source_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "LIBERO package is not installed. Install it before running LIBERO eval."
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


def ensure_libero_import() -> Path:
    ensure_libero_config()
    from libero import libero as libero_pkg

    return Path(libero_pkg.__file__).resolve().parent.parent.parent


def ensure_libero_config() -> Path:
    libero_config_path = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    libero_config_path.mkdir(parents=True, exist_ok=True)
    config_file = libero_config_path / "config.yaml"
    if config_file.exists():
        return config_file

    source_root = _find_libero_source_root()
    benchmark_root = source_root / "libero" / "libero"

    default_path_dict = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(source_root / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(default_path_dict, f, sort_keys=False)
    return config_file


def get_libero_env(task, resolution: int, seed: int | None):
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def get_max_steps(task_suite_name: str) -> int:
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in max_steps_by_suite:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return max_steps_by_suite[task_suite_name]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def extract_libero_images(obs: dict) -> dict[str, np.ndarray]:
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    }


def _to_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))


def build_libero_raw_obs(
    obs: dict,
    task_description: str,
    shape_meta: dict | None = None,
    embodiment_type: str | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    images_hwc = extract_libero_images(obs)
    ee_pose = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]),
        )
    ).astype(np.float32)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_qpos.size < 1:
        raise ValueError("robot0_gripper_qpos is empty")

    raw_obs = {
        "images": {},
        "state": {},
        "task": str(task_description),
    }

    raw_obs["images"] = {
        "image": _to_chw_uint8(images_hwc["image"]),
        "wrist_image": _to_chw_uint8(images_hwc["wrist_image"]),
    }

    default_state = np.concatenate((ee_pose, gripper_qpos[:1])).astype(np.float32)
    raw_obs["state"] = {
        "right_ee_pose": ee_pose,
        "right_gripper": gripper_qpos[:1].astype(np.float32),
    }

    if shape_meta is not None:
        state_keys = {meta["key"] for meta in shape_meta["state"]}
        if state_keys == {"default"}:
            raw_obs["state"] = {"default": default_state}

        image_keys = {meta["key"] for meta in shape_meta["images"]}
        if image_keys != {"image", "wrist_image"}:
            raise ValueError(f"Unsupported LIBERO image keys: {sorted(image_keys)}")

    if embodiment_type is not None:
        raw_obs["embodiment_type"] = embodiment_type

    return raw_obs, images_hwc


def _first_action_step(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0, 0]
    raise ValueError(f"Unsupported action shape: {arr.shape}")


def action_dict_gripper_scalar(action_dict: dict[str, np.ndarray]) -> float | None:
    gripper_key = next((key for key in action_dict if key.endswith("gripper")), None)
    if gripper_key is None:
        if list(action_dict.keys()) == ["default"]:
            arr = np.asarray(action_dict["default"], dtype=np.float32).reshape(-1)
            return float(arr[-1]) if arr.size else None
        return None
    arr = np.asarray(action_dict[gripper_key], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else None


def _is_gripper_hold_sentinel(value: np.ndarray | float) -> bool:
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    return bool(np.isclose(scalar, LIBERO_GRIPPER_HOLD_SENTINEL, atol=LIBERO_GRIPPER_HOLD_ATOL))


def libero_gripper_state_to_command(obs: dict) -> float:
    if obs is None or "robot0_gripper_qpos" not in obs:
        raise ValueError("LIBERO gripper hold sentinel requires current obs['robot0_gripper_qpos']")
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if qpos.size < 1:
        raise ValueError("robot0_gripper_qpos is empty")
    width = float(np.max(np.abs(qpos)))
    if width >= LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD:
        return LIBERO_GRIPPER_OPEN_COMMAND
    return LIBERO_GRIPPER_CLOSE_COMMAND


class LiberoGripperCommandState:
    def __init__(self, current_command: float | None = None):
        self.current_command = current_command

    @classmethod
    def from_obs(cls, obs: dict) -> "LiberoGripperCommandState":
        return cls(libero_gripper_state_to_command(obs))

    def reset_from_obs(self, obs: dict) -> None:
        self.current_command = libero_gripper_state_to_command(obs)

    def set_env_command(self, command: float) -> None:
        self.current_command = float(command)

    def command_for_policy_value(
        self,
        value: np.ndarray | float,
        *,
        obs: dict | None = None,
    ) -> float:
        if _is_gripper_hold_sentinel(value):
            if self.current_command is None:
                if obs is None:
                    raise ValueError("LIBERO gripper hold sentinel requires current obs or state")
                self.reset_from_obs(obs)
            return float(self.current_command)

        command = gripper_to_libero_command(value)
        self.current_command = command
        return command


def gripper_to_libero_command(value: np.ndarray | float, *, obs: dict | None = None) -> float:
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    if _is_gripper_hold_sentinel(scalar):
        return libero_gripper_state_to_command(obs)
    if 0.0 <= scalar <= 1.0:
        return float(LIBERO_GRIPPER_CLOSE_COMMAND if scalar <= 0.5 else LIBERO_GRIPPER_OPEN_COMMAND)
    return float(LIBERO_GRIPPER_OPEN_COMMAND if scalar > 0.0 else LIBERO_GRIPPER_CLOSE_COMMAND)


def action_dict_to_libero_actions(action_dict: dict[str, np.ndarray]) -> np.ndarray:
    action_keys = list(action_dict.keys())

    if action_keys == ["default"]:
        action = np.asarray(action_dict["default"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"Legacy LIBERO action must be [T,7], got {action.shape}")
        action = action.copy()
        action[:, -1] = [gripper_to_libero_command(x) for x in action[:, -1]]
        return action

    pose_key = next((key for key in action_keys if key.endswith("ee_pose")), None)
    gripper_key = next((key for key in action_keys if key.endswith("gripper")), None)
    if pose_key is None or gripper_key is None:
        raise ValueError(f"Unsupported LIBERO action keys: {action_keys}")

    pose = np.asarray(action_dict[pose_key], dtype=np.float32)
    gripper = np.asarray(action_dict[gripper_key], dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]
    if gripper.ndim == 3 and gripper.shape[0] == 1:
        gripper = gripper[0]
    if pose.ndim != 2 or pose.shape[1] != 6:
        raise ValueError(f"LIBERO ee_pose action must be [T,6] after postprocess, got {pose.shape}")
    if gripper.ndim != 2 or gripper.shape[1] != 1 or gripper.shape[0] != pose.shape[0]:
        raise ValueError(f"LIBERO gripper action must be [T,1] matching pose, got {gripper.shape}")
    gripper_cmd = np.asarray(
        [gripper_to_libero_command(x) for x in gripper[:, 0]], dtype=np.float32
    )
    return np.concatenate((pose, gripper_cmd[:, None]), axis=1)


def single_action_to_libero_action(
    action_dict: dict,
    *,
    obs: dict | None = None,
    gripper_state: LiberoGripperCommandState | None = None,
) -> list[float]:
    action_keys = list(action_dict.keys())

    if action_keys == ["default"]:
        arr = np.asarray(action_dict["default"], dtype=np.float32).reshape(-1)
        if arr.size != 7:
            raise ValueError(f"Legacy LIBERO single action must be 7-dim, got {arr.size}")
        arr = arr.copy()
        arr[-1] = (
            gripper_state.command_for_policy_value(arr[-1], obs=obs)
            if gripper_state is not None
            else gripper_to_libero_command(arr[-1], obs=obs)
        )
        return arr.tolist()

    pose_key = next((key for key in action_keys if key.endswith("ee_pose")), None)
    gripper_key = next((key for key in action_keys if key.endswith("gripper")), None)
    if pose_key is None:
        raise ValueError(f"Unsupported LIBERO action keys: {action_keys}")

    pose = np.asarray(action_dict[pose_key], dtype=np.float32).reshape(-1)
    if pose.size != 6:
        raise ValueError(f"LIBERO ee_pose single action must be 6-dim, got {pose.size}")

    if gripper_key is None:
        # The server returns only confidently-predicted keys; a missing gripper
        # means the AR head dropped it this chunk. Hold the last commanded
        # gripper value (the hold sentinel resolves to current_command / obs).
        gripper_cmd = (
            gripper_state.command_for_policy_value(LIBERO_GRIPPER_HOLD_SENTINEL, obs=obs)
            if gripper_state is not None
            else libero_gripper_state_to_command(obs)
        )
    else:
        gripper = np.asarray(action_dict[gripper_key], dtype=np.float32).reshape(-1)
        gripper_cmd = (
            gripper_state.command_for_policy_value(gripper[0], obs=obs)
            if gripper_state is not None
            else gripper_to_libero_command(gripper[0], obs=obs)
        )
    return np.concatenate([pose, [gripper_cmd]]).astype(np.float32).tolist()


def save_rollout_video(
    output_dir: Path,
    task_description: str,
    episode_idx: int,
    success: bool,
    replay_images: list[dict[str, np.ndarray]],
    fps: int = 10,
) -> Path | None:
    if not replay_images:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    task_segment = task_description.lower().replace(" ", "_").replace("/", "_")
    suffix = "success" if success else "failure"
    video_path = output_dir / f"rollout_{task_segment[:80]}_episode{episode_idx}_{suffix}.mp4"

    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in replay_images:
            ordered = [frame[key] for key in ("image", "wrist_image") if key in frame]
            if not ordered:
                continue
            writer.append_data(np.concatenate(ordered, axis=1) if len(ordered) > 1 else ordered[0])

    return video_path

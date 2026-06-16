"""Canonical SO100/SO101 LeRobot v3 dataset adapter.

SO100 public datasets use many camera names for the same physical roles.  This
adapter keeps one training embodiment by mapping raw camera names onto the model
contract: one exterior camera plus left/right wrist cameras.
"""

from __future__ import annotations

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from g05.data.base_lerobot_dataset import MAX_GETITEM_ATTEMPT
from g05.data.base_lerobot_datasetV3 import BaseLerobotDatasetV3

logger = logging.getLogger(__name__)


CANONICAL_EXTERIOR_KEY = "exterior"
CANONICAL_WRIST_LEFT_KEY = "wrist_left"
CANONICAL_WRIST_RIGHT_KEY = "wrist_right"


def _normalize_camera_name(key: str) -> str:
    name = key.split("observation.images.")[-1]
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", name)


LEFT_WRIST_NAMES = {
    "left_wrist",
    "wrist_left",
    "images_wrist_left",
    "left_follower",
    "left",
}

RIGHT_WRIST_NAMES = {
    "right_wrist",
    "wrist_right",
    "images_wrist_right",
    "right_follower",
    "right",
}

WRIST_LIKE_NAMES = {
    "wrist",
    "wristcam",
    "wrist_cam",
    "wristcamera",
    "wrist_camera",
    "wristview",
    "gripper",
    "gripper_view",
    "grap",
    "hand",
    "handeye",
    "hand_eye_cam",
    "eye_in_hand",
    "eyeinhand",
    "eyeonhand",
    "eye",
    "onhand",
    "endeffector",
    "end",
    "claw",
    "on_robot",
    "onboard",
    "robo_cam",
    "robocam",
    "robotarm00",
    "arm_camera",
    "arm_cam",
    "armcam",
    "attached_arm",
    "follower_wrist",
}

IGNORED_CAMERA_NAMES = {"depth", "realsense_depth", "seg"}

EXTERIOR_NAME_PRIORITIES = [
    (
        0,
        {
            "front",
            "front_l",
            "front_r",
            "frontview",
            "frontcamera",
            "front_cam",
            "frontcam",
            "main_camera",
            "main",
            "center_cam",
        },
    ),
    (
        10,
        {
            "laptop",
            "laptop1",
            "laptop2",
            "phone",
            "phone2",
            "iphone",
            "mac",
            "webcam",
            "webcam0",
            "webcam1",
            "webcam2",
            "camera",
            "new_camera",
            "cam",
            "camera1",
            "camera2",
            "camera_2",
            "camera_4",
            "cam1",
            "cam2",
            "c1",
            "c2",
            "c3",
            "logitech",
            "logitech_1",
            "logitech_2",
            "logi",
            "logic_camera",
            "hd500",
            "usbcam",
        },
    ),
    (
        20,
        {
            "side",
            "side1",
            "side_view",
            "sideview",
            "side_camera",
            "side_cam",
            "realsense_side",
            "extside",
            "rightfront",
        },
    ),
    (
        30,
        {
            "top",
            "top0",
            "top_camera",
            "topcamera",
            "topview",
            "top_view",
            "topdown",
            "top_down",
            "above",
            "overhead",
            "overhead_camera",
            "bird",
            "birdeye",
        },
    ),
    (
        40,
        {
            "base",
            "base_right",
            "stationary",
            "left_stationary",
            "right_stationary",
            "context",
            "workspace",
            "background",
            "global",
            "global_cam",
            "scene",
            "third",
            "thirdperson",
            "tabletop",
            "tripod",
            "desktop",
            "monitor",
            "pov",
            "perspective",
            "head",
            "head_camera",
            "body",
            "realsense",
            "realsense1",
            "realsense2",
            "realsense_rgb",
            "realsensergb",
            "rs_rgb",
            "rs_415_rgb",
            "rs_435_rgb",
            "rs_camera_color",
            "oak_color_camera",
            "orbbec",
            "wide",
            "rgb",
            "zrgb",
        },
    ),
]

FALLBACK_EXTERIOR_NAMES = {
    "airial",
    "blackboard",
    "lego",
    "robot",
    "detatched_arm",
    "arm",
    "flower",
    "so100",
    "crane",
    "red",
    "blue",
    "black",
    "pole",
}


def _exterior_priority(name: str) -> int:
    for priority, names in EXTERIOR_NAME_PRIORITIES:
        if name in names:
            return priority
    if name in FALLBACK_EXTERIOR_NAMES:
        return 100
    return 1000


def _camera_slot(key: str) -> Optional[str]:
    name = _normalize_camera_name(key)
    if name in IGNORED_CAMERA_NAMES:
        return None
    if name in LEFT_WRIST_NAMES:
        return CANONICAL_WRIST_LEFT_KEY
    if name in RIGHT_WRIST_NAMES:
        return CANONICAL_WRIST_RIGHT_KEY
    if name in WRIST_LIKE_NAMES:
        return CANONICAL_WRIST_RIGHT_KEY
    return CANONICAL_EXTERIOR_KEY


class SO100CanonicalLerobotDatasetV3(BaseLerobotDatasetV3):
    """Map heterogeneous SO100/SO101 camera names to a single model schema.

    Camera augmentation is training-only and controlled by ``random_drop_camera``.
    It keeps at most two real camera streams per episode, maps them into the
    deploy-consistent exterior/wrist_right slots, and sometimes swaps the two.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        manifest_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        max_dataset_dirs: Optional[int] = None,
        random_drop_camera: bool = False,
        *args,
        **kwargs,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.random_drop_camera = random_drop_camera

        if self.manifest_path is not None and self.dataset_root is not None:
            expanded_dirs = self._load_dataset_dirs_from_manifest()
        else:
            expanded_dirs = list(dataset_dirs)

        if max_dataset_dirs is not None:
            expanded_dirs = expanded_dirs[: int(max_dataset_dirs)]
        super().__init__(dataset_dirs=expanded_dirs, *args, **kwargs)

    def _load_dataset_dirs_from_manifest(self) -> List[str]:
        manifest = json.loads(self.manifest_path.read_text())
        repos = []
        seen = set()
        for group in manifest["groups"].values():
            for repo in group["repos"]:
                if repo in seen:
                    continue
                seen.add(repo)
                repos.append(repo)

        dataset_dirs = []
        missing = []
        for repo in sorted(repos):
            path = self.dataset_root / repo
            if (path / "meta" / "info.json").exists():
                dataset_dirs.append(str(path))
            else:
                missing.append(repo)

        if missing:
            logger.warning(
                "SO100 canonical manifest referenced %s missing dataset dirs; first few: %s",
                len(missing),
                missing[:5],
            )
        if not dataset_dirs:
            raise ValueError(
                f"No usable SO100 dataset directories found from manifest {self.manifest_path}"
            )
        return dataset_dirs

    def _build_delta_timestamps(self, fps, past_action_size, action_size) -> Dict[str, list]:
        """Build query offsets for action/state only.

        Raw image keys differ per dataset directory.  LeRobot v3 video loading already
        returns the raw camera frames for the current sample, so canonical image slots
        are selected later in :meth:`_get_image`.
        """
        query_offsets_by_key = {}

        obs_size = self.obs_size
        obs_stride = self.obs_stride

        for meta in self.state_meta:
            state_offsets = [
                (meta["time_offset"] + step) / fps
                for step in reversed(range(0, -obs_size * obs_stride, -obs_stride))
            ]
            meta["query_positions"] = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                state_offsets,
            )

        for meta in self.action_meta:
            action_offsets = [
                (meta["time_offset"] + step) / fps for step in range(-past_action_size, action_size)
            ]
            meta["query_positions"] = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                action_offsets,
            )

        return query_offsets_by_key

    @staticmethod
    def _is_raw_image_key(key: str, value: Any) -> bool:
        return key.startswith("observation.images.") and isinstance(value, torch.Tensor)

    def _available_image_keys(self, lerobot_sample: Dict[str, Any]) -> List[str]:
        return sorted(
            key for key, value in lerobot_sample.items() if self._is_raw_image_key(key, value)
        )

    def _select_raw_image_key(
        self, canonical_key: str, lerobot_sample: Dict[str, Any]
    ) -> Optional[str]:
        candidates = self._available_image_keys(lerobot_sample)
        if canonical_key == CANONICAL_EXTERIOR_KEY:
            exterior = [key for key in candidates if _camera_slot(key) == CANONICAL_EXTERIOR_KEY]
            if not exterior:
                return None
            return min(
                exterior, key=lambda key: (_exterior_priority(_normalize_camera_name(key)), key)
            )

        wrist = [key for key in candidates if _camera_slot(key) == canonical_key]
        if not wrist:
            return None
        return sorted(wrist)[0]

    @staticmethod
    def _dummy_image(meta: Dict[str, Any]) -> torch.Tensor:
        shape = meta.get("shape") or meta.get("raw_shape") or [3, 224, 224]
        if len(shape) != 3:
            shape = [3, 224, 224]
        c, h, w = [int(v) for v in shape]
        return torch.zeros(1, c, h, w, dtype=torch.uint8)

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        raw_key = self._select_raw_image_key(meta["key"], lerobot_sample)
        if raw_key is None:
            return self._dummy_image(meta)
        image: torch.Tensor = lerobot_sample[raw_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return (image * 255).to(torch.uint8)

    @staticmethod
    def _image_pad_mask(sample: Dict[str, Any], obs_size: int) -> torch.Tensor:
        for value in sample.get("images", {}).values():
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return torch.zeros(int(value.shape[0]), dtype=torch.bool)
        return torch.zeros(int(obs_size), dtype=torch.bool)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        requested_idx = idx
        overfit_active = hasattr(self, "_overfit_indices")
        sample_idx = int(self._overfit_indices[idx]) if overfit_active else idx + self._start_idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                if self.is_training_set and not lerobot_sample.get("step_is_qualified", True):
                    if overfit_active:
                        location = self._locate_sample(sample_idx)
                        raise RuntimeError(
                            "Overfit sample became unqualified after selection: "
                            f"index={sample_idx} ({location})"
                        )
                    self._record_invalid_sample(
                        stage="unqualified",
                        requested_idx=requested_idx,
                        sample_idx=sample_idx,
                    )
                    attempt += 1
                    sample_idx = self._resample_random_idx()
                    continue
                break
            except Exception as err:
                if overfit_active:
                    location = self._locate_sample(sample_idx)
                    raise RuntimeError(
                        f"Failed to load preselected overfit sample {sample_idx} ({location})."
                    ) from err
                attempt += 1
                last_exception = err
                self._record_invalid_sample(
                    stage="load_error",
                    requested_idx=requested_idx,
                    sample_idx=sample_idx,
                    error=err,
                )
                location = self._locate_sample(sample_idx)
                logger.warning(
                    f"Error loading sample {sample_idx} ({location}) "
                    f"(attempt {attempt}). Retrying with a random index. Error: {err}"
                )
                sample_idx = self._resample_random_idx()
        else:
            last_location = self._locate_sample(sample_idx)
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {requested_idx}. Last sample: {sample_idx} ({last_location})."
            ) from last_exception

        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                sample = {
                    "idx": sample_idx,
                    "task": lerobot_sample["task"],
                    "dataset_locator": self._locate_sample(sample_idx),
                    "action": {},
                    "state": {},
                    "images": {},
                    "frequency": self.model_fps,
                }

                state_is_pad = None
                for meta in self.state_meta:
                    sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
                    cur_state_is_pad = self._get_pad_mask(meta, lerobot_sample)
                    state_is_pad = (
                        cur_state_is_pad
                        if state_is_pad is None
                        else (state_is_pad | cur_state_is_pad)
                    )

                action_is_pad = None
                for meta in self.action_meta:
                    sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
                    cur_action_is_pad = self._get_pad_mask(meta, lerobot_sample)
                    action_is_pad = (
                        cur_action_is_pad
                        if action_is_pad is None
                        else (action_is_pad | cur_action_is_pad)
                    )

                if self.image_meta:
                    meta_by_key = {meta["key"]: meta for meta in self.image_meta}
                    for meta in self.image_meta:
                        sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

                    has_real = {
                        meta["key"]: self._select_raw_image_key(meta["key"], lerobot_sample)
                        is not None
                        for meta in self.image_meta
                    }
                    real_keys = [key for key, value in has_real.items() if value]

                    if self.random_drop_camera and self.is_training_set:
                        ep_idx = lerobot_sample.get("episode_index", 0)
                        ep_idx = int(ep_idx.item()) if hasattr(ep_idx, "item") else int(ep_idx)

                        if len(real_keys) > 2:
                            exterior_real = [
                                key
                                for key in real_keys
                                if meta_by_key[key].get("camera_type", key)
                                == CANONICAL_EXTERIOR_KEY
                            ]
                            wrist_real = [key for key in real_keys if key not in exterior_real]

                            rng_drop = np.random.default_rng(ep_idx)
                            if len(exterior_real) > 1 and wrist_real:
                                drop_key = exterior_real[int(rng_drop.integers(len(exterior_real)))]
                            elif len(wrist_real) > 1 and exterior_real:
                                drop_key = wrist_real[int(rng_drop.integers(len(wrist_real)))]
                            else:
                                drop_key = real_keys[int(rng_drop.integers(len(real_keys)))]

                            sample["images"][drop_key] = self._dummy_image(meta_by_key[drop_key])
                            real_keys = [key for key in real_keys if key != drop_key]

                        real_images = [sample["images"][key] for key in real_keys]

                        for meta in self.image_meta:
                            sample["images"][meta["key"]] = self._dummy_image(meta)

                        if len(real_images) >= 2:
                            sample["images"][CANONICAL_EXTERIOR_KEY] = real_images[0]
                            sample["images"][CANONICAL_WRIST_RIGHT_KEY] = real_images[1]
                        elif len(real_images) == 1:
                            rng_slot = np.random.default_rng(ep_idx ^ 0x1234)
                            if rng_slot.integers(2):
                                sample["images"][CANONICAL_WRIST_RIGHT_KEY] = real_images[0]
                            else:
                                sample["images"][CANONICAL_EXTERIOR_KEY] = real_images[0]

                        if len(real_images) >= 2:
                            rng_swap = np.random.default_rng(ep_idx ^ 0xABCD)
                            if rng_swap.integers(2):
                                (
                                    sample["images"][CANONICAL_EXTERIOR_KEY],
                                    sample["images"][CANONICAL_WRIST_RIGHT_KEY],
                                ) = (
                                    sample["images"][CANONICAL_WRIST_RIGHT_KEY],
                                    sample["images"][CANONICAL_EXTERIOR_KEY],
                                )
                    sample["image_is_pad"] = self._image_pad_mask(sample, self.obs_size)
                self._inject_dummy_images(sample)

                sample["action_is_pad"] = action_is_pad
                sample["state_is_pad"] = state_is_pad

                if (
                    "chunked_task_index" in lerobot_sample
                    and lerobot_sample["chunked_task_index"] is not None
                ):
                    chunked_task_index = lerobot_sample["chunked_task_index"]
                    mask = chunked_task_index != chunked_task_index[0]
                    assert mask.shape == sample["action_is_pad"].shape, (
                        f"Mask shape {mask.shape} does not match action_is_pad shape "
                        f"{sample['action_is_pad'].shape}"
                    )
                    sample["action_is_pad"] = sample["action_is_pad"] | mask
                    if mask.any():
                        last_valid = (~mask).nonzero(as_tuple=False)[-1].item()
                        for meta in self.action_meta:
                            sample["action"][meta["key"]][mask] = sample["action"][meta["key"]][
                                last_valid
                            ].clone()

                sample = self._get_additional_data(sample, lerobot_sample)

                for key in lerobot_sample:
                    if key not in sample and "observation" not in key and "action" not in key:
                        sample[key] = lerobot_sample[key]

                if self.processor is not None:
                    sample = self.processor.preprocess(sample)

                return sample

            except Exception as err:
                if overfit_active:
                    location = self._locate_sample(sample_idx)
                    raise RuntimeError(
                        f"Failed to build preselected overfit sample {sample_idx} ({location})."
                    ) from err

                attempt += 1
                last_exception = err
                self._record_invalid_sample(
                    stage="build_error",
                    requested_idx=requested_idx,
                    sample_idx=sample_idx,
                    error=err,
                )
                location = self._locate_sample(sample_idx)
                logger.warning(
                    f"Error building sample {sample_idx} ({location}) "
                    f"(attempt {attempt}). Retrying with a random index. Error: {err}"
                )
                sample_idx = self._resample_random_idx()
                lerobot_sample = self.multi_dataset[sample_idx]

        last_location = self._locate_sample(sample_idx)
        raise RuntimeError(
            f"Failed to build a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
            f"for index {requested_idx}. Last sample: {sample_idx} ({last_location})."
        ) from last_exception

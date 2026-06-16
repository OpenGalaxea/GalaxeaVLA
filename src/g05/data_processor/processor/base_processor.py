# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal, Tuple
import logging

from omegaconf import OmegaConf
import torch
import numpy as np
from copy import deepcopy
from g05.utils.data.normalizer import LinearNormalizer, NormMode
from g05.utils.common.pytorch_utils import dict_apply

from g05.data_processor.transforms.action_filter import BaseActionFilter

logger = logging.getLogger(__name__)


def _warn_nan_in_batch(data: Dict[str, Any], stage: str) -> bool:
    """
    Scan action/state dicts for NaN or Inf values and emit a warning.

    Returns True if any NaN/Inf was found.
    """
    found = False
    embodiment = data.get("embodiment", "unknown")
    dataset_locator = data.get("dataset_locator", "unknown")
    idx = data.get("idx", "?")
    for split in ("action", "state"):
        if split not in data or not isinstance(data[split], dict):
            continue
        for key, val in data[split].items():
            if not isinstance(val, torch.Tensor):
                continue
            n_nan = int(torch.isnan(val).sum())
            n_inf = int(torch.isinf(val).sum())
            if n_nan > 0 or n_inf > 0:
                logger.warning(
                    "[NaN/Inf] stage=%s split=%s key=%s nan=%d inf=%d | "
                    "embodiment=%s dataset=%s idx=%s",
                    stage,
                    split,
                    key,
                    n_nan,
                    n_inf,
                    embodiment,
                    dataset_locator,
                    idx,
                )
                found = True
    return found


IGNORE_INDEX = -100

_FORBIDDEN_META_KEYS = {
    "source",
    "target_key",
    "target_offset",
    "target_from",
    "semantic_key",
    "resolved_lerobot_key",
    "resolved_start_index",
}


class BaseProcessor(ABC):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,
        # instruction transform
        drop_high_level_prob: float,
        use_zh_instruction: bool,
        # camera size config: {camera_type: [H, W]}; resolves shape and injects T.Resize per camera
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # optional: per-key norm override (None = use norm_default_mode for all keys)
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # vlm input action normalization (independent from training action norm)
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy mode clip range (applied when norm_mode is "dummy")
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        # tail norm scale (used when norm_mode is "z-score-tail" or "q01/q99-tail")
        norm_tail_scale: float = 0.075,
    ):
        # Convert to plain Python dict so downstream mutations (e.g. _resolve_image_shapes)
        # work unconditionally — OmegaConf struct mode blocks new-key assignment.
        from omegaconf import OmegaConf, DictConfig

        if isinstance(shape_meta, DictConfig):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)

        # Shape meta must already be fully explicit; do not silently drop placeholders.
        processed_action_meta, processed_state_meta = [], []
        for meta in shape_meta["action"]:
            key = meta.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"action meta key must be a non-empty string, got {key!r}.")
            unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
            if unexpected:
                raise ValueError(
                    f"action meta for key={key!r} contains forbidden fields: {unexpected}."
                )
            processed_action_meta.append(meta)
        for meta in shape_meta["state"]:
            key = meta.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"state meta key must be a non-empty string, got {key!r}.")
            unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
            if unexpected:
                raise ValueError(
                    f"state meta for key={key!r} contains forbidden fields: {unexpected}."
                )
            processed_state_meta.append(meta)
        shape_meta["action"] = processed_action_meta
        shape_meta["state"] = processed_state_meta

        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # Resolve image shapes and inject per-camera T.Resize from camera_size_config.
        self._resolve_image_shapes(self.shape_meta, camera_size_config)
        camera_key_to_size = {
            img_meta["key"]: img_meta["shape"][1:]
            for img_meta in (self.shape_meta.get("images") or [])
            if img_meta.get("shape") is not None
        }
        # Any image_meta key without an entry in train/val transforms gets an
        # empty list — tolerates dataset-side zero-injected slots (e.g. DROID
        # dummy_wrist_right) without forcing the embodiment yaml to repeat
        # transform stacks for them.
        self.train_transforms = self._ensure_transforms_for_keys(
            self.train_transforms, camera_key_to_size.keys()
        )
        self.val_transforms = self._ensure_transforms_for_keys(
            self.val_transforms, camera_key_to_size.keys()
        )
        self.train_transforms = self._patch_resize_per_camera(
            self.train_transforms, camera_key_to_size
        )
        self.val_transforms = self._patch_resize_per_camera(self.val_transforms, camera_key_to_size)

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        # BehaviorPerKeyTransform (B1K only) merges/deletes keys in the transform.
        # B1K proprio fields are not contiguous: trunk_qpos[236:240] and
        # base_qvel[253:256] must be read as separate shape_meta keys, then
        # merged in the transform as lower_body(7D) = trunk_qpos(4) + base_qvel(3).
        # The gripper is similar: proprio stores 2D two-finger qpos that must
        # collapse to 1D width.
        # When this flag is True, post-transform shape asserts skip keys already
        # consumed by merging. Other embodiments (r1lite, etc.) do not change key
        # structure, so flag=False keeps assertions strict and preserves behavior.
        self._transforms_alter_keys = any(
            getattr(t, "alters_key_structure", False) for t in (action_state_transforms or [])
        )
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.action_filter = action_filter
        self.action_filter.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.vlm_input_action_norm_default_mode = vlm_input_action_norm_default_mode
        self.vlm_input_action_norm_exception_mode = vlm_input_action_norm_exception_mode
        self._vlm_input_action_normalizer = None

        self.dummy_clip_default = dummy_clip_default
        self.dummy_clip_exception = dummy_clip_exception
        self.norm_tail_scale = norm_tail_scale

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError(
                "normalizer has not been set. Please call set_normalizer_from_stats() first."
            )
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            dummy_clip_default=self.dummy_clip_default,
            dummy_clip_exception=self.dummy_clip_exception,
            missing_key_mode="warn",
            tail_scale=self.norm_tail_scale,
        )

        if self.vlm_input_action_norm_default_mode is not None:
            self._vlm_input_action_normalizer = LinearNormalizer(
                use_stepwise_action_norm=self.use_stepwise_action_norm,
                shape_meta=self.shape_meta,
                default_mode=self.vlm_input_action_norm_default_mode,
                exception_mode=self.vlm_input_action_norm_exception_mode
                if self.vlm_input_action_norm_exception_mode is not None
                else self.norm_exception_mode,
                stats=dataset_stats,
                dummy_clip_default=self.dummy_clip_default,
                dummy_clip_exception=self.dummy_clip_exception,
                missing_key_mode="warn",
                tail_scale=self.norm_tail_scale,
            )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        Args:
            data: Dict[str, str] | List[str], lerobot sample in raw mcap

        Returns:
            List[str], processed instructions
        """
        # if single instruction, convert to list
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"[Low]: {low_level_instruction}"
        else:
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"

        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, (
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                )

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, (
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape{meta_shape}."
            )

        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)

        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                if self._transforms_alter_keys and k not in batch["action"]:
                    continue
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, (
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
                )

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            if self._transforms_alter_keys and k not in batch["state"]:
                continue
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, (
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
            )

        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index

        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}

        # 2. image
        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            assert image.ndim == 4, (
                f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            )

            # Apply transforms efficiently on the merged batch
            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            meta_shape = tuple([self.num_obs_steps] + shape)
            assert image.shape == meta_shape, (
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"
            )

            processed_images.append(image)

        pixel_values = torch.cat(processed_images, dim=0)  # [num_input_cameras, C, H, W]
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros(
                (self.num_output_cameras,) + pixel_values.shape[1:],
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            out[0 : pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        else:
            sample["pixel_values"] = pixel_values

        # Copy action before transform for open-loop evaluation
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # 3. action & state
        data = self.action_filter.forward(data)
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"]  # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"]  # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]  # [action_dim,]
            if "action_op_mask" in data:
                sample["action_op_mask"] = data["action_op_mask"]

        sample["proprio"] = data["state"]  # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"]  # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["proprio_dim_is_pad"]  # [proprio_dim,]

        sample["idx"] = data["idx"]
        for meta_key in ("task", "embodiment", "dataset_locator"):
            if meta_key in data:
                sample[meta_key] = data[meta_key]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.

        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data

    @staticmethod
    def _resolve_image_shapes(
        shape_meta: Dict, camera_size_config: Optional[Dict[str, List[int]]] = None
    ) -> None:
        """Resolve [C, H, W] in place for every entry in shape_meta.images.

        Resolution priority (camera_size_config is the single source of truth):
          1. If camera_type + camera_size_config[cam_type] exists, use
             csc[cam_type] as (H, W). If img_meta.shape is explicitly set but
             disagrees with csc, csc silently overrides it and emits a warning.
             C comes from explicit shape or raw_shape[0], defaulting to 3.
          2. If csc is missing but img_meta.shape is explicitly set, use shape as
             the fallback.
          3. If both are missing, raise ValueError.

        Note: this mutates img_meta in place. Reusing the same shape_meta is
        idempotent because shape has already been rewritten to the csc value, so
        the csc override no longer emits a warning.
        """
        images = shape_meta.get("images", [])
        for img_meta in images:
            key = img_meta.get("key")
            cam_type = img_meta.get("camera_type")
            explicit_shape = img_meta.get("shape")

            # Prefer csc override.
            if cam_type and camera_size_config and cam_type in camera_size_config:
                H, W = camera_size_config[cam_type]
                if explicit_shape is not None:
                    C = explicit_shape[0]
                    if list(explicit_shape[-2:]) != [H, W]:
                        logger.warning(
                            f"[shape_meta] Image '{key}' shape {list(explicit_shape)} "
                            f"is overridden by camera_size_config[{cam_type}]={[H, W]} -> "
                            f"[{C}, {H}, {W}]. The shape field in the embodiment yaml is stale; "
                            f"consider removing it."
                        )
                else:
                    raw_shape = img_meta.get("raw_shape", [3])
                    C = raw_shape[0] if hasattr(raw_shape, "__len__") else 3
                img_meta["shape"] = [C, H, W]
                continue

            # No csc: fall back to explicit shape.
            if explicit_shape is not None:
                continue

            if cam_type is None:
                raise ValueError(
                    f"Image '{key}' needs either explicit 'shape: [C, H, W]' "
                    f"or 'camera_type' + processor.camera_size_config to resolve shape."
                )
            if camera_size_config is None or cam_type not in camera_size_config:
                available = list(camera_size_config.keys()) if camera_size_config else []
                raise ValueError(
                    f"Image '{key}' has 'camera_type: {cam_type}' but no matching "
                    f"camera_size_config entry. Either add 'shape: [C, H, W]' to the "
                    f"image config, or set 'processor.camera_size_config.{cam_type}: [H, W]'."
                    + (f" (available types: {available})" if available else "")
                )

            H, W = camera_size_config[cam_type]
            raw_shape = img_meta.get("raw_shape", [3])
            C = raw_shape[0] if hasattr(raw_shape, "__len__") else 3
            img_meta["shape"] = [C, H, W]

    @staticmethod
    def _ensure_transforms_for_keys(transforms_dict, keys):
        """Return a transforms_dict with an empty list for every key not yet present.

        Used so dataset-side zero-injected image slots (declared in shape_meta.images
        but absent from the embodiment's train/val transforms) don't trip a KeyError
        in process_images. The downstream T.Resize patch still runs on every key.
        """
        if transforms_dict is None:
            return None
        result = dict(transforms_dict)
        for key in keys:
            result.setdefault(key, [])
        return result

    @staticmethod
    def _patch_resize_per_camera(transforms_dict, camera_key_to_size):
        """Prepend T.Resize to each camera's transform list based on camera_key_to_size.

        Args:
            transforms_dict: {camera_key: [transform, ...]}
            camera_key_to_size: {camera_key: [H, W]} — only keys present here are patched.

        Assumes embodiment configs do NOT define T.Resize when camera_size_config is used.
        """
        import torchvision.transforms as T

        if transforms_dict is None:
            return None
        result = {}
        for key, ts in transforms_dict.items():
            if key not in camera_key_to_size:
                result[key] = ts
                continue
            result[key] = [T.Resize(list(camera_key_to_size[key]))] + list(ts)
        return result


class ActionProcessor(BaseProcessor):
    """
    Processor that handles action/state processing without VLM inputs (images, tokenization).

    Provides:
    - action_filter: Filters action dimensions based on movement detection
    - action_state_transform: Transforms action/state (e.g., relative joint)
    - normalizer: Normalizes action/state values
    - action_state_merger: Merges multiple action/state keys into single tensors

    Use this processor when you only need action/state processing (e.g., action tokenizer training).
    """

    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # image transform (optional, not used in preprocess)
        train_transforms: Dict[str, List[Any]] | None = None,
        val_transforms: Dict[str, List[Any]] | None = None,
        # camera size config (optional, forwarded to BaseProcessor)
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # instruction transform (optional, not used in preprocess)
        drop_high_level_prob: float = 0.0,
        use_zh_instruction: bool = False,
        # tokenization (optional, for MixtureProcessor compatibility)
        pad_token_id: int = 0,
        # optional: per-key norm override (None = use norm_default_mode for all keys)
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # vlm input action normalization (independent from training action norm)
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy mode clip range (applied when norm_mode is "dummy")
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        # tail norm scale (used when norm_mode is "z-score-tail" or "q01/q99-tail")
        norm_tail_scale: float = 0.075,
    ):
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps,
            num_output_cameras=num_output_cameras,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            action_filter=action_filter,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            camera_size_config=camera_size_config,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
            norm_tail_scale=norm_tail_scale,
        )

        # Required by MixtureProcessor
        self.pad_token_id = pad_token_id

    def preprocess_action_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process action and state through the pipeline:
        action_filter -> action_state_transform -> normalizer -> action_state_merger

        Returns updated data dict with processed action/state.
        """
        data = self.action_filter.forward(data)
        data = self.action_state_transform(data)
        # Detect NaN/Inf introduced by rotation transforms (e.g. quaternion_to_matrix,
        # matrix_to_euler_angles) before the normalizer silently passes them through.
        _warn_nan_in_batch(data, stage="post_transform")

        # If vlm input action normalizer is configured, normalize a separate copy for VLM input
        vlm_action = None
        if self._vlm_input_action_normalizer is not None and "action" in data:
            vlm_action_data = {"action": deepcopy(data["action"])}
            self._vlm_input_action_normalizer.forward(vlm_action_data)
            vlm_action_data = self.action_state_merger.forward(
                {
                    "action": vlm_action_data["action"],
                    "action_is_pad": data["action_is_pad"],
                }
            )
            vlm_action = vlm_action_data["action"]

        data = self.normalizer.forward(data)
        # Detect any NaN/Inf that survived normalizer (should be 0 after nan_to_num, but log anyway).
        _warn_nan_in_batch(data, stage="post_norm")
        data = self.action_state_merger.forward(data)

        if vlm_action is not None:
            data["vlm_action"] = vlm_action

        return data

    def build_action_sample(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build action-related sample fields from processed data.

        Call this after preprocess_action_state().
        """
        if "action" in data:
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            if getattr(self.action_state_merger, "max_action_shape_meta", None) is not None:
                sample["action_parts_meta"] = dict(self.action_state_merger.max_action_shape_meta)
            if "action_op_mask" in data:
                sample["action_op_mask"] = data["action_op_mask"]

        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["proprio_dim_is_pad"]

        if "frequency" in data:
            sample["frequency"] = data["frequency"]

        return sample

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model (action/state only, no VLM inputs).
        """
        sample = {}

        # Process action and state through transforms first
        data = self.action_filter.forward(data)
        data = self.action_state_transform(data)

        # Copy action after transform (but before normalizer) for tokenizer evaluation
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # Continue with normalizer
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        sample = self.build_action_sample(data, sample)

        sample["idx"] = data["idx"]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model (reverse action/state processing).

        Supports dual-path (AR + FM) inference:
        - action_fm: FM path action (z-score space) → denormalize with self.normalizer
        - action_ar: AR path action (dummy space) → no denormalization needed
        - action: backward-compatible key, resolves to action_fm > action_ar
        """
        import logging

        logger = logging.getLogger(__name__)

        def _action_stats(action_val):
            """Get min/max stats for action, handling dict (per-part) format."""
            if isinstance(action_val, dict):
                all_tensors = [
                    v.flatten() for v in action_val.values() if isinstance(v, torch.Tensor)
                ]
                if all_tensors:
                    return f"dict with {list(action_val.keys())}"
                return "dict (no tensors)"
            elif isinstance(action_val, torch.Tensor):
                return f"min={action_val.min().item():.4f}, max={action_val.max().item():.4f}"
            return str(type(action_val))

        has_fm = "action_fm" in data
        has_ar = "action_ar" in data

        logger.debug(f"[postprocess] has_fm={has_fm}, has_ar={has_ar}")
        logger.debug(
            f"[postprocess] processor normalizer: norm_default_mode={getattr(self, 'norm_default_mode', 'N/A')}"
        )
        logger.debug(
            f"[postprocess] processor vlm_input_action_norm_default_mode={getattr(self, 'vlm_input_action_norm_default_mode', 'N/A')}"
        )

        if has_fm:
            logger.debug(f"[postprocess] BEFORE FM denorm: {_action_stats(data['action_fm'])}")
            fm_data = {
                "action": data["action_fm"],
                "state": data.get("state", data.get("proprio")),
            }
            fm_data = self.action_state_merger.backward(fm_data)
            fm_data = self.normalizer.backward(fm_data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    fm_data = trans.backward(fm_data)
            fm_data = self.action_filter.backward(fm_data)
            start_obs_step = self.num_obs_steps - 1
            data["action_fm"] = dict_apply(fm_data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER FM denorm: {_action_stats(data['action_fm'])}")

        if has_ar:
            logger.debug(f"[postprocess] BEFORE AR denorm: {_action_stats(data['action_ar'])}")
            ar_data = {
                "action": data["action_ar"],
                "state": data.get("state", data.get("proprio")),
            }
            ar_data = self.action_state_merger.backward(ar_data)
            ar_data = self.normalizer.backward(ar_data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    ar_data = trans.backward(ar_data)
            ar_data = self.action_filter.backward(ar_data)
            start_obs_step = self.num_obs_steps - 1
            data["action_ar"] = dict_apply(ar_data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER AR denorm: {_action_stats(data['action_ar'])}")

        if not has_fm and not has_ar:
            assert "action" in data, "Action is required in postprocess"
            logger.debug(f"[postprocess] Fallback (legacy): {_action_stats(data['action'])}")
            data["state"] = data.pop("proprio")
            data = self.action_state_merger.backward(data)
            data = self.normalizer.backward(data)
            if self.action_state_transforms is not None:
                for trans in reversed(self.action_state_transforms):
                    data = trans.backward(data)
            data = self.action_filter.backward(data)
            start_obs_step = self.num_obs_steps - 1
            data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
            logger.debug(f"[postprocess] AFTER fallback denorm: {_action_stats(data['action'])}")
            return data

        data["action"] = data["action_fm"] if has_fm else data["action_ar"]
        logger.debug(f"[postprocess] Final 'action' key set: {_action_stats(data['action'])}")
        return data


class FullProcessor(ActionProcessor):
    """
    Processor that handles full VLM inputs including images and tokenization.

    Extends ActionProcessor with:
    - Image processing with transforms
    - Tokenization (PaliGemma style by default)
    - VLM input action normalization

    Use this processor for VLM training/inference with images and text instructions.
    """

    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,
        # instruction transform
        drop_high_level_prob: float,
        use_zh_instruction: bool,
        # tokenization
        pad_token_id: int,
        image_token_index: int,
        max_text_tokens: int,
        num_input_cameras: int,
        # camera size config: {camera_type: [H, W]}; used to auto-resolve shape from camera_type
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        tokenizer_params: Optional[Dict[str, Any]] = None,
        # optional: per-key norm override (None = use norm_default_mode for all keys)
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # vlm input action normalization (independent from training action norm)
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # dummy mode clip range (applied when norm_mode is "dummy")
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ):
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps,
            num_output_cameras=num_output_cameras,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            action_filter=action_filter,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            camera_size_config=camera_size_config,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
        )

        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.tokenizer = (
            self._load_tokenizer(tokenizer_params) if tokenizer_params is not None else None
        )
        self.max_text_tokens = max_text_tokens
        self.num_input_images = num_input_cameras

    def _load_tokenizer(self, tokenizer_params):
        """Load tokenizer. Subclasses may override for custom loaders (e.g. Qwen3.5)."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(**tokenizer_params)
        return processor.tokenizer

    def process_images(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process images through transforms.

        Returns:
            pixel_values: Dict[str, Tensor[num_obs_steps, C, H_k, W_k]]
                Each camera key maps to its own tensor. Different cameras may have
                different (H, W) — multi-resolution is supported.
        """
        result: Dict[str, torch.Tensor] = {}
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            assert image.ndim == 4, (
                f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            )

            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            meta_shape = tuple([self.num_obs_steps] + shape)
            assert image.shape == meta_shape, (
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"
            )

            result[meta["camera_type"]] = image  # [num_obs_steps, C, H_k, W_k]

        return result

    def build_pixel_values(self, pixel_values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adjust pixel_values dict to num_output_cameras.

        Multi-resolution case (cameras have different H/W):
            Requires total obs frames == num_output_cameras; returns dict as-is.
        Same-resolution case (all cameras share H, W):
            Supports zero-padding or first-N/random selection to match num_output_cameras.
        """
        n = sum(v.shape[0] for v in pixel_values.values())
        if n == self.num_output_cameras:
            return pixel_values

        # Check if all cameras share the same spatial shape (same-resolution)
        shapes = [tuple(v.shape[1:]) for v in pixel_values.values()]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"Multi-resolution cameras require n_frames == num_output_cameras "
                f"({n} != {self.num_output_cameras}). Padding/selection not supported "
                "across cameras of different resolutions."
            )
        else:
            import logging

            logging.warning(
                f"All cameras share the same resolution {shapes[0]}. "
                f"Applying padding/selection to match num_output_cameras={self.num_output_cameras}."
            )

        # Same-resolution: flatten to [n, C, H, W], apply selection/padding, re-split
        keys = list(pixel_values.keys())
        flat = torch.cat(list(pixel_values.values()), dim=0)  # [n, C, H, W]
        if n < self.num_output_cameras:
            out = torch.zeros(
                (self.num_output_cameras,) + flat.shape[1:],
                device=flat.device,
                dtype=flat.dtype,
            )
            out[:n] = flat
            flat = out
        else:  # n > num_output_cameras
            if self.is_train:
                indices = torch.randperm(n, device=flat.device)[: self.num_output_cameras]
                indices = indices.sort().values
            else:
                indices = torch.arange(self.num_output_cameras, device=flat.device)
            flat = flat[indices]

        # Re-wrap: distribute frames uniformly back to camera keys
        n_out = flat.shape[0]
        frames_per_key = n_out // len(keys)
        return {k: flat[i * frames_per_key : (i + 1) * frames_per_key] for i, k in enumerate(keys)}

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index

        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": Dict[str, Tensor[num_obs_steps, C, H_k, W_k]]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}

        # 2. image
        pixel_values = self.process_images(data)
        sample["pixel_values"] = self.build_pixel_values(pixel_values)

        # Copy action before transform for open-loop evaluation
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # 3. action & state
        data = self.preprocess_action_state(data)
        sample = self.build_action_sample(data, sample)

        sample["idx"] = data["idx"]

        return sample

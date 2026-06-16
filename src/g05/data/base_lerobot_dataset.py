import bisect
import multiprocessing as mp
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Union

import numpy as np
import torch
from tqdm import tqdm

from g05.data_processor.processor.base_processor import BaseProcessor
from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 10000
_FORBIDDEN_META_KEYS = {
    "source",
    "target_key",
    "target_offset",
    "target_from",
    "semantic_key",
    "resolved_lerobot_key",
    "resolved_start_index",
}


def _to_plain(obj):
    """Recursively convert OmegaConf containers to plain Python objects."""
    if obj is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


def shift_sequence_with_replication(data: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Shift a single-episode sequence by `offset`.

    Args:
        data: Episode-level tensor with shape `(T, D)` or `(T,)`.
        offset: Temporal offset applied to the first dimension. Positive values
            mean "look into the future"; negative values mean "look into the past".

    Returns:
        Tensor with the same shape as `data`. Indices that would cross the
        episode boundary are clamped to the first/last valid frame, so this
        function never crosses episodes and naturally implements the padding
        policy used for target construction.
    """
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    length = data.shape[0]
    indices = torch.arange(length, device=data.device) + offset
    indices = indices.clamp_(0, length - 1)
    return data[indices]


class BaseLerobotDataset(torch.utils.data.Dataset):
    _multi_dataset_cache = {}
    _shared_manager = None
    _cache_hit_count = 0  # tracks how many datasets were reused from cache (for summary log)

    @staticmethod
    def _make_cache_key(dataset_dirs, delta_timestamps, tolerances_s, load_images, in_memory=False):
        ds_key = tuple(sorted(dataset_dirs))
        dt_key = tuple(sorted((k, tuple(v)) for k, v in delta_timestamps.items()))
        tol_key = tuple(sorted(tolerances_s.items()))
        return (ds_key, dt_key, tol_key, bool(load_images), bool(in_memory))

    @classmethod
    def clear_cache(cls):
        cls._multi_dataset_cache.clear()

    @classmethod
    def _get_shared_manager(cls):
        if cls._shared_manager is None:
            cls._shared_manager = mp.Manager()
        return cls._shared_manager

    def __init__(
        self,
        dataset_dirs: List[str],
        # shapes
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,  # Excludes the current frame.
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        # lerobot_ds_version
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "2.1",
        # tolerance
        tolerance_s: Optional[float] = None,
        # fps override
        override_fps: Optional[int] = None,
        load_images: Optional[bool] = None,
        in_memory: bool = False,
        **kwargs,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = int(obs_size)
        assert self.obs_size >= 1, f"obs_size must be >= 1, got {self.obs_size}"
        self.obs_stride_second = float(obs_stride_second)
        self.processor = None  # Will be set externally
        manager = self._get_shared_manager()
        self._invalid_sample_records = manager.dict()
        self._invalid_sample_order = manager.list()
        metas = []
        self.lerobot_ds_version = lerobot_ds_version
        if lerobot_ds_version == "2.1":
            from g05.data.lerobot.lerobot_dataset import (
                LeRobotDatasetMetadata,
                MultiLeRobotDataset,
            )

            logger.warning(
                f"[lerobot_ds_version=2.1] {dataset_dirs[0]} "
                f"(+{len(dataset_dirs) - 1} more dirs) — using legacy v2 dataset path"
            )
        elif lerobot_ds_version == "3.0":
            from g05.data.lerobot.lerobot_dataset_v3 import (
                LeRobotDatasetMetadata,
                MultiLeRobotDataset,
            )
        else:
            raise ValueError(f"Unsupported lerobot_ds_version: {lerobot_ds_version}")

        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)
        try:
            # data_fps: actual recording frequency read from meta/info.json.
            # All data-read fields, including delta_timestamps and tolerance_s, use it.
            data_fps = meta.fps
        except Exception as e:
            logger.warning(f"Failed to read fps from dataset meta, falling back to 15: {e}")
            data_fps = 15
        self.fps = data_fps

        # obs_stride: convert time-based stride (seconds) to step count using actual fps
        if self.obs_stride_second > 0:
            self.obs_stride = max(1, round(self.obs_stride_second * data_fps))
        else:
            self.obs_stride = 1

        if self.obs_size > 1:
            history_sec = (self.obs_size - 1) * self.obs_stride / data_fps
            logger.info(
                f"[obs] {dataset_dirs[0]}: obs_size={self.obs_size}, "
                f"obs_stride_second={self.obs_stride_second}s → stride_steps={self.obs_stride} "
                f"(fps={data_fps}), history_window={history_sec:.1f}s"
            )

        # override_fps: model control frequency, i.e. the FPS at which the model runs.
        # It is unrelated to data reads and is only passed to the action tokenizer
        # through the "frequency" field. Defaults to data_fps when unset.
        self.model_fps = override_fps if override_fps is not None else data_fps
        if override_fps is not None:
            logger.info(f"model_fps={self.model_fps} (override), data_fps={data_fps} (from meta)")

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        # Convert meta lists to plain Python so OmegaConf ListConfig/DictConfig
        # don't leak into downstream isinstance(x, (list, tuple)) checks.
        # Filter out dummy cameras: they have no lerobot_key in YAML, so the base
        # class neither validates nor loads them. Entries with dummy: true are
        # injected as zero tensors by _inject_dummy_images in __getitem__.
        all_image_meta = _to_plain(shape_meta.get("images", None)) or []
        self._dummy_image_meta = [m for m in all_image_meta if m.get("dummy", False)]
        self.image_meta = [m for m in all_image_meta if not m.get("dummy", False)] or None
        self.state_meta = _to_plain(shape_meta["state"])
        self.action_meta = _to_plain(shape_meta["action"])
        default_load_images = bool(self.image_meta)
        self.load_images = default_load_images if load_images is None else bool(load_images)
        self.load_images = self.load_images and default_load_images
        self.return_images = self.load_images
        self.in_memory = in_memory
        if in_memory:
            logger.info(
                f"[in_memory] BaseLerobotDataset: in_memory=True, "
                f"version={lerobot_ds_version}, dirs={len(dataset_dirs)}, "
                f"first_dir={dataset_dirs[0]}"
            )

        # Validate explicit raw-layout fields provided by config.
        self._setup_meta_lerobot_keys()

        # Build delta_timestamps for querying frames from parquet.
        # data_fps is the actual recording frequency and determines the interval
        # between adjacent query points: 1/data_fps seconds per step.
        delta_timestamps = self._build_delta_timestamps(data_fps, past_action_size, action_size)

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                if self.is_training_set:
                    episodes.update({meta.repo_id: list(range(split_idx))})
                else:
                    episodes.update({meta.repo_id: list(range(split_idx, meta.total_episodes))})

        if lerobot_ds_version == "3.0":
            episodes = None

        import time

        start_time = time.time()
        # tolerance_s: parquet timestamp matching tolerance, based on the actual
        # recording frequency with frame spacing 1/data_fps.
        # 0.4/data_fps is about 13.3 ms at 30 fps, accommodating 0.6-4 ms timestamp
        # drift caused by ffmpeg H.264 re-encoding.
        self.tolerance_s = tolerance_s if tolerance_s is not None else 0.4 / data_fps
        logger.info(
            f"tolerance_s={'user-specified' if tolerance_s is not None else 'auto (0.4/data_fps)'}: {self.tolerance_s:.6f}s (data_fps={data_fps}, model_fps={self.model_fps})"
        )
        tolerances_s = dict.fromkeys(self.dataset_dirs, self.tolerance_s)
        cache_key = self._make_cache_key(
            self.dataset_dirs,
            delta_timestamps,
            tolerances_s,
            self.load_images,
            self.in_memory,
        )
        if cache_key in BaseLerobotDataset._multi_dataset_cache and lerobot_ds_version == "3.0":
            self.multi_dataset = BaseLerobotDataset._multi_dataset_cache[cache_key]
            BaseLerobotDataset._cache_hit_count += 1
            logger.debug(f"Reusing cached MultiLeRobotDataset (cache hit)")
        else:
            self.multi_dataset = MultiLeRobotDataset(
                dataset_dirs=self.dataset_dirs,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                tolerances_s=tolerances_s,
                load_images=self.load_images,
                in_memory=self.in_memory,
            )
            BaseLerobotDataset._multi_dataset_cache[cache_key] = self.multi_dataset
            logger.debug(f"MultiLeRobotDataset initialized in {time.time() - start_time:.2f}s")
        self.multi_dataset.set_load_images(self.load_images)

        # Build cumulative frame boundaries for O(log N) sample-to-dataset lookup
        cumulative = 0
        self._ds_cumulative_frames = []
        for ds in self.multi_dataset._datasets:
            cumulative += ds.num_frames
            self._ds_cumulative_frames.append(cumulative)

        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }
        # dataset range: [self._start_idx, self._end_idx)
        if lerobot_ds_version == "3.0" and val_set_proportion > 1e-6:
            total_episodes = len(self.episode_data_index["from"])
            split_ep = int(total_episodes * (1 - val_set_proportion))
            # at least one episode for training and validation to avoid empty dataset
            split_ep = max(1, split_ep)
            val_start_ep = min(split_ep, total_episodes - 1)
            if self.is_training_set:
                self._start_idx = 0
                self._end_idx = self.episode_data_index["to"][split_ep - 1].item()
            else:
                self._start_idx = self.episode_data_index["from"][val_start_ep].item()
                self._end_idx = self.episode_data_index["to"][-1].item()
        else:
            self._start_idx = 0
            self._end_idx = self.multi_dataset.num_frames

    def _setup_meta_lerobot_keys(self):
        """Configs must provide the explicit raw layout for every meta."""
        for group_name, metas in (
            ("images", self.image_meta or []),
            ("state", self.state_meta),
            ("action", self.action_meta),
        ):
            for meta in metas:
                required = ["key", "lerobot_key", "start_index", "raw_shape", "shape"]
                required += ["camera_type"] if group_name == "images" else []
                missing = [field for field in required if field not in meta]
                if missing:
                    raise KeyError(
                        f"{group_name} meta for key={meta.get('key')!r} is missing fields: {missing}."
                    )
                unexpected = [field for field in _FORBIDDEN_META_KEYS if field in meta]
                if unexpected:
                    raise ValueError(
                        f"{group_name} meta for key={meta.get('key')!r} contains forbidden fields: {unexpected}."
                    )
                key = meta["key"]
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        f"{group_name} meta key must be a non-empty string, got {key!r}."
                    )
                meta["start_index"] = int(meta["start_index"])
                if "time_offset" in meta:
                    meta["time_offset"] = int(meta["time_offset"])
                else:
                    meta["time_offset"] = 0
                meta["query_positions"] = None

    @staticmethod
    def _append_offsets(
        delta_timestamps: Dict[str, List[float]], lerobot_key: str, offsets: List[float]
    ) -> List[int]:
        """
        Merge requested offsets into one lerobot query list for a raw key.

        Args:
            delta_timestamps: Global query plan being built for
                `MultiLeRobotDataset`. Each raw lerobot key maps to a list of
                time offsets in seconds.
            lerobot_key: Raw lerobot column to update, e.g.
                `observation.state.left_arm`.
            offsets: Offsets requested by one meta for that raw key.

        Returns:
            A list of integer positions. Each position tells the caller where
            its requested offset lands inside `delta_timestamps[lerobot_key]`
            after de-duplication.
        """
        existing = delta_timestamps.setdefault(lerobot_key, [])
        positions = []
        for offset in offsets:
            try:
                pos = existing.index(offset)
            except ValueError:
                existing.append(offset)
                pos = len(existing) - 1
            positions.append(pos)
        return positions

    def _build_delta_timestamps(self, fps, past_action_size, action_size) -> Dict[str, list]:
        """
        Build the raw lerobot query plan used by `MultiLeRobotDataset`.

        Args:
            fps: Dataset frequency. Integer frame offsets are converted into
                seconds by dividing with `fps`.
            past_action_size: Number of past action steps. Current dataset
                assumes `0`.
            action_size: Number of action target steps returned to training.

        Returns:
            Dict mapping raw lerobot keys to the list of offsets (in seconds)
            that should be fetched for each key.

        Side effects:
            - Fills `query_positions` for every state meta.
            - Fills `query_positions` for every action meta.

        Important detail:
            State keys and action keys may point to the same raw lerobot column.
            `_append_offsets()` merges those requests into one offset list and
            records which positions belong to which meta, so later
            `__getitem__` can slice the shared query result back into per-key
            tensors.

        obs_size / obs_stride:
            obs_size frames, stride obs_stride steps apart (both image and state).
            Offsets: [-(obs_size-1)*obs_stride/fps, ..., -obs_stride/fps, 0]
        """
        query_offsets_by_key = {}

        obs_size = self.obs_size
        obs_stride = self.obs_stride

        # Images: sample obs_size frames spaced obs_stride steps apart.
        if self.image_meta is not None:
            for meta in self.image_meta:
                image_offsets = [
                    (meta["time_offset"] + step) / fps
                    for step in reversed(range(0, -obs_size * obs_stride, -obs_stride))
                ]
                query_offsets_by_key[meta["lerobot_key"]] = image_offsets

        # States: sample obs_size steps spaced obs_stride steps apart.
        for meta in self.state_meta:
            state_offsets = [
                (meta["time_offset"] + step) / fps
                for step in reversed(range(0, -obs_size * obs_stride, -obs_stride))
            ]
            query_positions = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                state_offsets,
            )
            meta["query_positions"] = query_positions

        # Actions query the raw source declared by their own meta.
        for meta in self.action_meta:
            action_offsets = [
                (meta["time_offset"] + step) / fps for step in range(-past_action_size, action_size)
            ]
            query_positions = self._append_offsets_for_meta(
                query_offsets_by_key,
                meta["lerobot_key"],
                action_offsets,
            )
            meta["query_positions"] = query_positions

        return query_offsets_by_key

    @staticmethod
    def _is_multi_source_meta(meta: Dict[str, Any]) -> bool:
        return isinstance(meta.get("lerobot_key"), (list, tuple))

    @staticmethod
    def _slice_single_meta_feature(data: torch.Tensor, start: int, width: int) -> torch.Tensor:
        if data.ndim == 0:
            return data
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        elif data.ndim > 2:
            data = data.reshape(*data.shape[:-2], -1)
        return data[..., start : start + width]

    @staticmethod
    def _slice_meta_feature(data: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        if BaseLerobotDataset._is_multi_source_meta(meta):
            starts = meta.get("source_start_indices")
            widths = meta.get("source_raw_shapes")
            if starts is None or widths is None:
                raise KeyError(
                    "Multi-source meta requires 'source_start_indices' and 'source_raw_shapes'."
                )
            if not isinstance(data, (list, tuple)):
                raise TypeError(
                    f"Multi-source meta expects a list/tuple of tensors, got {type(data)!r}."
                )
            if not (len(data) == len(starts) == len(widths)):
                raise ValueError("Multi-source meta lengths must match between sources and slices.")

            parts = [
                BaseLerobotDataset._slice_single_meta_feature(source, start, width)
                for source, start, width in zip(data, starts, widths, strict=True)
            ]
            return torch.cat(parts, dim=-1)

        start = meta["start_index"]
        width = meta["raw_shape"]
        if not isinstance(width, int):
            return data
        return BaseLerobotDataset._slice_single_meta_feature(data, start, width)

    def _append_offsets_for_meta(
        self,
        query_offsets_by_key: Dict[str, List[float]],
        lerobot_key,
        offsets: List[float],
    ):
        if isinstance(lerobot_key, (list, tuple)):
            return [self._append_offsets(query_offsets_by_key, key, offsets) for key in lerobot_key]
        return self._append_offsets(query_offsets_by_key, lerobot_key, offsets)

    @staticmethod
    def _get_meta_source_data(container: Dict[str, Any], meta: Dict[str, Any]):
        lerobot_key = meta["lerobot_key"]
        if isinstance(lerobot_key, (list, tuple)):
            return [container[key] for key in lerobot_key]
        return container[lerobot_key]

    @staticmethod
    def _slice_query_tensor(data: torch.Tensor, positions: Optional[List[int]]) -> torch.Tensor:
        """
        Select the positions belonging to one meta from a shared query tensor.

        Args:
            data: Tensor returned by `MultiLeRobotDataset` for one raw lerobot
                key. The first dimension enumerates queried offsets.
            positions: Integer positions previously produced by
                `_append_offsets()`. `None` means "use the whole tensor".

        Returns:
            Tensor restricted to the requested positions, preserving the input
            dtype/device.
        """
        if positions is None:
            return data
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected tensor query result, got {type(data)!r}")
        pos = torch.as_tensor(positions, device=data.device, dtype=torch.long)
        return data.index_select(0, pos)

    def _get_sample_action_tensor(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one action key from the shared lerobot sample.

        Args:
            meta: One action meta after `_build_delta_timestamps()`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Tensor for this action key with time dimension already sliced to the
            offsets specified by `query_positions`.
        """
        if self._is_multi_source_meta(meta):
            action = [
                self._slice_query_tensor(lerobot_sample[key], positions)
                for key, positions in zip(meta["lerobot_key"], meta["query_positions"], strict=True)
            ]
        else:
            action = self._slice_query_tensor(
                lerobot_sample[meta["lerobot_key"]], meta["query_positions"]
            )
        return self._slice_meta_feature(action, meta)

    def _get_sample_state_tensor(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one state key from the shared lerobot sample.

        Args:
            meta: One state meta after `_build_delta_timestamps()`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Tensor for this state key with time dimension already sliced to the
            offsets specified by `query_positions`.
        """
        if self._is_multi_source_meta(meta):
            state = [
                self._slice_query_tensor(lerobot_sample[key], positions)
                for key, positions in zip(meta["lerobot_key"], meta["query_positions"], strict=True)
            ]
        else:
            state = self._slice_query_tensor(
                lerobot_sample[meta["lerobot_key"]], meta["query_positions"]
            )
        return self._slice_meta_feature(state, meta)

    def _get_pad_mask(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract the pad mask corresponding to one state/action query.

        Args:
            meta: State or action meta with resolved query information.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.

        Returns:
            Boolean tensor aligned with the query tensor produced for this meta.
            Pad mask aligned with this meta's declared raw source.
        """
        query_key = (
            meta["lerobot_key"][0] if self._is_multi_source_meta(meta) else meta["lerobot_key"]
        )
        pad_key = f"{query_key}_is_pad"
        if pad_key not in lerobot_sample:
            raise KeyError(f"Missing pad key {pad_key!r} in lerobot sample.")
        positions = (
            meta["query_positions"][0]
            if self._is_multi_source_meta(meta)
            else meta["query_positions"]
        )
        pad = self._slice_query_tensor(lerobot_sample[pad_key], positions)
        if pad.ndim == 0:
            pad = pad.unsqueeze(0)
        return pad.bool()

    def _get_episode_state_sequence(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Read one full-episode state sequence for stats / episode utilities.

        Args:
            meta: One state meta.
            lerobot_sample: Episode-level sample returned by
                `MultiLeRobotDataset.get_episode_data()`.

        Returns:
            Tensor with shape `(T, D)` for the requested state key.
        """
        state = self._slice_meta_feature(self._get_meta_source_data(lerobot_sample, meta), meta)
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        if meta["time_offset"] != 0:
            state = shift_sequence_with_replication(state, meta["time_offset"])
        assert state.shape[-1] == meta["raw_shape"], (
            f"State '{meta['key']}' shape {state.shape[-1]} mismatch with meta {meta['raw_shape']}."
        )
        return state

    def _get_episode_action_sequence(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Build one full-episode raw action target sequence.

        Args:
            meta: One action meta.
            lerobot_sample: Episode-level sample returned by
                `MultiLeRobotDataset.get_episode_data()`.

        Returns:
            Tensor with shape `(T, D)` representing the raw supervision target
            for this action key before sliding-window expansion.
        """
        action = self._slice_meta_feature(self._get_meta_source_data(lerobot_sample, meta), meta)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if meta["time_offset"] != 0:
            action = shift_sequence_with_replication(action, meta["time_offset"])
        assert action.shape[-1] == meta["raw_shape"], (
            f"Action '{meta['key']}' shape {action.shape[-1]} mismatch with meta {meta['raw_shape']}."
        )
        return action

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one action key from a sample-level lerobot query result.

        Args:
            meta: One action meta after initialization has filled
                `query_positions`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.
                For each queried raw lerobot key, the typical tensor shape is
                `[num_requested_offsets, raw_shape]` (or `[num_requested_offsets]`
                for scalar keys before unsqueeze).

        Returns:
            Tensor with shape `[action_size, raw_shape]` for the current action
            key.
        """
        key, raw_shape = meta["key"], meta["raw_shape"]
        action: torch.Tensor = self._get_sample_action_tensor(meta, lerobot_sample)
        if action.ndim == 1:  # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, (
            f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        )
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        """
        Extract one state key from a sample-level lerobot query result.

        Args:
            meta: One state meta after initialization has filled
                `query_positions`.
            lerobot_sample: Sample returned by `MultiLeRobotDataset.__getitem__`.
                For each queried raw lerobot key, the typical tensor shape is
                `[num_requested_offsets, raw_shape]` (or `[num_requested_offsets]`
                for scalar keys before unsqueeze).

        Returns:
            Tensor with shape `[obs_size, raw_shape]` for the current state key.
            In the current dataset assumptions `obs_size == 1`, so the typical
            result is `[1, raw_shape]`.
        """
        key, raw_shape = meta["key"], meta["raw_shape"]
        state: torch.Tensor = self._get_sample_state_tensor(meta, lerobot_sample)
        if state.ndim == 1:  # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, (
            f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        )
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        lerobot_key, raw_shape = meta["lerobot_key"], meta["raw_shape"]
        if lerobot_key == "__dummy__":
            C, H, W = raw_shape[0], raw_shape[1], raw_shape[2]
            return torch.zeros(1, C, H, W, dtype=torch.uint8)
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3:  # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8)  # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image

    def _inject_dummy_images(self, sample: dict) -> dict:
        """Inject zero tensors for image entries marked ``dummy: true`` in shape_meta."""
        if not self._dummy_image_meta:
            return sample
        images = sample.setdefault("images", {})
        T = None
        for v in images.values():
            if hasattr(v, "shape") and v.ndim >= 1:
                T = int(v.shape[0])
                break
        if T is None:
            T = self.obs_size
        for meta in self._dummy_image_meta:
            key = meta["key"]
            if key in images:
                continue
            shape = meta.get("shape") or meta.get("raw_shape")
            if shape is None or len(shape) != 3:
                raise ValueError(f"dummy image meta {key!r} requires shape=[C,H,W], got {shape!r}")
            C, H, W = shape
            images[key] = torch.zeros(T, C, H, W, dtype=torch.uint8)
        sample["images"] = images
        return sample

    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_episode_state_sequence(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_episode_action_sequence(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.load_images = bool(flag) and bool(self.image_meta)
        self.multi_dataset.set_load_images(self.load_images)
        self.multi_dataset.set_during_training(flag)

    def enable_overfit(self, n_samples: int):
        """Pin overfit mode to a stable subset of qualified samples."""
        target_samples = min(int(n_samples), self._end_idx - self._start_idx)
        self._overfit_indices = self._collect_overfit_indices(target_samples)
        self._overfit_len = len(self._overfit_indices)

    def _collect_overfit_indices(self, n_samples: int) -> List[int]:
        if n_samples <= 0:
            return []

        if not self.is_training_set:
            return list(range(self._start_idx, self._start_idx + n_samples))

        qualified_indices = []
        skipped_unqualified = 0

        for sample_idx in range(self._start_idx, self._end_idx):
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
            except Exception as err:
                location = self._locate_sample(sample_idx)
                logger.warning(
                    f"Skipping corrupted overfit candidate {sample_idx} ({location}). Error: {err}"
                )
                continue

            if lerobot_sample.get("step_is_qualified", True):
                qualified_indices.append(sample_idx)
                if len(qualified_indices) >= n_samples:
                    break
            else:
                skipped_unqualified += 1

        if not qualified_indices:
            raise RuntimeError(
                "Overfit mode could not find any qualified samples in the selected dataset range."
            )

        if len(qualified_indices) < n_samples:
            logger.warning(
                "Overfit mode requested %s samples but only found %s qualified samples.",
                n_samples,
                len(qualified_indices),
            )

        if skipped_unqualified > 0:
            logger.info(
                "Overfit mode skipped %s unqualified samples before selecting %s stable samples.",
                skipped_unqualified,
                len(qualified_indices),
            )

        return qualified_indices

    def __len__(self):
        if hasattr(self, "_overfit_len"):
            return self._overfit_len
        return self._end_idx - self._start_idx

    def _locate_sample(self, sample_idx: int) -> str:
        """Map a global sample index back to its dataset directory and local index (O(log N) bisect)."""
        i = bisect.bisect_right(self._ds_cumulative_frames, sample_idx)
        if i < len(self._ds_cumulative_frames):
            ds = self.multi_dataset._datasets[i]
            local_idx = sample_idx - (self._ds_cumulative_frames[i - 1] if i > 0 else 0)
            ds_dir = getattr(ds, "root", getattr(ds, "repo_id", f"dataset[{i}]"))
            return f"dataset_dir={ds_dir}, local_idx={local_idx}"
        return f"sample_idx={sample_idx} (out of bounds)"

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def _record_invalid_sample(
        self,
        *,
        stage: str,
        requested_idx: int,
        sample_idx: int,
        error: Optional[Exception] = None,
    ) -> None:
        location = self._locate_sample(sample_idx)
        key = f"{stage}|{sample_idx}|{location}"
        record = dict(self._invalid_sample_records.get(key, {}))
        if not record:
            record = {
                "stage": stage,
                "requested_idx": int(requested_idx),
                "sample_idx": int(sample_idx),
                "location": location,
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
                "count": 0,
            }
            self._invalid_sample_order.append(key)
        record["count"] = int(record.get("count", 0)) + 1
        self._invalid_sample_records[key] = record

    def get_invalid_sample_report(self, reset: bool = False) -> List[Dict[str, Any]]:
        report = [
            dict(self._invalid_sample_records[key]) for key in list(self._invalid_sample_order)
        ]
        if reset:
            self.clear_invalid_sample_report()
        return report

    def clear_invalid_sample_report(self) -> None:
        self._invalid_sample_records.clear()
        self._invalid_sample_order[:] = []

    def _resample_random_idx(self) -> int:
        """Random absolute sample_idx for retry fallback.

        Subclasses with index filtering (e.g. DroidLerobotDataset's
        _valid_local_indices) override to ensure retry stays within valid frames.
        """
        return np.random.randint(self._start_idx, self._end_idx)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        requested_idx = idx
        overfit_active = hasattr(self, "_overfit_indices")
        if overfit_active:
            sample_idx = int(self._overfit_indices[idx])
        else:
            sample_idx = idx + self._start_idx
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
                    f"(attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
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
                # Get data from lerobot, organized in nested dict
                # action:
                #   left_arm: torch.Tensor
                #   right_arm: torch.Tensor
                # state:
                #   left_arm: torch.Tensor
                #   right_arm: torch.Tensor
                # images:
                #   head_rgb: torch.Tensor
                sample = {
                    "idx": sample_idx,
                    "task": lerobot_sample["task"],
                    "dataset_locator": self._locate_sample(sample_idx),
                    "action": {},
                    "state": {},
                    "images": {},
                    "frequency": self.model_fps,  # Model control frequency (override_fps), used by action tokenizer encode/decode.
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
                    for meta in self.image_meta:
                        sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
                    sample["image_is_pad"] = lerobot_sample[
                        f"{self.image_meta[0]['lerobot_key']}_is_pad"
                    ]
                self._inject_dummy_images(sample)

                sample["action_is_pad"] = action_is_pad
                sample["state_is_pad"] = state_is_pad

                if (
                    "chunked_task_index" in lerobot_sample
                    and lerobot_sample["chunked_task_index"] is not None
                ):
                    chunked_task_index = lerobot_sample["chunked_task_index"]
                    mask = (
                        chunked_task_index != chunked_task_index[0]
                    )  # (T,), True for steps with different subtask
                    assert mask.shape == sample["action_is_pad"].shape, (
                        f"Mask shape {mask.shape} does not match action_is_pad shape {sample['action_is_pad'].shape}"
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

                # Preprocess the sample using the processor
                # for quick data loading
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

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(
        self,
        preprocessor: BaseProcessor,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Compute dataset statistics for normalization.

        Args:
            preprocessor: Processor to transform action/state data
            only_keys: If provided, only compute stats for specified keys.
                       Format: {"action": {"left_arm", "right_arm"}, "state": {"torso"}}
                       If None, compute stats for all keys.

        Returns:
            Dict with structure: {"action": {key: {...stats}}, "state": {key: {...stats}}}
        """
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)
        state_q001 = DefaultDict(list)
        state_q999 = DefaultDict(list)
        state_q0001 = DefaultDict(list)
        state_q9999 = DefaultDict(list)
        state_q00001 = DefaultDict(list)
        state_q99999 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)
        action_q001 = DefaultDict(list)
        action_q999 = DefaultDict(list)
        action_q0001 = DefaultDict(list)
        action_q9999 = DefaultDict(list)
        action_q00001 = DefaultDict(list)
        action_q99999 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes

        state_keys_to_compute = (
            set(m["key"] for m in self.state_meta)
            if only_keys is None
            else only_keys.get("state", set())
        )
        action_keys_to_compute = (
            set(m["key"] for m in self.action_meta)
            if only_keys is None
            else only_keys.get("action", set())
        )

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, num) for num in range(episodes_num)]

            for future in tqdm(
                as_completed(futures),
                total=episodes_num,
                desc="Iterating dataset to get normalization",
            ):
                try:
                    batch = future.result()
                    for meta in self.state_meta:
                        key = meta["key"]
                        if key not in state_keys_to_compute:
                            continue
                        cur_state: torch.Tensor = batch["state"][key]  # (B, T, dim)
                        state_min[key].append(cur_state.amin(0))
                        state_max[key].append(cur_state.amax(0))
                        state_mean[key].append(cur_state.mean(0))
                        state_var[key].append(cur_state.var(0))
                        state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                        state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                        state_q001[key].append(
                            torch.quantile(cur_state, 0.001, dim=0, keepdim=False)
                        )
                        state_q999[key].append(
                            torch.quantile(cur_state, 0.999, dim=0, keepdim=False)
                        )
                        state_q0001[key].append(
                            torch.quantile(cur_state, 0.0001, dim=0, keepdim=False)
                        )
                        state_q9999[key].append(
                            torch.quantile(cur_state, 0.9999, dim=0, keepdim=False)
                        )
                        state_q00001[key].append(
                            torch.quantile(cur_state, 0.00001, dim=0, keepdim=False)
                        )
                        state_q99999[key].append(
                            torch.quantile(cur_state, 0.99999, dim=0, keepdim=False)
                        )

                    for meta in self.action_meta:
                        key = meta["key"]
                        if key not in action_keys_to_compute:
                            continue
                        cur_action: torch.Tensor = batch["action"][key]  # (B, T, dim)
                        action_min[key].append(cur_action.amin(0))
                        action_max[key].append(cur_action.amax(0))
                        action_mean[key].append(cur_action.mean(0))
                        action_var[key].append(cur_action.var(0))
                        action_q01[key].append(
                            torch.quantile(cur_action, 0.01, dim=0, keepdim=False)
                        )
                        action_q99[key].append(
                            torch.quantile(cur_action, 0.99, dim=0, keepdim=False)
                        )
                        action_q001[key].append(
                            torch.quantile(cur_action, 0.001, dim=0, keepdim=False)
                        )
                        action_q999[key].append(
                            torch.quantile(cur_action, 0.999, dim=0, keepdim=False)
                        )
                        action_q0001[key].append(
                            torch.quantile(cur_action, 0.0001, dim=0, keepdim=False)
                        )
                        action_q9999[key].append(
                            torch.quantile(cur_action, 0.9999, dim=0, keepdim=False)
                        )
                        action_q00001[key].append(
                            torch.quantile(cur_action, 0.00001, dim=0, keepdim=False)
                        )
                        action_q99999[key].append(
                            torch.quantile(cur_action, 0.99999, dim=0, keepdim=False)
                        )

                except Exception as e:
                    logger.error(f"Error processing episode: {e}")

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict)}
        for meta in self.state_meta:
            key = meta["key"]
            if key not in state_keys_to_compute:
                continue
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            stats["state"][key]["stepwise_q001"] = torch.stack(state_q001[key]).amin(0)
            stats["state"][key]["stepwise_q999"] = torch.stack(state_q999[key]).amax(0)
            stats["state"][key]["global_q001"] = stats["state"][key]["stepwise_q001"].amin(0)
            stats["state"][key]["global_q999"] = stats["state"][key]["stepwise_q999"].amax(0)
            stats["state"][key]["stepwise_q0001"] = torch.stack(state_q0001[key]).amin(0)
            stats["state"][key]["stepwise_q9999"] = torch.stack(state_q9999[key]).amax(0)
            stats["state"][key]["global_q0001"] = stats["state"][key]["stepwise_q0001"].amin(0)
            stats["state"][key]["global_q9999"] = stats["state"][key]["stepwise_q9999"].amax(0)
            stats["state"][key]["stepwise_q00001"] = torch.stack(state_q00001[key]).amin(0)
            stats["state"][key]["stepwise_q99999"] = torch.stack(state_q99999[key]).amax(0)
            stats["state"][key]["global_q00001"] = stats["state"][key]["stepwise_q00001"].amin(0)
            stats["state"][key]["global_q99999"] = stats["state"][key]["stepwise_q99999"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            if key not in action_keys_to_compute:
                continue
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            stats["action"][key]["stepwise_q001"] = torch.stack(action_q001[key]).amin(0)
            stats["action"][key]["stepwise_q999"] = torch.stack(action_q999[key]).amax(0)
            stats["action"][key]["global_q001"] = stats["action"][key]["stepwise_q001"].amin(0)
            stats["action"][key]["global_q999"] = stats["action"][key]["stepwise_q999"].amax(0)
            stats["action"][key]["stepwise_q0001"] = torch.stack(action_q0001[key]).amin(0)
            stats["action"][key]["stepwise_q9999"] = torch.stack(action_q9999[key]).amax(0)
            stats["action"][key]["global_q0001"] = stats["action"][key]["stepwise_q0001"].amin(0)
            stats["action"][key]["global_q9999"] = stats["action"][key]["stepwise_q9999"].amax(0)
            stats["action"][key]["stepwise_q00001"] = torch.stack(action_q00001[key]).amin(0)
            stats["action"][key]["stepwise_q99999"] = torch.stack(action_q99999[key]).amax(0)
            stats["action"][key]["global_q00001"] = stats["action"][key]["stepwise_q00001"].amin(0)
            stats["action"][key]["global_q99999"] = stats["action"][key]["stepwise_q99999"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].

    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)

    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window

    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0

    N, D = x.shape

    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)  # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices  # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out

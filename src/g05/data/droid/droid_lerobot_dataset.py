import copy
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from g05.data.base_lerobot_datasetV3 import BaseLerobotDatasetV3

logger = logging.getLogger(__name__)


class DroidActionSpace(Enum):
    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_VELOCITY = auto()
    CARTESIAN_POSITION_DELTA = auto()


# Default DROID-specific image schema. Lives in dataset (not yaml) because:
# (1) DROID always provides exactly two exterior cams + one wrist;
# (2) processor.shape_meta only declares the cams the model sees (1 exterior +
#     wrist + optional zero-injected slots), keeping the per-emb configs minimal.
_DEFAULT_AUX_EXTERIOR_LEROBOT_KEY = "observation.images.exterior_2_left"


class DroidLerobotDataset(BaseLerobotDatasetV3):
    """
    DROID dataset on the unified explicit schema.

    Gripper flip (`1 - x`) is handled in `_slice_meta_feature`, so sample path
    and fast-stats path share the exact same behavior.

    Droid-specific behavior happens entirely in this class so the processor
    side stays embodiment-agnostic:

    - ``random_swap_exterior_images``: training time 50/50 between two ext cams,
      then drop the auxiliary key — model sees exactly 1 exterior slot.
    - ``random_select_instruction``: 3-choose-1 over (task, lang2, lang3) loaded
      from raw parquet (lang2/3 are not in lerobot v3's whitelist).
    - dummy-image injection: any meta in ``shape_meta.images`` with
      ``dummy: true`` is filled with zero tensors here, so the processor never
      needs to know about Droid-specific layout (or about dummy slots in
      general). This lets a 3-cam processor share its keyset with cross-emb
      batches that mix in 2-cam Droid samples.
    """

    GRIPPER_LEROBOT_KEYS = {
        "observation.state.gripper_position",
        "action.gripper_position",
    }

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,
        obs_size: int = 1,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "2.1",
        action_space: str = "joint_position",
        filter_failed_trajectories: bool = False,
        load_lang_alternatives: bool = True,
        random_swap_exterior_images: bool = True,
        random_select_instruction: bool = True,
        use_molmo_annotations: bool = False,
        primary_exterior_key: str = "exterior_image",
        aux_exterior_key: str = "exterior_image_2",
        aux_exterior_lerobot_key: str = _DEFAULT_AUX_EXTERIOR_LEROBOT_KEY,
        **kwargs,
    ):
        self.filter_failed_trajectories = filter_failed_trajectories
        self._valid_local_indices: Optional[List[int]] = None
        self.random_swap_exterior_images = random_swap_exterior_images
        self.random_select_instruction = random_select_instruction
        self.primary_exterior_key = primary_exterior_key
        self.aux_exterior_key = aux_exterior_key

        # Deep-copy because we are about to inject the aux-exterior entry, and
        # the same shape_meta is also referenced by processor config — must not
        # mutate it in place.
        from omegaconf import OmegaConf, DictConfig
        if isinstance(shape_meta, DictConfig):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            shape_meta = copy.deepcopy(shape_meta)

        # Save dummy meta (driven by ``dummy: true`` in shape_meta.images) so we
        # can inject zeros after super loads images. base_lerobot already filters
        # these out of self.image_meta, so we cache a copy here.
        all_image_meta = list(shape_meta.get("images") or [])
        self._dummy_image_meta = [m for m in all_image_meta if m.get("dummy", False)]

        # Inject the aux-exterior entry into shape_meta so base_lerobot loads it.
        # The processor never sees this entry — we drop the key from sample after
        # _swap_exterior_images.
        if self.aux_exterior_key not in {m.get("key") for m in all_image_meta}:
            primary_meta = next(
                (m for m in all_image_meta if m.get("key") == self.primary_exterior_key),
                None,
            )
            if primary_meta is not None:
                aux_meta = copy.deepcopy(primary_meta)
                aux_meta["key"] = self.aux_exterior_key
                aux_meta["lerobot_key"] = aux_exterior_lerobot_key
                idx = all_image_meta.index(primary_meta)
                all_image_meta.insert(idx + 1, aux_meta)
                shape_meta["images"] = all_image_meta

        super().__init__(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            action_size=action_size,
            past_action_size=past_action_size,
            obs_size=obs_size,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            lerobot_ds_version=lerobot_ds_version,
            **kwargs,
        )
        if self.filter_failed_trajectories:
            self._build_valid_local_indices()
        self._lang_alternatives: Dict[str, Tuple[str, str]] = {}
        if load_lang_alternatives:
            self._lang_alternatives = self._build_lang_alternatives()
        self._molmo_annotations: List[Dict[int, str]] = []
        if use_molmo_annotations:
            self._molmo_annotations = self._load_molmo_annotations()

    def _build_valid_local_indices(self) -> None:
        """Filter failed episodes at construction time and build local valid-frame indices."""
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        def _is_success(v):
            if isinstance(v, (list, tuple)):
                return bool(v[0])
            return bool(v)

        col = "stats/is_episode_successful/min"

        def _load_valid_set(ds):
            """Return (ds, valid_set | None), reading NAS parquet files in parallel."""
            pq_files = sorted(ds.root.glob("meta/episodes/**/*.parquet"))
            if not pq_files:
                return ds, None
            try:
                df = pd.concat(
                    [pd.read_parquet(str(f), columns=[col]) for f in pq_files],
                    ignore_index=True,
                )
            except Exception:
                return ds, None
            if col not in df.columns:
                return ds, None
            valid_set = set(df.index[df[col].map(_is_success)].tolist())
            before, after = len(df), len(valid_set)
            logger.info(
                f"🔍 [DROID filter_failed] {ds.root.name}: "
                f"{before} → {after} episodes ({before - after} failed removed)"
            )
            return ds, valid_set

        # Read all splits in parallel. This is mostly NAS I/O-bound, so threads help.
        datasets = self.multi_dataset._datasets
        with ThreadPoolExecutor(max_workers=min(len(datasets), 16)) as pool:
            results = dict(pool.map(_load_valid_set, datasets))

        valid_ep_sets = [results[ds] for ds in datasets]

        # Map global episode index → dataset index
        ep_counts = [len(ds.episode_data_index["from"]) for ds in self.multi_dataset._datasets]
        ep_ds_offsets = []
        acc = 0
        for c in ep_counts:
            ep_ds_offsets.append(acc)
            acc += c

        def _ds_idx_for_ep(global_ep: int) -> int:
            for i in range(len(ep_ds_offsets) - 1, -1, -1):
                if global_ep >= ep_ds_offsets[i]:
                    return i
            return 0

        valid_local: List[int] = []
        for gep in range(len(self.episode_data_index["from"])):
            ds_i = _ds_idx_for_ep(gep)
            local_ep = gep - ep_ds_offsets[ds_i]
            valid_set = valid_ep_sets[ds_i]
            if valid_set is not None and local_ep not in valid_set:
                continue
            g_from = int(self.episode_data_index["from"][gep])
            g_to = int(self.episode_data_index["to"][gep])
            eff_from = max(g_from, self._start_idx)
            eff_to = min(g_to, self._end_idx)
            if eff_from < eff_to:
                valid_local.extend(range(eff_from - self._start_idx, eff_to - self._start_idx))

        total = self._end_idx - self._start_idx
        split = "train" if self.is_training_set else "val"
        if not valid_local:
            logger.warning(
                f"⚠️ [DROID filter_failed] {split}: no valid frames found, disabling filter"
            )
            return
        logger.info(
            f"✅ [DROID filter_failed] {split}: {total} → {len(valid_local)} frames valid"
        )
        self._valid_local_indices = valid_local

    def _build_lang_alternatives(self) -> Dict[str, Tuple[str, str]]:
        """Build {primary_task → (lang2, lang3)} from Droid data parquets.

        language_instruction_2/3 are constant within each episode and stored in
        the frame-level parquet but excluded from lerobot_dataset_v3's _META_COLS
        whitelist. We read them once at init time (3 columns only, parallelised)
        so _pick_lang_alternative() can do true 3-choose-1.
        """
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        COLS = ["language_instruction", "language_instruction_2", "language_instruction_3"]

        def _read_one(ds):
            pq_files = sorted(ds.root.glob("data/**/*.parquet"))
            if not pq_files:
                return pd.DataFrame()
            try:
                frames = [pd.read_parquet(f, columns=COLS) for f in pq_files]
                return (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset=["language_instruction"])
                )
            except Exception:
                return pd.DataFrame()

        datasets = self.multi_dataset._datasets
        with ThreadPoolExecutor(max_workers=min(len(datasets), 16)) as pool:
            results = list(pool.map(_read_one, datasets))

        mapping: Dict[str, Tuple[str, str]] = {}
        for df in results:
            if df.empty:
                continue
            for _, row in df.iterrows():
                key = str(row.get("language_instruction", "") or "").strip()
                if key and key not in mapping:
                    mapping[key] = (
                        str(row.get("language_instruction_2", "") or "").strip(),
                        str(row.get("language_instruction_3", "") or "").strip(),
                    )

        logger.info(
            f"🗣️ [DROID lang_alternatives] Loaded {len(mapping)} task→(lang2,lang3) mappings"
        )
        return mapping

    def _load_molmo_annotations(self) -> List[Dict[int, str]]:
        """Load per-episode Molmo language annotations from tasks_annotated.parquet.

        Returns a list (one entry per dataset in multi_dataset) of
        {local_episode_index: molmo_task_str} dicts. Datasets without
        tasks_annotated.parquet get an empty dict (3-choose-1 fallback applies).
        """
        import pandas as pd

        result = []
        for ds in self.multi_dataset._datasets:
            ann_path = ds.root / "meta" / "meta" / "tasks_annotated.parquet"
            if not ann_path.exists():
                result.append({})
                continue
            df = pd.read_parquet(ann_path)  # index=episode_index, col='task'
            mapping = {
                int(ep_idx): str(row["task"] or "").strip()
                for ep_idx, row in df.iterrows()
                if str(row["task"] or "").strip()
            }
            logger.info(
                f"🤖 [DROID molmo] {ds.root.name}: {len(mapping)} annotations loaded"
            )
            result.append(mapping)
        return result

    def _pick_lang_alternative(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Language instruction selection.

        When Molmo annotations are available for this episode, use them directly.
        Otherwise falls back to 3-choose-1 over (task, lang2, lang3).
        Only mutates the existing 'task' key so cross-emb collate stays consistent.
        """
        if not self.is_training_set:
            return sample

        # Molmo path: per-episode annotation, use directly when available
        if self._molmo_annotations:
            ds_idx = int(sample.get("dataset_index", torch.tensor(0)).item())
            ep_idx = int(sample["episode_index"].item())
            if ds_idx < len(self._molmo_annotations):
                molmo_task = self._molmo_annotations[ds_idx].get(ep_idx, "")
                if molmo_task:
                    sample["task"] = molmo_task
                    return sample

        # Fallback: 3-choose-1 over original lang alternatives
        task = sample.get("task", "")
        if not (task and task in self._lang_alternatives):
            return sample
        lang2, lang3 = self._lang_alternatives[task]
        candidates = [task] + [c for c in (lang2, lang3) if c and c.strip()]
        if len(candidates) > 1:
            sample["task"] = candidates[np.random.randint(len(candidates))]
        return sample

    @staticmethod
    def _slice_meta_feature(data: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        sliced = BaseLerobotDatasetV3._slice_meta_feature(data, meta)
        if meta["lerobot_key"] in DroidLerobotDataset.GRIPPER_LEROBOT_KEYS:
            return 1 - sliced
        return sliced

    def __len__(self) -> int:
        if self._valid_local_indices is not None:
            return len(self._valid_local_indices)
        return super().__len__()

    def _resample_random_idx(self) -> int:
        """Retry fallback: stay within valid (non-failed-episode) frames."""
        if self._valid_local_indices is None:
            return super()._resample_random_idx()
        filtered_pos = np.random.randint(0, len(self._valid_local_indices))
        return self._valid_local_indices[filtered_pos] + self._start_idx

    def _swap_exterior_images(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """openpi-style 2-choose-1 exterior camera selection.

        Training: 50% chance to overwrite primary with aux. Eval: keep primary.
        Always drop the aux key after — model only has 1 exterior slot.
        """
        images = sample.get("images")
        if not images:
            return sample
        if (
            self.random_swap_exterior_images
            and self.is_training_set
            and self.aux_exterior_key in images
            and self.primary_exterior_key in images
            and np.random.rand() > 0.5
        ):
            images[self.primary_exterior_key] = images[self.aux_exterior_key].clone()
        images.pop(self.aux_exterior_key, None)
        sample["images"] = images
        return sample

    def _inject_dummy_images(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Inject zero tensors for image entries marked ``dummy: true``.

        Uses the meta's target ``shape`` (== camera_size_config[cam_type]) and
        float32 dtype so the processor's transforms — including the auto-injected
        T.Resize — pass through as identity and pixel_values matches sibling
        non-dummy cameras' downstream layout.
        """
        if not self._dummy_image_meta:
            return sample
        images = sample.setdefault("images", {})
        T: Optional[int] = None
        for v in images.values():
            if hasattr(v, "shape") and v.ndim >= 1:
                T = int(v.shape[0])
                break
        if T is None:
            pad = sample.get("image_is_pad")
            if pad is not None and hasattr(pad, "shape") and pad.ndim >= 1:
                T = int(pad.shape[0])
        if T is None:
            T = int(getattr(self, "obs_size", 1) or 1)

        for meta in self._dummy_image_meta:
            key = meta["key"]
            if key in images:
                continue
            shape = meta.get("shape") or meta.get("raw_shape")
            if shape is None or len(shape) != 3:
                raise ValueError(
                    f"dummy image meta {key!r} requires shape=[C,H,W], got {shape!r}"
                )
            C, H, W = shape
            images[key] = torch.zeros(T, C, H, W, dtype=torch.float32)
        sample["images"] = images
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Remap idx if filter_failed_trajectories is on.
        if self._valid_local_indices is not None:
            if not 0 <= idx < len(self._valid_local_indices):
                raise IndexError(
                    f"Index {idx} out of bounds {len(self._valid_local_indices)}."
                )
            idx = self._valid_local_indices[idx]

        # Disable processor and idx remap while super loads, then run all
        # Droid-specific transforms (swap aux exterior → drop aux → inject dummy
        # zeros → 3-choose-1 lang) before manually calling processor.preprocess.
        # Doing it here (not inside processor.preprocess) keeps the processor
        # embodiment-agnostic and ensures samples_builder reads the augmented
        # `task` field.
        saved_processor = self.processor
        saved_valid = self._valid_local_indices
        self.processor = None
        self._valid_local_indices = None
        try:
            sample = super().__getitem__(idx)
        finally:
            self.processor = saved_processor
            self._valid_local_indices = saved_valid

        sample = self._swap_exterior_images(sample)
        sample = self._inject_dummy_images(sample)
        sample = self._pick_lang_alternative(sample)

        if self.processor is not None:
            sample = self.processor.preprocess(sample)
        return sample


def create_droid_dataset(
    dataset_dirs: List[str],
    action_space: str = "joint_position",
    action_size: int = 16,
    is_training_set: bool = True,
    **kwargs,
) -> DroidLerobotDataset:
    if action_space == "joint_position":
        shape_meta = {
            "state": [
                {
                    "key": "joint_position",
                    "lerobot_key": "observation.state.joint_position",
                    "start_index": 0,
                    "raw_shape": 7,
                    "shape": 7,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "observation.state.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "action": [
                {
                    "key": "joint_position",
                    "lerobot_key": "action.joint_position",
                    "start_index": 0,
                    "raw_shape": 7,
                    "shape": 7,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "action.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "images": [
                {
                    "key": "exterior_image",
                    "lerobot_key": "observation.images.exterior_1_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "exterior_image_2",
                    "lerobot_key": "observation.images.exterior_2_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "wrist_image",
                    "lerobot_key": "observation.images.wrist_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
            ],
        }
    elif action_space == "joint_velocity":
        shape_meta = {
            "state": [
                {
                    "key": "joint_position",
                    "lerobot_key": "observation.state.joint_position",
                    "start_index": 0,
                    "raw_shape": 7,
                    "shape": 7,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "observation.state.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "action": [
                {
                    "key": "joint_velocity",
                    "lerobot_key": "action.joint_velocity",
                    "start_index": 0,
                    "raw_shape": 7,
                    "shape": 7,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "action.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "images": [
                {
                    "key": "exterior_image",
                    "lerobot_key": "observation.images.exterior_1_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "exterior_image_2",
                    "lerobot_key": "observation.images.exterior_2_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "wrist_image",
                    "lerobot_key": "observation.images.wrist_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
            ],
        }
    elif action_space == "cartesian_velocity":
        shape_meta = {
            "state": [
                {
                    "key": "cartesian_position",
                    "lerobot_key": "observation.state.cartesian_position",
                    "start_index": 0,
                    "raw_shape": 6,
                    "shape": 6,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "observation.state.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "action": [
                {
                    "key": "cartesian_velocity",
                    "lerobot_key": "action.cartesian_velocity",
                    "start_index": 0,
                    "raw_shape": 6,
                    "shape": 6,
                    "time_offset": 0,
                },
                {
                    "key": "gripper",
                    "lerobot_key": "action.gripper_position",
                    "start_index": 0,
                    "raw_shape": 1,
                    "shape": 1,
                    "time_offset": 0,
                },
            ],
            "images": [
                {
                    "key": "exterior_image",
                    "lerobot_key": "observation.images.exterior_1_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "exterior_image_2",
                    "lerobot_key": "observation.images.exterior_2_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
                {
                    "key": "wrist_image",
                    "lerobot_key": "observation.images.wrist_left",
                    "start_index": 0,
                    "raw_shape": [3, 256, 256],
                    "shape": [3, 224, 224],
                    "time_offset": 0,
                },
            ],
        }
    else:
        raise ValueError(f"Unknown action space: {action_space}")

    return DroidLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        action_size=action_size,
        action_space=action_space,
        is_training_set=is_training_set,
        **kwargs,
    )

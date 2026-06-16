from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import torch
import json
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hashlib
from pathlib import Path
from git import Repo
import logging

from g05.utils.common.pytorch_utils import dict_apply
from g05.data_processor import BaseActionStateTransform

logger = logging.getLogger(__name__)

ConstConstStr = Annotated[
    str, "format: 'const_min/const_max', where const_min and const_max give the constant range"
]
DummyClipStr = Annotated[
    str,
    "format: 'clip_min/clip_max', where clip_min and clip_max give the clip range for dummy mode",
]
NormMode = Union[
    Literal[
        "dummy",
        "min/max",
        "q01/q99",
        "q001/q999",
        "q0001/q9999",
        "q00001/q99999",
        "z-score",
        "z-score-tail",      # fused: tail_compress → z-score
        "q01/q99-tail",      # fused: tail_compress → q01/q99 linear
        "tanh",
    ],
    ConstConstStr,
]


class LinearNormalizer(BaseActionStateTransform):
    """
    Multi-field linear normalizer for action/state data.

    Forward: normalize each field according to its mode
    Backward: denormalize each field

    This transform is invertible (within the valid range).
    """

    invertible = True

    def __init__(
        self,
        shape_meta,
        use_stepwise_action_norm,
        default_mode: NormMode,
        exception_mode: Dict[str, Dict[str, NormMode]],
        stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        dummy_clip_default: Tuple[float, float] = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        missing_key_mode: str = "error",
        tail_scale: float = 0.075,
    ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]

            if (
                exception_mode is not None
                and "action" in exception_mode
                and key in exception_mode["action"]
            ):
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            if (
                dummy_clip_exception is not None
                and "action" in dummy_clip_exception
                and key in dummy_clip_exception["action"]
            ):
                cur_dummy_clip = dummy_clip_exception["action"][key]
            else:
                cur_dummy_clip = dummy_clip_default

            if cur_mode == "dummy":
                cur_stats = {}
            else:
                if key not in stats["action"]:
                    if missing_key_mode == "error":
                        raise KeyError(
                            f"Action key '{key}' not found in stats['action']. "
                            f"Required for mode='{cur_mode}'. Available keys: {list(stats['action'].keys())}"
                        )
                    else:
                        logger.warning(
                            f"Action key '{key}' not found in stats, falling back to dummy normalization."
                        )
                        cur_mode = "dummy"
                        cur_stats = {}
                if cur_mode != "dummy":
                    if use_stepwise_action_norm:
                        cur_stats = {
                            k.removeprefix("stepwise_"): v
                            for k, v in stats["action"][key].items()
                            if k.startswith("stepwise_")
                        }
                    else:
                        cur_stats = {
                            k.removeprefix("global_"): v
                            for k, v in stats["action"][key].items()
                            if k.startswith("global_")
                        }
                    # Check feature dim: stats shape must match shape_meta shape
                    stat_tensors = [v for v in cur_stats.values() if isinstance(v, torch.Tensor)]
                    if stat_tensors:
                        stats_feat_dim = stat_tensors[0].shape[-1]
                        meta_feat_dim = meta["shape"]
                        if stats_feat_dim != meta_feat_dim:
                            if missing_key_mode == "error":
                                raise ValueError(
                                    f"Action key '{key}' stats feature dim {stats_feat_dim} != "
                                    f"shape_meta dim {meta_feat_dim}. Stats may be stale."
                                )
                            else:
                                logger.warning(
                                    f"Action key '{key}' stats feature dim {stats_feat_dim} != "
                                    f"shape_meta dim {meta_feat_dim}. "
                                    f"Falling back to dummy normalization."
                                )
                                cur_mode = "dummy"
                                cur_stats = {}

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                dummy_clip_range=cur_dummy_clip,
                tail_scale=tail_scale,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]

            if (
                exception_mode is not None
                and "state" in exception_mode
                and key in exception_mode["state"]
            ):
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            if (
                dummy_clip_exception is not None
                and "state" in dummy_clip_exception
                and key in dummy_clip_exception["state"]
            ):
                cur_dummy_clip = dummy_clip_exception["state"][key]
            else:
                cur_dummy_clip = dummy_clip_default

            if cur_mode == "dummy":
                cur_stats = {}
            else:
                if key not in stats["state"]:
                    if missing_key_mode == "error":
                        raise KeyError(
                            f"State key '{key}' not found in stats['state']. "
                            f"Required for mode='{cur_mode}'. Available keys: {list(stats['state'].keys())}"
                        )
                    else:
                        logger.warning(
                            f"State key '{key}' not found in stats, falling back to dummy normalization."
                        )
                        cur_mode = "dummy"
                        cur_stats = {}
                if cur_mode != "dummy":
                    cur_stats = {
                        k.removeprefix("global_"): v
                        for k, v in stats["state"][key].items()
                        if k.startswith("global_")
                    }
                    # Check feature dim: stats shape must match shape_meta shape
                    stat_tensors = [v for v in cur_stats.values() if isinstance(v, torch.Tensor)]
                    if stat_tensors:
                        stats_feat_dim = stat_tensors[0].shape[-1]
                        meta_feat_dim = meta["shape"]
                        if stats_feat_dim != meta_feat_dim:
                            if missing_key_mode == "error":
                                raise ValueError(
                                    f"State key '{key}' stats feature dim {stats_feat_dim} != "
                                    f"shape_meta dim {meta_feat_dim}. Stats may be stale."
                                )
                            else:
                                logger.warning(
                                    f"State key '{key}' stats feature dim {stats_feat_dim} != "
                                    f"shape_meta dim {meta_feat_dim}. "
                                    f"Falling back to dummy normalization."
                                )
                                cur_mode = "dummy"
                                cur_stats = {}

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                dummy_clip_range=cur_dummy_clip,
                tail_scale=tail_scale,
            )

        # Build normalizers for transform-derived keys present in stats but not in shape_meta.
        # These are keys added by transforms (e.g. MultiRotationExpansionTransform adds
        # _euler/_quat/_6drot variants) that the merger will include in the flat tensor.
        # We use default_mode for them unless overridden by exception_mode.
        shape_meta_action_keys = set(self.normalizers["action"].keys())
        for key, key_stats in stats.get("action", {}).items():
            if key in shape_meta_action_keys:
                continue
            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode
            if dummy_clip_exception is not None and "action" in dummy_clip_exception and key in dummy_clip_exception["action"]:
                cur_dummy_clip = dummy_clip_exception["action"][key]
            else:
                cur_dummy_clip = dummy_clip_default
            if cur_mode == "dummy":
                cur_stats = {}
            else:
                if use_stepwise_action_norm:
                    cur_stats = {k.removeprefix("stepwise_"): v for k, v in key_stats.items() if k.startswith("stepwise_")}
                else:
                    cur_stats = {k.removeprefix("global_"): v for k, v in key_stats.items() if k.startswith("global_")}
            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                dummy_clip_range=cur_dummy_clip,
                tail_scale=tail_scale,
            )

        shape_meta_state_keys = set(self.normalizers["state"].keys())
        for key, key_stats in stats.get("state", {}).items():
            if key in shape_meta_state_keys:
                continue
            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode
            if dummy_clip_exception is not None and "state" in dummy_clip_exception and key in dummy_clip_exception["state"]:
                cur_dummy_clip = dummy_clip_exception["state"][key]
            else:
                cur_dummy_clip = dummy_clip_default
            if cur_mode == "dummy":
                cur_stats = {}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in key_stats.items() if k.startswith("global_")}
            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                dummy_clip_range=cur_dummy_clip,
                tail_scale=tail_scale,
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()},
        }
        return stats

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                if key in batch["action"]:
                    batch["action"][key] = norm.forward(batch["action"][key])

        if "state" in batch:
            for key, norm in self.normalizers["state"].items():
                if key in batch["state"]:
                    batch["state"][key] = norm.forward(batch["state"][key])

        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                if key in batch["action"]:
                    batch["action"][key] = norm.backward(batch["action"][key])

        if "state" in batch:
            for key, norm in self.normalizers["state"].items():
                if key in batch["state"]:
                    batch["state"][key] = norm.backward(batch["state"][key])

        return batch


class SingleFieldLinearNormalizer(BaseActionStateTransform):
    """
    Single-field linear normalizer with forward/backward transforms.

    Forward: normalize data according to mode (z-score, min/max, quantile, etc.)
    Backward: denormalize data

    This transform is invertible (within the valid range).
    """

    invertible = True
    std_reg = 1e-8
    tanh_eps = 1e-7
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(
        self,
        stats,
        mode: NormMode = "min/max",
        dummy_clip_range: Tuple[float, float] = (-5.0, 5.0),
        tail_scale: float = 0.075,
    ):
        self.stats = stats
        self.mode = mode
        self.dummy_clip_min, self.dummy_clip_max = dummy_clip_range
        self._horizon_mismatch_warned = False

        # tail compress buffers — non-None only for *-tail modes
        self._tail_q01 = None
        self._tail_q99 = None
        self._tail_c_pos = None
        self._tail_c_neg = None

        if mode == "dummy":
            self.scale = None
            self.offset = None
            return

        _is_tail = mode.endswith("-tail")
        _base_mode = mode[:-5] if _is_tail else mode  # strip "-tail" suffix

        if _base_mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            # Detect near-constant dimensions (std too small) to avoid
            # amplifying noise by a factor of 1/std (~100000x when std=1e-5).
            ignore_dim = input_std < self.range_tol
            scale = 1.0 / (input_std + self.std_reg)
            offset = -input_mean / (input_std + self.std_reg)
            # For near-constant dims: scale=1, offset=-mean → output ≈ 0
            scale[ignore_dim] = 1.0
            offset[ignore_dim] = -input_mean[ignore_dim]
        else:
            if _base_mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif _base_mode in ("q01/q99", "tanh"):
                input_min, input_max = stats["q01"], stats["q99"]
            elif _base_mode == "q001/q999":
                input_min, input_max = stats["q001"], stats["q999"]
            elif _base_mode == "q0001/q9999":
                input_min, input_max = stats["q0001"], stats["q9999"]
            elif _base_mode == "q00001/q99999":
                input_min, input_max = stats["q00001"], stats["q99999"]
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, _base_mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
        # For range-based modes, clamp to +-5 to prevent extreme outliers
        # from blowing up downstream action tokenization.
        # z-score is unbounded by nature, so no clamp.
        # self._clamp = mode not in ("z-score", "tanh")

        if _is_tail:
            q01 = stats["q01"]
            q99 = stats["q99"]
            mean = stats["mean"]
            # For degenerate dims (q01 >= q99 or mean outside [q01, q99]), disable tail (identity)
            degenerate = (q99 <= q01) | (mean <= q01) | (mean >= q99)
            c_pos = tail_scale * (q99 - mean)
            c_neg = tail_scale * (mean - q01)
            # Ensure c > 0 even for degenerate dims (avoid div-by-zero in log1p)
            c_pos = torch.where(degenerate, torch.ones_like(c_pos), c_pos)
            c_neg = torch.where(degenerate, torch.ones_like(c_neg), c_neg)
            self._tail_q01 = q01
            self._tail_q99 = q99
            self._tail_c_pos = c_pos
            self._tail_c_neg = c_neg

    def get_stats(self):
        return self.stats

    def _apply_tail_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise log1p tail compression — identity on [q01, q99]."""
        q01 = self._tail_q01.to(x.device)
        q99 = self._tail_q99.to(x.device)
        c_pos = self._tail_c_pos.to(x.device)
        c_neg = self._tail_c_neg.to(x.device)
        pos_tail = q99 + c_pos * torch.log1p(torch.clamp((x - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.log1p(torch.clamp((q01 - x) / c_neg, min=0.0))
        return torch.where(x > q99, pos_tail, torch.where(x < q01, neg_tail, x))

    def _apply_tail_backward(self, y: torch.Tensor) -> torch.Tensor:
        """Exact inverse of _apply_tail_forward."""
        q01 = self._tail_q01.to(y.device)
        q99 = self._tail_q99.to(y.device)
        c_pos = self._tail_c_pos.to(y.device)
        c_neg = self._tail_c_neg.to(y.device)
        pos_tail = q99 + c_pos * torch.expm1(torch.clamp((y - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.expm1(torch.clamp((q01 - y) / c_neg, min=0.0))
        return torch.where(y > q99, pos_tail, torch.where(y < q01, neg_tail, y))

    def _match_horizon(self, x: torch.Tensor):
        """Slice scale/offset along the horizon (h) dim to match x when stats have more steps."""
        scale, offset = self.scale, self.offset
        if scale.ndim < 2:
            return scale, offset

        h_stats, h_data = scale.shape[-2], x.shape[-2]
        if h_data == h_stats:
            return scale, offset

        if h_data > h_stats:
            raise ValueError(
                f"Data horizon ({h_data}) exceeds stats horizon ({h_stats}). "
                f"Cannot normalize without sufficient statistics."
            )

        if not self._horizon_mismatch_warned:
            logger.warning(
                f"Normalizer horizon mismatch: stats have {h_stats} steps "
                f"but data has {h_data}. Using first {h_data} steps of stats."
            )
            self._horizon_mismatch_warned = True

        return scale[..., :h_data, :], offset[..., :h_data, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "dummy":
            return x.clamp(self.dummy_clip_min, self.dummy_clip_max)

        scale, offset = self._match_horizon(x)
        feat_dim = scale.shape[-1]
        x_main, x_pad = x[..., :feat_dim], x[..., feat_dim:]

        if self._tail_c_pos is not None:
            x_main = self._apply_tail_forward(x_main)

        x_main = x_main * scale + offset
        if self.mode == "tanh":
            x_main = torch.tanh(x_main)
        else:
            x_main = x_main.clamp(-5.0, 5.0)
        # Safety net: replace any residual NaN/Inf with 0 to prevent training crashes.
        # Upstream rotation ops are clamped, but raw data corruption can still produce NaN.
        x_main = x_main.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([x_main, x_pad], dim=-1) if x_pad.shape[-1] > 0 else x_main

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "dummy":
            return x

        scale, offset = self._match_horizon(x)
        feat_dim = scale.shape[-1]
        x_main, x_pad = x[..., :feat_dim], x[..., feat_dim:]
        if self.mode == "tanh":
            x_main = x_main.clamp(-1.0 + self.tanh_eps, 1.0 - self.tanh_eps)
            x_main = torch.atanh(x_main)
        x_main = (x_main - offset) / scale

        if self._tail_c_pos is not None:
            x_main = self._apply_tail_backward(x_main)

        return torch.cat([x_main, x_pad], dim=-1) if x_pad.shape[-1] > 0 else x_main


def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):
    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)


def load_dataset_stats_from_json(file_path: str, try_convert_tensor: bool = True) -> Dict[str, Any]:
    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(data, lambda x: x.to(torch.float32))

    return data


def search_dataset_stats_cache_json(
    cache_dir: str | Path, data_config: DictConfig
) -> Tuple[bool, str | None]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_git_hash() -> Optional[str]:
        repo = Repo(__file__, search_parent_directories=True)
        return repo.head.commit.hexsha

    def to_plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def normalize_str_list(value: Any) -> List[str]:
        value = to_plain(value)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value if item is not None]

    def normalize_transforms(value: Any) -> Any:
        value = to_plain(value)
        if isinstance(value, dict):
            return [value]
        return value

    def normalize_dataset_dirs(cfg: DictConfig) -> Any:
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            return None
        embodiment_datasets = dataset_cfg.get("embodiment_datasets")
        if embodiment_datasets is not None:
            emb_dirs: Dict[str, List[str]] = {}
            for emb, emb_cfg in embodiment_datasets.items():
                dataset_groups = emb_cfg.get("dataset_groups")
                if dataset_groups is None:
                    emb_dirs[emb] = []
                    continue
                dirs: List[str] = []
                for group in dataset_groups:
                    group_dirs = group.get("dataset_dirs")
                    if group_dirs is None:
                        continue
                    dirs.extend(normalize_str_list(group_dirs))
                emb_dirs[emb] = sorted(dirs)
            return emb_dirs

        dataset_dirs = dataset_cfg.get("dataset_dirs")
        return sorted(normalize_str_list(dataset_dirs))

    def normalize_action_state_transforms(cfg: DictConfig) -> Any:
        processor_cfg = cfg.get("processor")
        if processor_cfg is None:
            return None
        embodiment_processors = processor_cfg.get("embodiment_processors")
        if embodiment_processors is not None:
            emb_transforms: Dict[str, Any] = {}
            for emb, emb_cfg in embodiment_processors.items():
                transforms = emb_cfg.get("action_state_transforms")
                emb_transforms[emb] = normalize_transforms(transforms)
            return emb_transforms

        transforms = processor_cfg.get("action_state_transforms")
        return normalize_transforms(transforms)

    signature = {
        "action_size": data_config.dataset.action_size,
        "dataset_dirs": normalize_dataset_dirs(data_config),
        "action_state_transforms": normalize_action_state_transforms(data_config),
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    dataset_hash = hashlib.sha256(signature_json.encode("utf-8"), usedforsecurity=False).hexdigest()

    git_hash = get_git_hash()
    precise_name = f"dataset_stats_{dataset_hash}_{git_hash}.json"
    precise = cache_dir / precise_name
    if precise.exists():
        logger.info(
            f"Found dataset stats cache with precisely matching dataset and git hash: {precise_name}."
        )
        return True, str(precise)

    candidates = sorted(cache_dir.glob(f"dataset_stats_{dataset_hash}_*.json"))
    if not candidates:
        logger.info(f"No dataset stats cache found for dataset hash {dataset_hash}")
        return False, str(precise)  # return precise cache path for saving cache

    picked = candidates[0]
    prefix = f"dataset_stats_{dataset_hash}_"
    picked_git_hash = picked.name[len(prefix) : -5]
    assert picked_git_hash != git_hash
    logger.warning(
        f"Found substitute dataset stats cache {picked.name} which mismatch current git hash {git_hash}."
    )
    return True, str(picked)

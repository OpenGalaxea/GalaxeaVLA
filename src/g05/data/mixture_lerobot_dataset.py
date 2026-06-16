import gc
import logging
import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path

import re

from g05.data import __all__
from g05.utils.common.import_utils import get_obj_from_str
from g05.data.base_lerobot_dataset import BaseLerobotDataset
from g05.data_processor.processor.mixture_processor import MixtureProcessor


def load_embodiment_config(emb_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load and merge embodiment config from a base config file.

    If `config` field is present, load that YAML file as base config,
    then merge with current config (current config takes precedence).
    `type` is extracted from `_target_` in the loaded config if not specified.
    """
    emb_cfg = dict(emb_cfg)

    if "config" not in emb_cfg:
        return emb_cfg

    config_path = emb_cfg.pop("config")
    # Convert to absolute path relative to Hydra's original working directory
    config_path = to_absolute_path(config_path)
    base_cfg = OmegaConf.load(config_path)
    base_cfg = OmegaConf.to_container(base_cfg, resolve=True)

    # Extract type from _target_
    if "type" not in emb_cfg and "_target_" in base_cfg:
        emb_cfg["type"] = base_cfg.pop("_target_")
    else:
        base_cfg.pop("_target_", None)

    # Merge: base_cfg first, then emb_cfg overrides
    merged = {**base_cfg, **emb_cfg}
    return merged


def normalize_dataset_weights(lengths: List[int], given_weights: List[float]) -> List[float]:
    """Normalize raw dataset_group weights without changing total logical length.

    Raw weights define relative sampling preference. For sampling/stat aggregation
    we use normalized weights so:
        sum_i lengths[i] * normalized_weights[i] == sum_i lengths[i]

    Example:
        lengths=[100, 100], raw=[1, 3] -> normalized=[0.5, 1.5]
        effective lengths become [50, 150], preserving total length 200 while
        changing the sampling ratio to 1:3.
    """
    if len(lengths) != len(given_weights):
        raise ValueError("lengths and given_weights must have the same length")
    total_len = sum(lengths)
    denom = sum(l * w for l, w in zip(lengths, given_weights))
    if denom <= 0:
        raise ValueError("Invalid weight normalization denominator (<= 0)")
    k = total_len / denom
    return [k * w for w in given_weights]


def _should_show_loading_bar() -> bool:
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    return (rank in (None, "0")) and (local_rank in (None, "0")) and sys.stderr.isatty()


def _stats_write(message: str, color: Optional[str] = None) -> None:
    try:
        from termcolor import colored

        output = colored(message, color) if color else message
    except Exception:
        output = message

    tqdm.write(output)


class MixtureLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        embodiment_datasets: Dict[str, Any],
        use_weight_normalization: bool,
        action_size: int,
        past_action_size: int,
        val_set_proportion: float,
        is_training_set: bool,
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        use_weight_for_sampling: bool = False,
        n_datasets: Optional[int] = None,
        load_images: Optional[bool] = None,
        in_memory: bool = False,
    ):
        if n_datasets is not None:
            n_datasets = int(n_datasets)
            if n_datasets <= 0:
                raise ValueError(f"n_datasets must be a positive integer or None, got {n_datasets}")

        self.n_datasets = n_datasets
        self.embodiments = []
        self.weights = []
        self.datasets: List[BaseLerobotDataset] = []
        self.embodiments2types = {}

        # Pre-load all embodiment configs once (avoids double YAML I/O for configs with "config" key)
        emb_configs: Dict[str, Any] = {}
        total_groups = 0
        for emb in embodiment_datasets:
            emb_ds_cfg = load_embodiment_config(embodiment_datasets[emb])
            emb_configs[emb] = emb_ds_cfg
            dataset_groups = emb_ds_cfg.get("dataset_groups")
            if dataset_groups is not None:
                total_groups += len(dataset_groups)

        show_loading_bar = _should_show_loading_bar()

        with tqdm(
            total=total_groups,
            desc="Loading datasets",
            dynamic_ncols=True,
            leave=False,
            disable=not show_loading_bar,
        ) as progress:
            for emb in embodiment_datasets:
                emb_ds_cfg = dict(emb_configs[emb])  # shallow copy; .pop() below mutates it
                dataset_groups = emb_ds_cfg.pop("dataset_groups")
                dataset_type = emb_ds_cfg.pop("type")
                if dataset_groups is None:
                    continue

                emb_type_raw = emb_ds_cfg.pop("embodiment_type")
                if not isinstance(emb_type_raw, str):
                    emb_type_raw = str(emb_type_raw)
                self.embodiments2types[emb] = emb_type_raw

                # Per-dataset params with global defaults
                ds_action_size = emb_ds_cfg.pop("action_size", action_size)
                ds_past_action_size = emb_ds_cfg.pop("past_action_size", past_action_size)
                ds_obs_size = emb_ds_cfg.pop("obs_size", obs_size)
                ds_obs_stride_second = emb_ds_cfg.pop("obs_stride_second", obs_stride_second)
                ds_val_set_proportion = emb_ds_cfg.pop("val_set_proportion", val_set_proportion)
                ds_override_fps = emb_ds_cfg.pop("override_fps", None)

                for group_idx, group in enumerate(dataset_groups):
                    weight = group.weight
                    dataset_dirs = list(group.dataset_dirs)
                    original_n_dirs = len(dataset_dirs)
                    if self.n_datasets is not None:
                        if len(dataset_dirs) > self.n_datasets:
                            logger.info(
                                "Limiting dataset_dirs for %s group %s: %s -> %s",
                                emb,
                                group_idx,
                                len(dataset_dirs),
                                self.n_datasets,
                            )
                        dataset_dirs = dataset_dirs[: self.n_datasets]

                    if show_loading_bar:
                        progress.set_postfix_str(
                            f"{emb} g{group_idx} dirs={len(dataset_dirs)}",
                            refresh=False,
                        )
                    else:
                        logger.info(
                            "Loading dataset for embodiment %s group %s with %s dirs%s",
                            emb,
                            group_idx,
                            len(dataset_dirs),
                            (
                                f" (from {original_n_dirs})"
                                if len(dataset_dirs) != original_n_dirs
                                else ""
                            ),
                        )

                    emb_ds_cfg["dataset_dirs"] = dataset_dirs
                    # Global load_images override: mixture-level setting wins over per-embodiment config
                    if load_images is not None:
                        emb_ds_cfg["load_images"] = load_images
                    if in_memory:
                        emb_ds_cfg["in_memory"] = True
                        logger.info(f"[in_memory] Injecting in_memory=True for {emb}")
                    dataset = get_obj_from_str(dataset_type)(
                        **emb_ds_cfg,
                        action_size=ds_action_size,
                        past_action_size=ds_past_action_size,
                        obs_size=ds_obs_size,
                        obs_stride_second=ds_obs_stride_second,
                        val_set_proportion=ds_val_set_proportion,
                        override_fps=ds_override_fps,
                        is_training_set=is_training_set,
                    )
                    self.embodiments.append(emb)
                    self.weights.append(weight)
                    self.datasets.append(dataset)
                    progress.update(1)

        self.is_training_set = bool(is_training_set)
        self.in_memory = in_memory
        self.use_weight_for_sampling = bool(use_weight_for_sampling)
        self.sampling_epoch = 0
        self.actual_lengths = [len(ds) for ds in self.datasets]

        if in_memory:
            total_rows = sum(self.actual_lengths)
            import psutil
            rss_gb = psutil.Process().memory_info().rss / 1024**3
            logger.info(
                f"[in_memory] MixtureLerobotDataset summary: "
                f"{len(self.datasets)} sub-datasets, {total_rows:,} total rows, "
                f"process RSS={rss_gb:.1f} GB"
            )
        self._raw_weights = list(self.weights)

        if use_weight_normalization:
            self.weights = normalize_dataset_weights(self.actual_lengths, self.weights)

        self.effective_lengths = self._compute_effective_lengths()
        self._set_effective_offsets(self.effective_lengths)

    def _set_effective_offsets(self, lengths: List[int]) -> None:
        self.effective_ends = np.cumsum(lengths)
        self.effective_starts = np.concatenate([[0], self.effective_ends[:-1]])

    def _compute_effective_lengths(self) -> List[int]:
        """Return logical sampling lengths for all inner dataset_groups."""
        if not (self.is_training_set and self.use_weight_for_sampling):
            return self.actual_lengths.copy()

        effective_lengths = []
        for dataset_idx, (actual_len, weight) in enumerate(zip(self.actual_lengths, self.weights)):
            if weight <= 0:
                raise ValueError(
                    "use_weight_for_sampling=True requires positive dataset_group weights; "
                    f"dataset_idx={dataset_idx} has weight={weight}."
                )
            effective_lengths.append(max(1, int(actual_len * weight)))
        return effective_lengths

    def set_epoch(self, epoch: int) -> None:
        """Set epoch used by deterministic undersampling permutation."""
        self.sampling_epoch = int(epoch)

    def _map_weighted_local_index(self, dataset_idx: int, local_idx: int) -> int:
        """Map a group-local effective index to a valid real index."""
        actual_len = int(self.actual_lengths[dataset_idx])
        effective_len = int(self.effective_lengths[dataset_idx])
        if actual_len <= 0:
            raise ValueError(f"dataset_idx={dataset_idx} has non-positive actual length: {actual_len}")

        if not (self.is_training_set and self.use_weight_for_sampling):
            return local_idx
        if effective_len >= actual_len:
            return local_idx % actual_len

        # Undersampling must not always take prefix [0, effective_len).
        # Example: actual_len=10, effective_len=4, stride=7, offset=3 maps
        # local_idx 0..3 -> actual_idx [3, 0, 7, 4], a deterministic subset
        # spread across the real dataset. Offset changes with epoch so the
        # subset changes after Trainer advances sampler epoch.
        golden_ratio_conjugate = 0.61803398875
        epoch_offset_multiplier = 1000003
        stride = max(1, int(actual_len * golden_ratio_conjugate))
        while math.gcd(stride, actual_len) != 1:
            stride += 1
            if stride >= actual_len:
                stride = 1
                break
        offset = (self.sampling_epoch * epoch_offset_multiplier + dataset_idx) % actual_len
        return int((local_idx * stride + offset) % actual_len)

    @staticmethod
    def _clear_inner_overfit(dataset: BaseLerobotDataset) -> None:
        for attr in ("_overfit_len", "_overfit_indices"):
            if hasattr(dataset, attr):
                delattr(dataset, attr)

    def _set_overfit_offsets(self, lengths: List[int]) -> None:
        self._overfit_effective_lengths = lengths
        self._overfit_effective_ends = np.cumsum(lengths)
        self._overfit_effective_starts = np.concatenate([[0], self._overfit_effective_ends[:-1]])

    def _disable_overfit(self) -> None:
        for dataset in self.datasets:
            self._clear_inner_overfit(dataset)

        for attr in (
            "_overfit_len",
            "_overfit_effective_lengths",
            "_overfit_effective_ends",
            "_overfit_effective_starts",
            "_overfit_mode",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def all_embodiment_types(self) -> List[str]:
        """Return sorted list of unique embodiment types in this mixture."""
        vals = set(self.embodiments2types.values())
        for v in vals:
            if not isinstance(v, str):
                logger.warning(
                    "embodiment_type value %r (type=%s) is not a str! Full set: %s",
                    v, type(v).__name__, vals, exc_info=True,
                )
        return sorted(str(v) for v in set(self.embodiments2types.values()))

    def enable_overfit(self, n_samples: int, mode: str = "global"):
        """Pin overfit mode to a stable deterministic subset across datasets.

        Args:
            n_samples: Stable samples requested. In ``global`` mode this is the
                total across the whole mixture. In ``per_dataset`` mode this is
                the stable sample count requested for each inner dataset.
            mode: ``global`` or ``per_dataset``.
        """
        target_samples = int(n_samples)
        if target_samples <= 0:
            self._disable_overfit()
            return

        if mode not in {"global", "per_dataset"}:
            raise ValueError(f"Unsupported overfit mode: {mode}")

        self._disable_overfit()

        overfit_lengths = []
        selected = 0

        if mode == "per_dataset":
            for dataset_idx, dataset in enumerate(self.datasets):
                effective_len = int(self.effective_lengths[dataset_idx])
                actual_len = int(self.actual_lengths[dataset_idx])
                take = min(target_samples, effective_len, actual_len)

                if take > 0 and hasattr(dataset, "enable_overfit"):
                    dataset.enable_overfit(take)
                else:
                    self._clear_inner_overfit(dataset)

                overfit_lengths.append(take)
                selected += take
        else:
            target_samples = min(target_samples, int(self.effective_ends[-1]))
            remaining = target_samples

            for dataset_idx, dataset in enumerate(self.datasets):
                effective_len = int(self.effective_lengths[dataset_idx])
                actual_len = int(self.actual_lengths[dataset_idx])
                take = min(remaining, effective_len, actual_len)

                if take > 0 and hasattr(dataset, "enable_overfit"):
                    dataset.enable_overfit(take)
                else:
                    self._clear_inner_overfit(dataset)

                overfit_lengths.append(take)
                remaining -= take
                selected += take
                if remaining <= 0:
                    overfit_lengths.extend([0] * (len(self.datasets) - dataset_idx - 1))
                    break

        if selected < target_samples and mode == "global":
            logger.warning(
                "Mixture overfit requested %s samples but only pinned %s stable samples.",
                target_samples,
                selected,
            )
        elif mode == "per_dataset":
            expected_samples = target_samples * len(self.datasets)
            if selected < expected_samples:
                logger.warning(
                    "Mixture overfit requested %s stable samples per dataset (%s total) but only pinned %s.",
                    target_samples,
                    expected_samples,
                    selected,
                )

        self._overfit_mode = mode
        self._set_overfit_offsets(overfit_lengths)
        self._overfit_len = selected

    def __len__(self):
        if hasattr(self, "_overfit_len"):
            return self._overfit_len
        return int(self.effective_ends[-1])

    def _resolve_index(self, idx: int):
        overfit_active = hasattr(self, "_overfit_len")
        active_ends = self._overfit_effective_ends if overfit_active else self.effective_ends
        active_starts = self._overfit_effective_starts if overfit_active else self.effective_starts
        dataset_idx = int(np.searchsorted(active_ends, idx, side="right"))
        local_idx = int(idx - active_starts[dataset_idx])
        if overfit_active:
            return dataset_idx, local_idx
        return dataset_idx, self._map_weighted_local_index(dataset_idx, local_idx)

    def get_item_with_meta(self, idx: int):
        dataset_idx, local_idx = self._resolve_index(int(idx))
        sample = self.datasets[dataset_idx][local_idx]
        emb_type = self.embodiments2types[self.embodiments[dataset_idx]]
        sample["embodiment"] = emb_type
        sample["embodiment_type"] = emb_type
        existing_locator = sample.get("dataset_locator")
        if existing_locator is None:
            dataset_dirs = getattr(self.datasets[dataset_idx], "dataset_dirs", None)
            dataset_name = dataset_dirs[0] if dataset_dirs else f"dataset_group_{dataset_idx}"
            # Approximation fallback: if the inner dataset did not provide its own precise
            # locator, fall back to the first dataset_dir of the sampled group plus local_idx.
            # This is exact for single-dir groups and usually close for grouped mixtures, but
            # multi-dir groups can still differ by one inner shard.
            sample["dataset_locator"] = f"dataset_dir={dataset_name}, local_idx={local_idx}"

        return sample, dataset_idx, local_idx

    def __getitem__(self, idx):
        sample, dataset_idx, _local_idx = self.get_item_with_meta(int(idx))
        emb_type = self.embodiments2types[self.embodiments[dataset_idx]]

        if "samples" in sample and "action" in sample["samples"] and "proprio" in sample["samples"]:
            sample["samples"]["action"]["embodiment"] = emb_type
            sample["samples"]["proprio"]["embodiment"] = emb_type

        return sample

    def get_dataset_stats(
        self,
        processor: MixtureProcessor,
        only_keys: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        embodiments: Optional[Set[str]] = None,
    ):
        """
        Compute dataset statistics for normalization.

        Args:
            processor: MixtureProcessor containing per-embodiment processors
            only_keys: If provided, only compute stats for specified keys per embodiment.
                       Format: {"galaxea_r1lite": {"action": {"left_arm"}, "state": {"torso"}}}
                       If None, compute stats for all keys.
            embodiments: If provided, only return stats for the requested embodiments.
                         To preserve per-type aggregation semantics, all inner datasets
                         that share the same `embodiment_type` as any requested
                         embodiment are still included in the computation.

        Returns:
            Dict with structure: {embodiment_type: {"action": {...}, "state": {...}}}
        """
        requested_embodiments = None
        requested_types = None
        if embodiments is not None:
            requested_embodiments = set(embodiments)
            unknown_embodiments = requested_embodiments - set(self.embodiments2types.keys())
            if unknown_embodiments:
                raise ValueError(
                    f"Unknown embodiments requested for stats computation: {sorted(unknown_embodiments)}"
                )
            requested_types = {self.embodiments2types[emb] for emb in requested_embodiments}

        stats_by_type = defaultdict(list)
        weights_by_type = defaultdict(list)

        _stats_write(
            f"📊 [Stats] Mixture stats requested for {len(processor.processors)} embodiment processors",
            "cyan",
        )

        for emb, w, ds in zip(self.embodiments, self.weights, self.datasets):
            emb_type = self.embodiments2types[emb]
            if requested_types is not None and emb_type not in requested_types:
                continue

            ds_frames = sum(
                int(getattr(inner_ds, "num_frames", 0)) for inner_ds in ds.multi_dataset._datasets
            )
            ds_episodes = sum(
                int(getattr(inner_ds, "num_episodes", 0)) for inner_ds in ds.multi_dataset._datasets
            )
            _stats_write(
                "🚚 [Stats] Computing {} (type={}, weight={:.4f}, len={}, frames={}, episodes={})".format(
                    emb,
                    emb_type,
                    w,
                    len(ds),
                    ds_frames,
                    ds_episodes,
                ),
                "cyan",
            )
            setattr(ds, "_stats_debug_name", emb)
            setattr(ds, "_stats_debug_type", emb_type)

            emb_only_keys = only_keys.get(emb) if only_keys else None
            stats = ds.get_dataset_stats(processor[emb_type], only_keys=emb_only_keys)

            stats_by_type[emb_type].append(stats)
            weights_by_type[emb_type].append(w)

        # Aggregate stats for each type
        aggregated_stats_by_type = {}
        for emb_type in stats_by_type:
            aggregated_stats_by_type[emb_type] = self._aggregate_weighted_stats(
                weights_by_type[emb_type], stats_by_type[emb_type]
            )

        return aggregated_stats_by_type

    def sync_weights_for_sharding(self):
        """Recompute normalized weights using global actual_lengths across all nodes.

        When dataset sharding splits dirs across nodes, each node sees different
        actual_lengths for the same embodiment/group.  ``normalize_dataset_weights``
        (called in ``__init__``) therefore produces divergent weights per node
        because the normalization factor ``k = sum(L) / sum(L*w)`` depends on
        local lengths.

        This method computes a *global* normalization factor by all-reducing the
        partial sums ``sum(L_i)`` and ``sum(L_i * raw_w_i)`` across all ranks,
        then re-applies it to the raw (pre-normalization) weights.  Because each
        node has a different subset of (embodiment, group) entries (and thus
        different-length weight vectors), we only reduce two scalars — not the
        per-dataset vectors — so alignment is not needed.

        GPUs on the same node share the same dataset, so the all-reduce sum is
        ``global_value * gpus_per_node``, but the factor cancels in the ratio.

        Must be called after ``__init__`` and before any sampling takes place,
        only when ``shard_datasets_by_node`` is ``True``.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        device = torch.cuda.current_device()

        # Partial sums from this node's datasets
        local_total_len = float(sum(self.actual_lengths))
        local_weighted_sum = float(
            sum(l * w for l, w in zip(self.actual_lengths, self._raw_weights))
        )

        # All-reduce to get global sums (factor of gpus_per_node cancels)
        pair = torch.tensor(
            [local_total_len, local_weighted_sum], device=device, dtype=torch.float64
        )
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        global_total, global_weighted = pair.tolist()

        if global_weighted == 0:
            return

        k_global = global_total / global_weighted
        self.weights = [k_global * w for w in self._raw_weights]

        # Recompute effective lengths with the globally-consistent weights
        self.effective_lengths = self._compute_effective_lengths()
        self._set_effective_offsets(self.effective_lengths)

    def cap_length_for_sharding(self):
        """Equalize effective dataset length across nodes by capping to the global min.

        After node-level sharding, different nodes may have very different total
        dataset sizes (e.g. one node gets a giant 27M-frame dir while others have
        ~20M each).  This causes epoch-length imbalance: the overloaded node has
        longer epochs and other nodes must wrap and re-see data.

        This method all-reduces to find the minimum ``len(dataset)`` across all
        nodes, then scales down each local dataset's effective_lengths
        proportionally so that ``len(self)`` matches the global min.  Datasets
        whose effective_length is reduced will randomly sample from their full
        actual_length each epoch (same mechanism as ``use_weight_for_sampling``).

        Must be called after ``sync_weights_for_sharding()`` and before sampler
        creation, only when ``shard_datasets_by_node`` is ``True``.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        device = torch.cuda.current_device()
        local_len = torch.tensor([len(self)], device=device, dtype=torch.long)
        dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
        target_len = int(local_len.item())
        current_len = len(self)

        if target_len >= current_len:
            return  # this node is already at or below the min

        # Different shard bins can still have different logical epoch lengths
        # after dir assignment and weighted sampling. Cap longer bins to the
        # shortest bin so all ranks finish each epoch together. This only shrinks
        # the logical sampling space; it does not unload dirs or reduce init memory.
        scale = target_len / current_len
        logger.info(
            f"[Dataset Sharding] Capping dataset length: {current_len} -> {target_len} "
            f"(scale={scale:.4f})"
        )
        self.effective_lengths = [max(1, int(l * scale)) for l in self.effective_lengths]
        self._set_effective_offsets(self.effective_lengths)

    def set_processor(self, processor: MixtureProcessor):
        for emb, ds in zip(self.embodiments, self.datasets):
            emb_type = self.embodiments2types[emb]
            p = processor[emb_type]
            p.embodiment_type = emb_type
            ds.set_processor(p)

    def get_invalid_sample_report(self, reset: bool = False) -> List[Dict[str, Any]]:
        report = []
        for emb_name, dataset in zip(self.embodiments, self.datasets):
            if not hasattr(dataset, "get_invalid_sample_report"):
                continue
            for record in dataset.get_invalid_sample_report(reset=reset):
                item = dict(record)
                item["embodiment"] = emb_name
                item["embodiment_type"] = self.embodiments2types.get(emb_name)
                report.append(item)
        return report

    def clear_invalid_sample_report(self) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, "clear_invalid_sample_report"):
                dataset.clear_invalid_sample_report()

    @staticmethod
    def _aggregate_weighted_stats(weights: List, stats: List):
        """
        Aggregate multiple dataset stats with the given weights.

        Args:
            weights: List of weights corresponding to each dataset.
            stats: List of stats dicts returned by
                GalaxeaLerobotDataset.get_dataset_stats.
        """
        assert len(weights) == len(stats), "weights and stats must have the same length"
        assert len(weights) > 0, "weights cannot be empty"

        def _weight_view(example: torch.Tensor) -> torch.Tensor:
            """Reshape weights for broadcasting."""
            w = torch.as_tensor(weights, dtype=example.dtype, device=example.device)
            total = w.sum()
            if total.item() == 0:
                raise ValueError("Sum of weights must be greater than zero.")
            w = w / total
            view_shape = [len(weights)] + [1] * (example.dim())
            return w.view(view_shape)

        def _weighted_mean_std(means_list, std_list):
            means = torch.stack(means_list)
            vars = torch.stack([s**2 for s in std_list])
            w_view = _weight_view(means[0])
            weighted_mean = (means * w_view).sum(dim=0)
            weighted_var = (vars + (means - weighted_mean) ** 2) * w_view
            weighted_var = weighted_var.sum(dim=0)
            return weighted_mean, weighted_var.sqrt()

        def _weighted_avg(tensor_list):
            stacked = torch.stack(tensor_list)
            w_view = _weight_view(stacked[0])
            return (stacked * w_view).sum(dim=0)

        aggregated_stats = {"state": defaultdict(dict), "action": defaultdict(dict)}

        for field in ["state", "action"]:
            # Collect all keys across datasets for this field
            keys = set()
            for s in stats:
                keys.update(s[field].keys())

            for key in keys:
                field_stats = [s[field][key] for s in stats]

                # Stepwise min/max: take the extreme values across datasets
                stepwise_min = torch.stack([fs["stepwise_min"] for fs in field_stats]).amin(dim=0)
                stepwise_max = torch.stack([fs["stepwise_max"] for fs in field_stats]).amax(dim=0)

                # Global min/max: same approach as stepwise
                global_min = torch.stack([fs["global_min"] for fs in field_stats]).amin(dim=0)
                global_max = torch.stack([fs["global_max"] for fs in field_stats]).amax(dim=0)

                # Quantiles: approximate with weighted average
                stepwise_q01 = _weighted_avg([fs["stepwise_q01"] for fs in field_stats])
                stepwise_q99 = _weighted_avg([fs["stepwise_q99"] for fs in field_stats])
                global_q01 = _weighted_avg([fs["global_q01"] for fs in field_stats])
                global_q99 = _weighted_avg([fs["global_q99"] for fs in field_stats])

                stepwise_q001 = _weighted_avg([fs["stepwise_q001"] for fs in field_stats])
                stepwise_q999 = _weighted_avg([fs["stepwise_q999"] for fs in field_stats])
                global_q001 = _weighted_avg([fs["global_q001"] for fs in field_stats])
                global_q999 = _weighted_avg([fs["global_q999"] for fs in field_stats])

                stepwise_q0001 = _weighted_avg([fs["stepwise_q0001"] for fs in field_stats])
                stepwise_q9999 = _weighted_avg([fs["stepwise_q9999"] for fs in field_stats])
                global_q0001 = _weighted_avg([fs["global_q0001"] for fs in field_stats])
                global_q9999 = _weighted_avg([fs["global_q9999"] for fs in field_stats])

                stepwise_q00001 = _weighted_avg([fs["stepwise_q00001"] for fs in field_stats])
                stepwise_q99999 = _weighted_avg([fs["stepwise_q99999"] for fs in field_stats])
                global_q00001 = _weighted_avg([fs["global_q00001"] for fs in field_stats])
                global_q99999 = _weighted_avg([fs["global_q99999"] for fs in field_stats])

                # Means/stds: weighted aggregation
                stepwise_mean, stepwise_std = _weighted_mean_std(
                    [fs["stepwise_mean"] for fs in field_stats],
                    [fs["stepwise_std"] for fs in field_stats],
                )
                global_mean, global_std = _weighted_mean_std(
                    [fs["global_mean"] for fs in field_stats],
                    [fs["global_std"] for fs in field_stats],
                )

                aggregated_stats[field][key]["stepwise_min"] = stepwise_min
                aggregated_stats[field][key]["stepwise_max"] = stepwise_max
                aggregated_stats[field][key]["global_min"] = global_min
                aggregated_stats[field][key]["global_max"] = global_max
                aggregated_stats[field][key]["stepwise_q01"] = stepwise_q01
                aggregated_stats[field][key]["stepwise_q99"] = stepwise_q99
                aggregated_stats[field][key]["global_q01"] = global_q01
                aggregated_stats[field][key]["global_q99"] = global_q99
                aggregated_stats[field][key]["stepwise_q001"] = stepwise_q001
                aggregated_stats[field][key]["stepwise_q999"] = stepwise_q999
                aggregated_stats[field][key]["global_q001"] = global_q001
                aggregated_stats[field][key]["global_q999"] = global_q999
                aggregated_stats[field][key]["stepwise_q0001"] = stepwise_q0001
                aggregated_stats[field][key]["stepwise_q9999"] = stepwise_q9999
                aggregated_stats[field][key]["global_q0001"] = global_q0001
                aggregated_stats[field][key]["global_q9999"] = global_q9999
                aggregated_stats[field][key]["stepwise_q00001"] = stepwise_q00001
                aggregated_stats[field][key]["stepwise_q99999"] = stepwise_q99999
                aggregated_stats[field][key]["global_q00001"] = global_q00001
                aggregated_stats[field][key]["global_q99999"] = global_q99999
                aggregated_stats[field][key]["stepwise_mean"] = stepwise_mean
                aggregated_stats[field][key]["stepwise_std"] = stepwise_std
                aggregated_stats[field][key]["global_mean"] = global_mean
                aggregated_stats[field][key]["global_std"] = global_std

        return aggregated_stats

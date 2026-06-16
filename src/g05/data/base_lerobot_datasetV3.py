"""LeRobot V3 dataset base class.

Provides BaseLerobotDatasetV3 with fast statistics computation for LeRobot 3.0
datasets. state/action meta uses a unified explicit raw layout:
- lerobot_key: raw parquet column name
- start_index/raw_shape: slice for this key within the raw column
- time_offset: temporal offset
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from g05.data.base_lerobot_dataset import BaseLerobotDataset

logger = logging.getLogger(__name__)


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _format_ratio(kept: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{100.0 * kept / total:.2f}%"


def _should_show_stats_pbar() -> bool:
    import os

    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    return (rank in (None, "0")) and (local_rank in (None, "0"))


def _stats_write(message: str, color: Optional[str] = None) -> None:
    try:
        from termcolor import colored

        output = colored(message, color) if color else message
    except Exception:
        output = message

    tqdm.write(output)


def fast_quantile_parallel(
    data: np.ndarray, q_values: List[float] = [0.01, 0.99], num_workers: int = 8
) -> Dict[float, np.ndarray]:
    """
    Compute quantiles with np.partition + parallelism, about 15-20x faster than np.quantile.

    Args:
        data: shape = (N, action_size, dim)
        q_values: list of quantile values to compute.
        num_workers: number of parallel workers.

    Returns:
        dict: {q: result}, where result shape = (action_size, dim).
    """
    N, action_size, dim = data.shape

    def compute_for_action_step(i: int) -> Tuple[int, dict]:
        """Compute all-dim quantiles for one action_step."""
        slice_data = data[:, i, :]  # (N, dim)
        result = {}
        for q in q_values:
            k = int(q * (N - 1))
            partitioned = np.partition(slice_data, k, axis=0)
            result[q] = partitioned[k, :]  # (dim,)
        return i, result

    results = {q: np.zeros((action_size, dim), dtype=data.dtype) for q in q_values}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(compute_for_action_step, range(action_size)))
        for i, q_dict in futures:
            for q, vals in q_dict.items():
                results[q][i, :] = vals

    return results


def sliding_window_with_episode_boundary(
    data: torch.Tensor, ep_indices: torch.Tensor, window_size: int
) -> torch.Tensor:
    """
    Vectorized sliding window that replicates at episode boundaries instead of crossing episodes.

    Args:
        data: input data, shape = (N, D).
        ep_indices: episode index per frame, shape = (N,).
        window_size: sliding window size.

    Returns:
        Sliding window result, shape = (N, window_size, D).
    """
    N, D = data.shape

    # Compute each episode's end position.
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_ends = torch.cat([ep_changes, torch.tensor([N])])
    ep_starts = torch.cat([torch.tensor([0]), ep_changes])

    # Find the end position of the containing episode for each frame.
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_end[start:end] = end - 1  # Index of the last frame.

    # Build the sliding window.
    result = torch.zeros(N, window_size, D, dtype=data.dtype)
    for j in range(window_size):
        # Target index.
        target_idx = torch.arange(N) + j
        # Use the episode's last frame when exceeding the episode boundary.
        target_idx = torch.minimum(target_idx, frame_ep_end)
        # Use the dataset's last frame when exceeding the data range.
        target_idx = torch.minimum(target_idx, torch.tensor(N - 1))
        result[:, j] = data[target_idx]

    return result


def shift_sequence_with_episode_boundary(
    data: torch.Tensor, ep_indices: torch.Tensor, offset: int
) -> torch.Tensor:
    """
    Shift a flat per-frame tensor inside episode boundaries.

    Args:
        data: Tensor with shape `(N, D)` or `(N,)`, where `N` is the total
            number of frames across all episodes.
        ep_indices: Tensor with shape `(N,)` indicating which episode each
            frame belongs to.
        offset: Temporal offset applied to the first dimension. Positive values
            mean "use future frames"; negative values mean "use past frames".

    Returns:
        Tensor with the same shape as `data`. For indices that would cross an
        episode boundary, this function clamps to the first/last valid frame of
        that episode, so no values are ever taken from another episode.
    """
    if data.ndim == 1:
        data = data.unsqueeze(-1)

    N, _ = data.shape
    if N == 0:
        return data

    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_ends = torch.cat([ep_changes, torch.tensor([N], dtype=torch.long)])
    ep_starts = torch.cat([torch.tensor([0], dtype=torch.long), ep_changes])

    frame_ep_start = torch.zeros(N, dtype=torch.long)
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_start[start:end] = start
        frame_ep_end[start:end] = end - 1

    indices = torch.arange(N, dtype=torch.long) + offset
    indices = torch.maximum(indices, frame_ep_start)
    indices = torch.minimum(indices, frame_ep_end)
    return data[indices]


def _compute_frame_ep_end(ep_indices: torch.Tensor, N: int) -> torch.Tensor:
    """
    Precompute per-frame episode end index for sliding window operations.

    Args:
        ep_indices: Episode index per frame, shape (N,)
        N: Total number of frames

    Returns:
        Tensor of shape (N,) where frame_ep_end[i] = index of the last frame
        in the episode containing frame i.
    """
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_ends = torch.cat([ep_changes, torch.tensor([N])])
    ep_starts = torch.cat([torch.tensor([0]), ep_changes])
    frame_ep_end = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_end[start:end] = end - 1
    return frame_ep_end


def _compute_frame_ep_start(ep_indices: torch.Tensor, N: int) -> torch.Tensor:
    ep_changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
    ep_starts = torch.cat([torch.tensor([0], dtype=torch.long), ep_changes])
    ep_ends = torch.cat([ep_changes, torch.tensor([N], dtype=torch.long)])
    frame_ep_start = torch.zeros(N, dtype=torch.long)
    for start, end in zip(ep_starts, ep_ends):
        frame_ep_start[start:end] = start
    return frame_ep_start


def _shift_indices_with_episode_boundary(
    indices: torch.Tensor,
    offset: int,
    frame_ep_start: torch.Tensor,
    frame_ep_end: torch.Tensor,
) -> torch.Tensor:
    shifted = indices + offset
    shifted = torch.maximum(shifted, frame_ep_start[indices])
    shifted = torch.minimum(shifted, frame_ep_end[indices])
    return shifted


def _get_episode_metadata_from_parquet(
    parquet_files: List[Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Read episode boundaries from meta/episodes to avoid reading the full episode_index column.

    In LeRobot v3, meta/episodes/*.parquet stores:
    - dataset_from_index: start frame for each episode
    - dataset_to_index: exclusive end frame for each episode

    Returns:
        ep_starts: (num_episodes,) start frame index for each episode, or None on failure.
        ep_ends: (num_episodes,) end frame index for each episode, or None on failure.
        total_frames: total frame count.
    """
    from pathlib import Path

    import pyarrow.parquet as pq

    if not parquet_files:
        return None, None, 0

    # Parquet path: <dataset_root>/data/chunk-XXX/file.parquet.
    # Walk three levels up (file -> chunk -> data -> dataset_root) to get
    # dataset_root, deduplicating while preserving order.
    # Historical bug: looking only at parquet_files[0] metadata treated the first
    # dataset's frame count as global N for multi-dataset mixtures, so later
    # sampled_base_idx only covered the first dataset.
    roots: List[Path] = []
    seen = set()
    for p in parquet_files:
        root = Path(p).parent.parent.parent
        if root not in seen:
            seen.add(root)
            roots.append(root)

    all_starts: List[torch.Tensor] = []
    all_ends: List[torch.Tensor] = []
    offset = 0
    for root in roots:
        meta_dir = root / "meta" / "episodes"
        if not meta_dir.exists():
            logger.debug(
                f"[Stats] meta/episodes not found at {meta_dir}, falling back to full scan"
            )
            return None, None, 0
        episode_files = sorted(meta_dir.rglob("*.parquet"))
        if not episode_files:
            logger.debug(f"[Stats] no parquet files in {meta_dir}, falling back to full scan")
            return None, None, 0
        try:
            ep_table = pq.read_table(
                episode_files, columns=["dataset_from_index", "dataset_to_index"]
            )
        except Exception as e:
            logger.debug(f"[Stats] failed to read {meta_dir}: {e}, falling back to full scan")
            return None, None, 0

        starts = torch.tensor(ep_table["dataset_from_index"].to_numpy(), dtype=torch.long)
        ends = torch.tensor(ep_table["dataset_to_index"].to_numpy(), dtype=torch.long)
        if len(ends) == 0:
            continue
        # Local coordinate range for this dataset: [starts[0], ends[-1]).
        # Shift to global coordinates by subtracting local starts[0] and adding
        # the current offset.
        local_start = int(starts[0].item())
        local_end = int(ends[-1].item())
        shift = offset - local_start
        all_starts.append(starts + shift)
        all_ends.append(ends + shift)
        offset += local_end - local_start

    if not all_ends:
        return None, None, 0

    return torch.cat(all_starts), torch.cat(all_ends), offset


def _get_ep_indices_for_frame_indices(
    frame_indices: torch.Tensor,
    ep_starts: torch.Tensor,
    ep_ends: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the episode index for each given frame index.

    Args:
        frame_indices: frame indices, shape (M,).
        ep_starts: episode start frames, shape (E,).
        ep_ends: episode end frames, shape (E,).

    Returns:
        episode_index: episode index for each frame, shape (M,).
    """
    ep_idx = torch.searchsorted(ep_ends, frame_indices, side="right")
    return ep_idx


def _get_frame_ep_boundaries_for_indices(
    frame_indices: torch.Tensor,
    ep_starts: torch.Tensor,
    ep_ends: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the containing episode's start/end frames for each given frame index.

    Args:
        frame_indices: frame indices, shape (M,).
        ep_starts: episode start frames, shape (E,).
        ep_ends: episode end frames, shape (E,).

    Returns:
        frame_ep_start: start frame of each frame's containing episode, shape (M,).
        frame_ep_end: inclusive end frame of each frame's containing episode, shape (M,).
    """
    ep_idx = torch.searchsorted(ep_ends, frame_indices, side="right")
    frame_ep_start = ep_starts[ep_idx]
    frame_ep_end = ep_ends[ep_idx] - 1
    return frame_ep_start, frame_ep_end


def _collect_required_frame_indices(
    sampled_base_idx: torch.Tensor,
    frame_ep_start: torch.Tensor,
    frame_ep_end: torch.Tensor,
    action_size: int,
    state_meta: List[Dict[str, Any]],
    action_meta: List[Dict[str, Any]],
) -> torch.Tensor:
    required = [sampled_base_idx]

    for meta in state_meta:
        offset = int(meta.get("time_offset", 0))
        if offset != 0:
            required.append(
                _shift_indices_with_episode_boundary(
                    sampled_base_idx,
                    offset,
                    frame_ep_start,
                    frame_ep_end,
                )
            )

    for meta in action_meta:
        offset = int(meta.get("time_offset", 0))
        for j in range(action_size):
            step_idx = torch.minimum(sampled_base_idx + j, frame_ep_end[sampled_base_idx])
            if offset != 0:
                step_idx = _shift_indices_with_episode_boundary(
                    step_idx,
                    offset,
                    frame_ep_start,
                    frame_ep_end,
                )
            required.append(step_idx)

    return torch.unique(torch.cat(required), sorted=True)


def _build_stats_sampling_plan(
    sampled_base_idx: torch.Tensor,
    action_size: int,
    state_meta: List[Dict[str, Any]],
    action_meta: List[Dict[str, Any]],
    ep_starts: Optional[torch.Tensor] = None,
    ep_ends: Optional[torch.Tensor] = None,
    frame_ep_start: Optional[torch.Tensor] = None,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """
    Build the sampling plan, supporting two modes:

    Mode 1 (memory efficient): provide ep_starts/ep_ends and compute frame
    boundaries on demand.
    Mode 2 (compatibility): provide precomputed frame_ep_start/frame_ep_end.

    Args:
        sampled_base_idx: sampling anchor indices.
        action_size: action sequence length.
        state_meta: state metadata list.
        action_meta: action metadata list.
        ep_starts: episode start frame indices, shape (E,).
        ep_ends: episode end frame indices, shape (E,).
        frame_ep_start: start frame of each frame's containing episode, shape (N,).
        frame_ep_end: end frame of each frame's containing episode, shape (N,).

    Returns:
        required_indices: all frame indices that need to be read.
        state_indices_by_key: indices for each state key.
        action_indices_by_key: per-step indices for each action key.
    """
    state_indices_by_key: Dict[str, torch.Tensor] = {}
    action_indices_by_key: Dict[str, List[torch.Tensor]] = {}
    required = [sampled_base_idx]

    use_ep_meta = ep_starts is not None and ep_ends is not None

    def get_boundaries_for_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get episode boundaries for the specified indices."""
        if frame_ep_start is not None and frame_ep_end is not None:
            return frame_ep_start[indices], frame_ep_end[indices]
        else:
            return _get_frame_ep_boundaries_for_indices(indices, ep_starts, ep_ends)

    for meta in state_meta:
        key = meta["key"]
        state_idx = sampled_base_idx
        offset = int(meta.get("time_offset", 0))
        if offset != 0:
            if use_ep_meta:
                fep_start, fep_end = get_boundaries_for_indices(state_idx)
                shifted = state_idx + offset
                shifted = torch.maximum(shifted, fep_start)
                shifted = torch.minimum(shifted, fep_end)
                state_idx = shifted
            else:
                state_idx = _shift_indices_with_episode_boundary(
                    state_idx, offset, frame_ep_start, frame_ep_end
                )
        state_indices_by_key[key] = state_idx
        required.append(state_idx)

    for meta in action_meta:
        key = meta["key"]
        offset = int(meta.get("time_offset", 0))
        per_step_indices = []

        if use_ep_meta:
            _, fep_end_for_base = get_boundaries_for_indices(sampled_base_idx)
        else:
            fep_end_for_base = frame_ep_end[sampled_base_idx]

        for j in range(action_size):
            step_idx = torch.minimum(sampled_base_idx + j, fep_end_for_base)
            if offset != 0:
                if use_ep_meta:
                    fep_start, fep_end = get_boundaries_for_indices(step_idx)
                    shifted = step_idx + offset
                    shifted = torch.maximum(shifted, fep_start)
                    shifted = torch.minimum(shifted, fep_end)
                    step_idx = shifted
                else:
                    step_idx = _shift_indices_with_episode_boundary(
                        step_idx, offset, frame_ep_start, frame_ep_end
                    )
            per_step_indices.append(step_idx)
            required.append(step_idx)
        action_indices_by_key[key] = per_step_indices

    required_indices = torch.unique(torch.cat(required), sorted=True)
    return required_indices, state_indices_by_key, action_indices_by_key


def _load_selected_rows_from_parquet(
    parquet_files: List[Any],
    columns: List[str],
    required_indices: torch.Tensor,
    desc: str,
) -> Dict[str, torch.Tensor]:
    import pyarrow as pa
    import pyarrow.dataset as ds

    required_np = required_indices.cpu().numpy()
    next_required = 0
    total_required = int(required_np.shape[0])
    global_offset = 0
    arrays_by_column: Dict[str, List[np.ndarray]] = {col: [] for col in columns}

    dataset = ds.dataset(parquet_files, format="parquet")
    scanner = dataset.scanner(columns=columns)
    show_pbar = _should_show_stats_pbar()

    with tqdm(
        total=total_required,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not show_pbar,
    ) as progress:
        for batch in scanner.to_batches():
            batch_size = batch.num_rows
            batch_end = global_offset + batch_size

            take_end = next_required
            while take_end < total_required and required_np[take_end] < batch_end:
                take_end += 1

            if take_end > next_required:
                local_indices = required_np[next_required:take_end] - global_offset
                taken = batch.take(pa.array(local_indices, type=pa.int64()))
                for col in columns:
                    col_np = taken.column(col).to_numpy(zero_copy_only=False)
                    arrays_by_column[col].append(col_np)
                progress.update(take_end - next_required)
                next_required = take_end

            global_offset = batch_end
            if next_required >= total_required:
                break

    selected: Dict[str, torch.Tensor] = {}
    for col in columns:
        if arrays_by_column[col]:
            col_np = np.concatenate(arrays_by_column[col], axis=0)
        else:
            col_np = np.empty((0,), dtype=np.float32)
        if col_np.dtype == object:
            col_np = np.stack(col_np)
        if col_np.ndim == 1:
            col_np = col_np[:, np.newaxis]
        selected[col] = torch.tensor(col_np, dtype=torch.float32)

    return selected


def compute_action_stats_columnwise(
    data: torch.Tensor,
    ep_indices: torch.Tensor,
    action_size: int,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-step action stats without materializing the full (N, action_size, D) tensor.

    Instead of creating the sliding window tensor (N × action_size × D), this function
    iterates over each step j in range(action_size) and computes stats on the (N, D) column
    directly. Peak memory per key: O(N × D) instead of O(N × action_size × D).

    Args:
        data: Action data for a single key, shape (N, D)
        ep_indices: Episode index per frame, shape (N,)
        action_size: Sliding window size
        quantile_low: Lower quantile value (default 0.01)
        quantile_high: Upper quantile value (default 0.99)
        frame_ep_end: Precomputed per-frame episode end index. If None, computed internally.

    Returns:
        Dict of stats with stepwise (action_size, D) and global (D,) tensors.
    """
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    data = data.float()
    N, D = data.shape

    if frame_ep_end is None:
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    quantile_spec = [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]

    stepwise_min = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_max = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_mean = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_std = torch.empty(action_size, D, dtype=torch.float32)
    stepwise_quantiles = {
        name: torch.empty(action_size, D, dtype=torch.float32) for name, _ in quantile_spec
    }

    base_idx = torch.arange(N, dtype=torch.long)
    N_limit = torch.tensor(N - 1, dtype=torch.long)

    # Precompute all k-th positions for np.partition
    ks = [max(0, min(int(q_val * (N - 1)), N - 1)) for _, q_val in quantile_spec]

    for j in range(action_size):
        target_idx = base_idx + j
        target_idx = torch.minimum(target_idx, frame_ep_end)
        target_idx = torch.minimum(target_idx, N_limit)
        col_data = data[target_idx]  # (N, D) — only 1× memory

        stepwise_min[j] = col_data.amin(0)
        stepwise_max[j] = col_data.amax(0)
        stepwise_mean[j] = col_data.mean(0)
        stepwise_std[j] = col_data.std(0)

        # Use np.partition with all k values at once for efficiency
        col_np = col_data.numpy()
        partitioned = np.partition(col_np, ks, axis=0)
        for (q_name, _), k in zip(quantile_spec, ks):
            stepwise_quantiles[q_name][j] = torch.from_numpy(partitioned[k].copy()).float()

        del col_data, col_np, partitioned

    # Build stats dict
    # Law of total variance: global_var = E[var_j] + Var(mean_j)
    if stepwise_mean.shape[0] > 1:
        _global_std = (stepwise_std.pow(2).mean(0) + stepwise_mean.var(0)).sqrt()
    else:
        _global_std = stepwise_std.squeeze(0)
    stats = {
        "stepwise_min": stepwise_min,
        "stepwise_max": stepwise_max,
        "stepwise_mean": stepwise_mean,
        "stepwise_std": stepwise_std,
        "global_min": stepwise_min.amin(0),
        "global_max": stepwise_max.amax(0),
        "global_mean": stepwise_mean.mean(0),
        "global_std": _global_std,
    }

    for q_name, _ in quantile_spec:
        stats[f"stepwise_{q_name}"] = stepwise_quantiles[q_name]

    # Global quantiles: lower → amin across steps, upper → amax across steps
    for q_name in ["q01", "q001", "q0001", "q00001"]:
        stats[f"global_{q_name}"] = stats[f"stepwise_{q_name}"].amin(0)
    for q_name in ["q99", "q999", "q9999", "q99999"]:
        stats[f"global_{q_name}"] = stats[f"stepwise_{q_name}"].amax(0)

    return stats


def compute_state_stats(
    data: torch.Tensor,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> Dict[str, torch.Tensor]:
    """
    Compute stats for state data (no sliding window needed).

    Args:
        data: State data for a single key, shape (N, D) or (N,)
        quantile_low: Lower quantile value (default 0.01)
        quantile_high: Upper quantile value (default 0.99)

    Returns:
        Dict of stats with stepwise (1, D) and global (D,) tensors.
    """
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    data = data.float()

    stats = {}
    stats["stepwise_min"] = data.amin(0).unsqueeze(0)
    stats["stepwise_max"] = data.amax(0).unsqueeze(0)
    stats["global_min"] = data.amin(0)
    stats["global_max"] = data.amax(0)
    stats["stepwise_mean"] = data.mean(0).unsqueeze(0)
    stats["stepwise_std"] = data.std(0).unsqueeze(0)
    stats["global_mean"] = data.mean(0)
    stats["global_std"] = data.std(0)

    data_np = data.numpy()
    N = data_np.shape[0]

    quantile_spec = [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]

    # Use np.partition (O(N)) instead of np.quantile (O(N log N))
    ks = [max(0, min(int(q_val * (N - 1)), N - 1)) for _, q_val in quantile_spec]
    partitioned = np.partition(data_np, ks, axis=0)
    for (q_name, _), k in zip(quantile_spec, ks):
        q_result = partitioned[k].copy()
        stats[f"stepwise_{q_name}"] = torch.from_numpy(q_result).unsqueeze(0).float()
        stats[f"global_{q_name}"] = torch.from_numpy(q_result).float()

    del data_np, partitioned
    return stats


def _get_quantile_spec(quantile_low: float = 0.01, quantile_high: float = 0.99):
    """Return standard quantile specification list."""
    return [
        ("q01", quantile_low),
        ("q99", quantile_high),
        ("q001", 0.001),
        ("q999", 0.999),
        ("q0001", 0.0001),
        ("q9999", 0.9999),
        ("q00001", 0.00001),
        ("q99999", 0.99999),
    ]


def _merge_dict_to_tensor(
    data_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[str], List[int]]:
    """Merge dict of tensors into a single (N, D_total) tensor. Returns (merged, keys, dims)."""
    keys = list(data_dict.keys())
    dims = []
    tensors = []
    for k in keys:
        t = data_dict[k].float()
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        dims.append(t.shape[-1])
        tensors.append(t)
    merged = torch.cat(tensors, dim=-1)
    return merged, keys, dims


def _split_stats_by_key(
    keys: List[str],
    dims: List[int],
    stepwise_min: torch.Tensor,
    stepwise_max: torch.Tensor,
    stepwise_mean: torch.Tensor,
    stepwise_std: torch.Tensor,
    stepwise_quantiles: torch.Tensor,
    quantile_spec: List[Tuple[str, float]],
    is_action: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split merged stats tensors back into per-key dicts."""
    result = {}
    offset = 0
    for key, dim in zip(keys, dims):
        s = slice(offset, offset + dim)
        sw_mean = stepwise_mean[:, s]
        sw_std = stepwise_std[:, s]
        # Law of total variance: global_var = E[var_j] + Var(mean_j)
        # When only 1 step (state), var(0) is undefined (N-1=0), use std directly
        if sw_mean.shape[0] > 1:
            global_std = (sw_std.pow(2).mean(0) + sw_mean.var(0)).sqrt()
        else:
            global_std = sw_std.squeeze(0)
        ks_dict = {
            "stepwise_min": stepwise_min[:, s],
            "stepwise_max": stepwise_max[:, s],
            "stepwise_mean": sw_mean,
            "stepwise_std": sw_std,
            "global_min": stepwise_min[:, s].amin(0),
            "global_max": stepwise_max[:, s].amax(0),
            "global_mean": sw_mean.mean(0),
            "global_std": global_std,
        }
        for qi, (qname, _) in enumerate(quantile_spec):
            ks_dict[f"stepwise_{qname}"] = stepwise_quantiles[qi, :, s]
        for qname in ["q01", "q001", "q0001", "q00001"]:
            ks_dict[f"global_{qname}"] = ks_dict[f"stepwise_{qname}"].amin(0)
        for qname in ["q99", "q999", "q9999", "q99999"]:
            ks_dict[f"global_{qname}"] = ks_dict[f"stepwise_{qname}"].amax(0)

        if not is_action:
            # State: stepwise has shape (1, D), global has shape (D,)
            for stat_name in list(ks_dict.keys()):
                if stat_name.startswith("stepwise_"):
                    ks_dict[stat_name] = ks_dict[stat_name][:1]  # keep only first row → (1, D)

        result[key] = ks_dict
        offset += dim
    return result


def compute_action_stats_merged(
    action_dict: Dict[str, torch.Tensor],
    ep_indices: torch.Tensor,
    action_size: int,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute action stats for all keys at once by merging into one tensor.

    Merges all action keys along the last dim, computes stats once (with sliding
    window per step), then splits results back per key. Uses GPU (torch.sort for
    quantiles) if available, falls back to CPU (np.partition).

    Peak GPU memory: ~3 × N × D_total × 4 bytes (merged + col_data + sorted copy).
    GPU memory is fully released after computation via del + empty_cache.

    Args:
        action_dict: {key: Tensor(N, D_key)}
        ep_indices: Episode index per frame (N,)
        action_size: Sliding window size
        quantile_low/high: Quantile boundaries
        frame_ep_end: Precomputed per-frame episode end index

    Returns:
        {key: {stat_name: Tensor}} per action key
    """
    if not action_dict:
        return {}

    merged, keys, dims = _merge_dict_to_tensor(action_dict)
    N, D_total = merged.shape

    if frame_ep_end is None:
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)
    num_q = len(quantile_spec)
    ks = [max(0, min(int(qv * (N - 1)), N - 1)) for _, qv in quantile_spec]

    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            merged_dev = merged.to(device)
            fep_dev = frame_ep_end.to(device)
            base_idx = torch.arange(N, dtype=torch.long, device=device)
            N_limit = torch.tensor(N - 1, dtype=torch.long, device=device)

            sw_min = torch.empty(action_size, D_total, device=device)
            sw_max = torch.empty(action_size, D_total, device=device)
            sw_mean = torch.empty(action_size, D_total, device=device)
            sw_std = torch.empty(action_size, D_total, device=device)
            sw_q = torch.empty(num_q, action_size, D_total, device=device)

            for j in range(action_size):
                tidx = torch.minimum(base_idx + j, fep_dev)
                tidx = torch.minimum(tidx, N_limit)
                col = merged_dev[tidx]

                sw_min[j] = col.amin(0)
                sw_max[j] = col.amax(0)
                sw_mean[j] = col.mean(0)
                sw_std[j] = col.std(0)

                sorted_col, _ = col.sort(dim=0)
                for qi in range(num_q):
                    sw_q[qi, j] = sorted_col[ks[qi]]
                del col, sorted_col

            # Move all to CPU at once (one sync point)
            stepwise_min = sw_min.cpu()
            stepwise_max = sw_max.cpu()
            stepwise_mean = sw_mean.cpu()
            stepwise_std = sw_std.cpu()
            stepwise_quantiles = sw_q.cpu()

            del merged, merged_dev, fep_dev, base_idx, N_limit
            del sw_min, sw_max, sw_mean, sw_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            # merged (CPU) is still alive for fallback
            torch.cuda.empty_cache()

    if not computed:
        # CPU fallback with np.partition
        if not merged.is_contiguous():
            merged = merged.contiguous()
        base_idx = torch.arange(N, dtype=torch.long)
        N_limit = torch.tensor(N - 1, dtype=torch.long)

        stepwise_min = torch.empty(action_size, D_total)
        stepwise_max = torch.empty(action_size, D_total)
        stepwise_mean = torch.empty(action_size, D_total)
        stepwise_std = torch.empty(action_size, D_total)
        stepwise_quantiles = torch.empty(num_q, action_size, D_total)

        for j in range(action_size):
            tidx = torch.minimum(base_idx + j, frame_ep_end)
            tidx = torch.minimum(tidx, N_limit)
            col = merged[tidx]

            stepwise_min[j] = col.amin(0)
            stepwise_max[j] = col.amax(0)
            stepwise_mean[j] = col.mean(0)
            stepwise_std[j] = col.std(0)

            col_np = col.numpy()
            partitioned = np.partition(col_np, ks, axis=0)
            for qi in range(num_q):
                stepwise_quantiles[qi, j] = torch.from_numpy(partitioned[ks[qi]].copy()).float()
            del col, col_np, partitioned
        del merged

    return _split_stats_by_key(
        keys,
        dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=True,
    )


def compute_action_stats_with_transforms(
    action_dict: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
    ep_indices: torch.Tensor,
    action_size: int,
    transforms: Optional[list] = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    frame_ep_end: Optional[torch.Tensor] = None,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 1,
    show_progress: bool = False,
    action_step_indices: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute action stats with transforms applied correctly per sliding-window column.

    Relative transforms (RelativeJointTransform, RelativePoseTransform) use
    ``state[..., -1:, :]`` to select the base state. On flat (N, D) data this
    resolves to the last frame of the *entire dataset* — which is wrong. By
    unsqueezing to (N, 1, D) before applying transforms, the indexing correctly
    gives per-frame state.

    Fast path: if transforms is None/empty, delegates to compute_action_stats_merged().

    Args:
        action_dict: {key: Tensor(N, D_key)} raw action data per key
        state_dict: {key: Tensor(N, D_key)} raw state data per key
        ep_indices: Episode index per frame (N,)
        action_size: Sliding window size
        transforms: List of transform objects (e.g. RelativeJointTransform,
            RelativePoseTransform, PoseRotationTransform). None or empty = no transforms.
        quantile_low/high: Quantile boundaries
        frame_ep_end: Precomputed per-frame episode end index

    Returns:
        {key: {stat_name: Tensor}} per action key (keys/dims may differ from input
        if a transform changes dimensionality, e.g. PoseRotationTransform 7→9).
    """
    transforms = transforms or []

    # Fast path: no transforms → delegate
    if not transforms and action_step_indices is None:
        return compute_action_stats_merged(
            action_dict,
            ep_indices,
            action_size,
            quantile_low,
            quantile_high,
            frame_ep_end,
        )

    if action_step_indices is not None:
        first_key = next(iter(action_step_indices))
        N = action_step_indices[first_key][0].shape[0]
    elif state_dict:
        N = next(iter(state_dict.values())).shape[0]
    else:
        N = next(iter(action_dict.values())).shape[0]

    if frame_ep_end is None and action_step_indices is None:
        frame_ep_end = _compute_frame_ep_end(ep_indices, N)

    # --- Discover output keys and dims via a trial transform on frame 0 ---
    if action_step_indices is None:
        trial_action = {k: v[0:1].unsqueeze(1).float() for k, v in action_dict.items()}  # (1,1,D)
    else:
        trial_action = {
            k: v[action_step_indices[k][0][:1]].unsqueeze(1).float() for k, v in action_dict.items()
        }
    trial_state = {k: v[0:1].unsqueeze(1).float() for k, v in state_dict.items()}  # (1,1,D)
    trial_batch = {"action": trial_action, "state": trial_state}
    for trans in transforms:
        trial_batch = trans.forward(trial_batch)
    out_keys = list(trial_batch["action"].keys())
    out_dims = [trial_batch["action"][k].shape[-1] for k in out_keys]
    D_total = sum(out_dims)
    del trial_action, trial_state, trial_batch

    # --- Prepare state as float for transform reuse across columns ---
    state_float = {k: v.float() for k, v in state_dict.items()}

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)
    num_q = len(quantile_spec)

    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            # Move state to GPU once (reused every column)
            state_gpu = {k: v.to(device) for k, v in state_float.items()}
            # Move action to GPU once
            action_gpu = {k: v.to(device).float() for k, v in action_dict.items()}
            if action_step_indices is None:
                fep_dev = frame_ep_end.to(device)
                base_idx = torch.arange(N, dtype=torch.long, device=device)
                N_limit = torch.tensor(N - 1, dtype=torch.long, device=device)
            else:
                fep_dev = None
                base_idx = None
                N_limit = None
                action_step_indices_dev = {
                    key: [idx.to(device) for idx in step_indices]
                    for key, step_indices in action_step_indices.items()
                }

            sw_min = torch.empty(action_size, D_total, device=device)
            sw_max = torch.empty(action_size, D_total, device=device)
            sw_mean = torch.empty(action_size, D_total, device=device)
            sw_std = torch.empty(action_size, D_total, device=device)
            sw_q = torch.empty(num_q, action_size, D_total, device=device)

            step_iter = tqdm(
                range(action_size),
                desc="🎬 Action stats (GPU)",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            for j in step_iter:
                if show_progress:
                    step_iter.set_postfix_str(f"step={j + 1}/{action_size}", refresh=False)

                if action_step_indices is None:
                    steps = torch.arange(j + 1, device=device)
                    all_tidx = base_idx.unsqueeze(0) + steps.unsqueeze(1)
                    all_tidx = torch.minimum(all_tidx, fep_dev.unsqueeze(0))
                    all_tidx = torch.minimum(all_tidx, N_limit)
                    action_col_3d = {k: v[all_tidx].permute(1, 0, 2) for k, v in action_gpu.items()}
                else:
                    action_col_3d = {
                        k: torch.stack(
                            [v[action_step_indices_dev[k][step]] for step in range(j + 1)], dim=1
                        )
                        for k, v in action_gpu.items()
                    }
                # State: unsqueeze to (N, 1, D_k) so [..., -1:, :] → [:, -1:, :] = per-frame
                state_3d = {k: v.unsqueeze(1) for k, v in state_gpu.items()}

                # Apply transforms (shallow dict copy for safety)
                batch = {"action": action_col_3d, "state": dict(state_3d)}
                action_trans_start = time.perf_counter()
                for trans in transforms:
                    batch = trans.forward(batch)
                if not show_progress:
                    logger.info(
                        "🧩 Applied transforms for action step %s/%s in %.2fs",
                        j + 1,
                        action_size,
                        time.perf_counter() - action_trans_start,
                    )

                # Take last frame of output for stats
                col_parts = [batch["action"][k][:, -1, :] for k in out_keys]
                if downsample_rate > 1:
                    col_parts = [part[::downsample_rate] for part in col_parts]
                col = torch.cat(col_parts, dim=-1)

                sw_min[j] = col.amin(0)
                sw_max[j] = col.amax(0)
                sw_mean[j] = col.mean(0)
                sw_std[j] = col.std(0)

                action_before_quantile = time.perf_counter()

                # Recompute quantile indices for downsampled data
                N_sampled = col.shape[0]
                ks_q_sampled = [
                    max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
                ]

                if quantile_method == "numpy_partition":
                    col_np = col.cpu().numpy()
                    partitioned = np.partition(col_np, ks_q_sampled, axis=0)
                    for qi in range(num_q):
                        sw_q[qi, j] = torch.from_numpy(partitioned[ks_q_sampled[qi]]).to(device)
                    del col_np, partitioned
                elif quantile_method == "torch_quantile":
                    quantile_values = torch.tensor([qv for _, qv in quantile_spec], device=device)
                    sw_q[:, j] = torch.quantile(col, quantile_values, dim=0)
                elif quantile_method == "torch_sort":
                    sorted_col, _ = col.sort(dim=0)
                    for qi in range(num_q):
                        sw_q[qi, j] = sorted_col[ks_q_sampled[qi]]
                    del sorted_col
                else:
                    raise ValueError(f"Unknown quantile_method: {quantile_method}")

                if not show_progress:
                    logger.info(
                        "📐 Computed action quantiles (%s) for step %s/%s in %.2fs",
                        quantile_method,
                        j + 1,
                        action_size,
                        time.perf_counter() - action_before_quantile,
                    )
                del col, action_col_3d, state_3d, batch, col_parts

            # Move all to CPU at once
            stepwise_min = sw_min.cpu()
            stepwise_max = sw_max.cpu()
            stepwise_mean = sw_mean.cpu()
            stepwise_std = sw_std.cpu()
            stepwise_quantiles = sw_q.cpu()

            del action_gpu, state_gpu
            if fep_dev is not None:
                del fep_dev, base_idx, N_limit
            if action_step_indices is not None:
                del action_step_indices_dev
            del sw_min, sw_max, sw_mean, sw_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    if not computed:
        # CPU fallback with np.partition
        if action_step_indices is None:
            base_idx = torch.arange(N, dtype=torch.long)
            N_limit = torch.tensor(N - 1, dtype=torch.long)
        else:
            base_idx = None
            N_limit = None

        stepwise_min = torch.empty(action_size, D_total)
        stepwise_max = torch.empty(action_size, D_total)
        stepwise_mean = torch.empty(action_size, D_total)
        stepwise_std = torch.empty(action_size, D_total)
        stepwise_quantiles = torch.empty(num_q, action_size, D_total)

        # Ensure action is float for transforms
        action_float = {k: v.float() for k, v in action_dict.items()}

        step_iter = tqdm(
            range(action_size),
            desc="🎬 Action stats (CPU)",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for j in step_iter:
            if show_progress:
                step_iter.set_postfix_str(f"step={j + 1}/{action_size}", refresh=False)
            if action_step_indices is None:
                steps = torch.arange(j + 1)
                all_tidx = base_idx.unsqueeze(0) + steps.unsqueeze(1)
                all_tidx = torch.minimum(all_tidx, frame_ep_end.unsqueeze(0))
                all_tidx = torch.minimum(all_tidx, N_limit)
                action_col_3d = {k: v[all_tidx].permute(1, 0, 2) for k, v in action_float.items()}
            else:
                action_col_3d = {
                    k: torch.stack(
                        [v[action_step_indices[k][step]] for step in range(j + 1)], dim=1
                    )
                    for k, v in action_float.items()
                }
            state_3d = {k: v.unsqueeze(1) for k, v in state_float.items()}

            batch = {"action": action_col_3d, "state": dict(state_3d)}
            for trans in transforms:
                batch = trans.forward(batch)

            col_parts = [batch["action"][k][:, -1, :] for k in out_keys]
            if downsample_rate > 1:
                col_parts = [part[::downsample_rate] for part in col_parts]
            col = torch.cat(col_parts, dim=-1)

            stepwise_min[j] = col.amin(0)
            stepwise_max[j] = col.amax(0)
            stepwise_mean[j] = col.mean(0)
            stepwise_std[j] = col.std(0)

            # Recompute quantile indices for downsampled data
            N_sampled = col.shape[0]
            ks_q_sampled = [
                max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
            ]

            if quantile_method == "numpy_partition":
                col_np = col.numpy()
                partitioned = np.partition(col_np, ks_q_sampled, axis=0)
                for qi in range(num_q):
                    stepwise_quantiles[qi, j] = torch.from_numpy(
                        partitioned[ks_q_sampled[qi]].copy()
                    ).float()
                del col_np, partitioned
            elif quantile_method == "torch_quantile":
                quantile_values = torch.tensor([qv for _, qv in quantile_spec])
                stepwise_quantiles[:, j] = torch.quantile(col, quantile_values, dim=0)
            elif quantile_method == "torch_sort":
                sorted_col, _ = col.sort(dim=0)
                for qi in range(num_q):
                    stepwise_quantiles[qi, j] = sorted_col[ks_q_sampled[qi]]
                del sorted_col
            else:
                raise ValueError(f"Unknown quantile_method: {quantile_method}")

            del col, action_col_3d, state_3d, batch, col_parts

    return _split_stats_by_key(
        out_keys,
        out_dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=True,
    )


def compute_state_stats_merged(
    state_dict: Dict[str, torch.Tensor],
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute state stats for all keys at once by merging into one tensor.

    Uses GPU (torch.sort) if available, CPU (np.partition) as fallback.
    No sliding window needed for state.

    Args:
        state_dict: {key: Tensor(N, D_key)}
        quantile_low/high: Quantile boundaries

    Returns:
        {key: {stat_name: Tensor}} per state key
    """
    if not state_dict:
        return {}

    merged, keys, dims = _merge_dict_to_tensor(state_dict)
    N, D_total = merged.shape

    quantile_spec = _get_quantile_spec(quantile_low, quantile_high)

    use_gpu = torch.cuda.is_available()
    computed = False

    if use_gpu:
        try:
            device = torch.device("cuda")
            merged_dev = merged.to(device)

            # Downsample for stats computation
            if downsample_rate > 1:
                merged_dev = merged_dev[::downsample_rate]

            s_min = merged_dev.amin(0)
            s_max = merged_dev.amax(0)
            s_mean = merged_dev.mean(0)
            s_std = merged_dev.std(0)

            # Recompute quantile indices for downsampled data
            N_sampled = merged_dev.shape[0]
            ks_sampled = [
                max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
            ]

            if quantile_method == "numpy_partition":
                merged_np = merged_dev.cpu().numpy()
                partitioned = np.partition(merged_np, ks_sampled, axis=0)
                sw_q = torch.stack([torch.from_numpy(partitioned[k]) for k in ks_sampled]).to(
                    device
                )
                del merged_np, partitioned, merged_dev
            elif quantile_method == "torch_quantile":
                quantile_values = torch.tensor([qv for _, qv in quantile_spec], device=device)
                sw_q = torch.quantile(merged_dev, quantile_values, dim=0)
                del merged_dev
            elif quantile_method == "torch_sort":
                sorted_dev, _ = merged_dev.sort(dim=0)
                sw_q = torch.stack([sorted_dev[k] for k in ks_sampled])
                del merged_dev, sorted_dev
            else:
                raise ValueError(f"Unknown quantile_method: {quantile_method}")

            # Move to CPU
            stepwise_min = s_min.cpu().unsqueeze(0)
            stepwise_max = s_max.cpu().unsqueeze(0)
            stepwise_mean = s_mean.cpu().unsqueeze(0)
            stepwise_std = s_std.cpu().unsqueeze(0)
            stepwise_quantiles = sw_q.cpu().unsqueeze(1)  # (num_q, 1, D_total)

            del merged, s_min, s_max, s_mean, s_std, sw_q
            torch.cuda.empty_cache()
            computed = True
        except torch.cuda.OutOfMemoryError:
            # merged (CPU) is still alive for fallback
            torch.cuda.empty_cache()

    if not computed:
        # Downsample for stats computation
        if downsample_rate > 1:
            merged = merged[::downsample_rate]

        stepwise_min = merged.amin(0).unsqueeze(0)
        stepwise_max = merged.amax(0).unsqueeze(0)
        stepwise_mean = merged.mean(0).unsqueeze(0)
        stepwise_std = merged.std(0).unsqueeze(0)

        # Recompute quantile indices for downsampled data
        N_sampled = merged.shape[0]
        ks_sampled = [
            max(0, min(int(qv * (N_sampled - 1)), N_sampled - 1)) for _, qv in quantile_spec
        ]

        if quantile_method == "numpy_partition":
            merged_np = merged.numpy()
            partitioned = np.partition(merged_np, ks_sampled, axis=0)
            q_rows = [torch.from_numpy(partitioned[k].copy()).float() for k in ks_sampled]
            stepwise_quantiles = torch.stack(q_rows).unsqueeze(1)  # (num_q, 1, D_total)
            del merged_np, partitioned
        elif quantile_method == "torch_quantile":
            quantile_values = torch.tensor([qv for _, qv in quantile_spec])
            sw_q = torch.quantile(merged, quantile_values, dim=0)
            stepwise_quantiles = sw_q.unsqueeze(1)
        elif quantile_method == "torch_sort":
            sorted_merged, _ = merged.sort(dim=0)
            q_rows = [sorted_merged[k] for k in ks_sampled]
            stepwise_quantiles = torch.stack(q_rows).unsqueeze(1)
            del sorted_merged
        else:
            raise ValueError(f"Unknown quantile_method: {quantile_method}")

        del merged

    return _split_stats_by_key(
        keys,
        dims,
        stepwise_min,
        stepwise_max,
        stepwise_mean,
        stepwise_std,
        stepwise_quantiles,
        quantile_spec,
        is_action=False,
    )


def compute_state_stats_with_transforms(
    state_dict: Dict[str, torch.Tensor],
    transforms: Optional[list] = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    quantile_method: Literal["numpy_partition", "torch_quantile", "torch_sort"] = "numpy_partition",
    downsample_rate: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute state stats after applying processor transforms to state.

    This matches the legacy stats path, where ``preprocessor.action_state_transform``
    runs before both state/action stats are collected. Transforms that only operate
    on action (e.g. RelativeJointTransform, RelativePoseTransform) become no-ops
    when only ``state`` is present in the batch, while state-affecting transforms
    (e.g. PoseRotationTransform on ee_pose) still update output dimensionality.
    """
    transforms = transforms or []

    if not state_dict:
        return {}

    if not transforms:
        return compute_state_stats_merged(
            state_dict,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            quantile_method=quantile_method,
            downsample_rate=downsample_rate,
        )

    state_batch = {"state": {k: v.unsqueeze(1).float() for k, v in state_dict.items()}}
    for trans in transforms:
        state_batch = trans.forward(state_batch)

    transformed_state_dict = {k: v.squeeze(1) for k, v in state_batch["state"].items()}
    return compute_state_stats_merged(
        transformed_state_dict,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        quantile_method=quantile_method,
        downsample_rate=downsample_rate,
    )


class BaseLerobotDatasetV3(BaseLerobotDataset):
    """
    LeRobot V3 dataset base class.

    Provides a fast get_dataset_stats implementation that reads all parquet files
    once to compute statistics. The current code uses a unified explicit raw layout:
    - lerobot_key: raw parquet column name
    - start_index/raw_shape: slice for this key within the raw column
    - time_offset: temporal offset
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int,
        past_action_size: int = 0,
        obs_size: int = 1,
        obs_stride_second: float = 0.0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        lerobot_ds_version: Optional[Literal["2.1", "3.0"]] = "3.0",
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        tolerance_s: Optional[float] = None,
        fast_stats_computation: bool = True,
        stats_downsample_rate: int = 1,
        **kwargs,
    ):
        super().__init__(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            action_size=action_size,
            past_action_size=past_action_size,
            obs_size=obs_size,
            obs_stride_second=obs_stride_second,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            lerobot_ds_version=lerobot_ds_version,
            tolerance_s=tolerance_s,
            **kwargs,
        )
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.stats_downsample_rate = int(stats_downsample_rate)
        if self.stats_downsample_rate <= 0:
            raise ValueError(
                f"stats_downsample_rate must be a positive integer, got {stats_downsample_rate}"
            )

        if fast_stats_computation:
            # Override get_dataset_stats with optimized version
            self.get_dataset_stats = self._fast_get_dataset_stats

    def _get_parquet_columns(self, only_keys: Optional[Dict[str, Set[str]]] = None) -> List[str]:
        """Return the unique raw parquet columns referenced by state/action metas."""
        columns = []
        for category, metas in (("state", self.state_meta), ("action", self.action_meta)):
            allowed_keys = None if only_keys is None else only_keys.get(category, set())
            for meta in metas:
                if allowed_keys is not None and meta["key"] not in allowed_keys:
                    continue
                col_names = meta["lerobot_key"]
                if not isinstance(col_names, (list, tuple)):
                    col_names = [col_names]
                for col_name in col_names:
                    if col_name not in columns:
                        columns.append(col_name)
        return columns

    def _fast_get_dataset_stats(
        self,
        _preprocessor=None,
        apply_processor: bool = True,
        use_fast_quantile: bool = True,
        num_workers: int = 32,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Optimized version using merged tensors and GPU acceleration for global stats.

        The only difference from V3 is the statistics computation stage:
        - State: compute_state_stats_merged() — merges all keys and computes once
        - Action: compute_action_stats_with_transforms() — column-wise sliding window + transforms
        The data read stage (pyarrow parquet) is identical to V3.

        Args:
            _preprocessor: optional processor used to apply action_state_transforms.
            apply_processor: whether to apply processor action_state_transforms; defaults to True.
            use_fast_quantile: unused; kept for V3 API compatibility.
            num_workers: unused; kept for V3 API compatibility.
            only_keys: when provided, only return stats for the specified keys.
                       Format: {"action": {"left_arm", "right_arm"}, "state": {"torso"}}.
                       Note: all stats are still computed internally because the
                       merged-tensor path is more efficient; filtering happens only
                       on return.
        """
        total_start = time.time()
        dataset_label = getattr(self, "_stats_debug_name", None)
        dataset_type = getattr(self, "_stats_debug_type", None)
        if dataset_label is not None:
            if dataset_type is not None:
                _stats_write(
                    f"🪪 [Stats] Dataset label={dataset_label} (type={dataset_type})",
                    "cyan",
                )
            else:
                _stats_write(f"🪪 [Stats] Dataset label={dataset_label}", "cyan")
        _stats_write("🚀 [Stats] Starting online stats computation...", "cyan")
        _stats_write(f"🔧 [Stats] apply_processor={apply_processor}", "cyan")

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

        state_meta_to_use = [m for m in self.state_meta if m["key"] in state_keys_to_compute]
        action_meta_to_use = [m for m in self.action_meta if m["key"] in action_keys_to_compute]

        columns = self._get_parquet_columns(only_keys=only_keys)

        # Collect all parquet file paths.
        parquet_files = []
        total_frames = 0
        total_episodes = 0
        for dataset in self.multi_dataset._datasets:
            data_dir = dataset.root / "data"
            parquet_files.extend(sorted(data_dir.rglob("*.parquet")))
            total_frames += int(getattr(dataset, "num_frames", 0))
            total_episodes += int(getattr(dataset, "num_episodes", 0))

        _stats_write(
            "📊 [Stats] Dataset has {} frames, {} episodes, {} parquet files, len(dataset)={}".format(
                _format_count(total_frames),
                _format_count(total_episodes),
                _format_count(len(parquet_files)),
                _format_count(len(self)),
            ),
            "cyan",
        )
        _stats_write(
            "🎯 [Stats] stats_downsample_rate={} -> base anchors ~{}".format(
                self.stats_downsample_rate,
                _format_count(
                    (total_frames + self.stats_downsample_rate - 1) // self.stats_downsample_rate
                ),
            ),
            "yellow",
        )

        read_start = time.time()

        # Prefer reading episode boundaries from meta/episodes for memory efficiency.
        ep_starts, ep_ends, N_from_meta = _get_episode_metadata_from_parquet(parquet_files)

        if ep_starts is not None and N_from_meta > 0:
            _stats_write(
                "📥 [Stats] Using episode metadata for boundary computation (memory efficient)",
                "blue",
            )
            N = N_from_meta
            sampled_base_idx = torch.arange(0, N, self.stats_downsample_rate, dtype=torch.long)

            required_indices, state_indices_by_key, action_indices_by_key = (
                _build_stats_sampling_plan(
                    sampled_base_idx,
                    self.action_size,
                    state_meta_to_use,
                    action_meta_to_use,
                    ep_starts=ep_starts,
                    ep_ends=ep_ends,
                )
            )
            _stats_write(
                "🧮 [Stats] Sample plan: {} base anchors -> {} required raw frames ({} of full data)".format(
                    _format_count(sampled_base_idx.numel()),
                    _format_count(required_indices.numel()),
                    _format_ratio(required_indices.numel(), N),
                ),
                "yellow",
            )

            _stats_write(
                f"📦 [Stats] Loading only required raw rows for {len(columns)} columns...",
                "blue",
            )
            process_start = time.time()
            raw_column_cache = _load_selected_rows_from_parquet(
                parquet_files,
                columns,
                required_indices,
                desc="📥 Loading sampled stats rows",
            )
            # Derive episode_index for required_indices from episode boundaries.
            selected_ep_indices = _get_ep_indices_for_frame_indices(
                required_indices, ep_starts, ep_ends
            )
        else:
            # Fallback: read episode_index from parquet for old formats or small datasets.
            _stats_write(
                "📥 [Stats] Reading episode_index column to build sampling plan...", "blue"
            )
            import pyarrow.dataset as ds

            dataset = ds.dataset(parquet_files, format="parquet")
            ep_table = dataset.to_table(columns=["episode_index"])
            all_ep_indices = torch.tensor(ep_table["episode_index"].to_numpy(), dtype=torch.long)
            N = int(all_ep_indices.shape[0])
            frame_ep_start = _compute_frame_ep_start(all_ep_indices, N)
            frame_ep_end = _compute_frame_ep_end(all_ep_indices, N)
            sampled_base_idx = torch.arange(0, N, self.stats_downsample_rate, dtype=torch.long)

            required_indices, state_indices_by_key, action_indices_by_key = (
                _build_stats_sampling_plan(
                    sampled_base_idx,
                    self.action_size,
                    state_meta_to_use,
                    action_meta_to_use,
                    frame_ep_start=frame_ep_start,
                    frame_ep_end=frame_ep_end,
                )
            )
            _stats_write(
                "🧮 [Stats] Sample plan: {} base anchors -> {} required raw frames ({} of full data)".format(
                    _format_count(sampled_base_idx.numel()),
                    _format_count(required_indices.numel()),
                    _format_ratio(required_indices.numel(), N),
                ),
                "yellow",
            )

            _stats_write(
                f"📦 [Stats] Loading only required raw rows for {len(columns)} columns...",
                "blue",
            )
            process_start = time.time()
            raw_column_cache = _load_selected_rows_from_parquet(
                parquet_files,
                columns,
                required_indices,
                desc="📥 Loading sampled stats rows",
            )
            selected_ep_indices = all_ep_indices[required_indices]
            del all_ep_indices, frame_ep_start, frame_ep_end
            gc.collect()
        # required_indices is generated by torch.unique(sorted=True), so it is ordered.
        # Use searchsorted instead of Python dict + list comprehension to avoid OOM
        # from dictionaries with millions of entries.
        sampled_local_idx = torch.searchsorted(required_indices, sampled_base_idx)
        state_local_indices_by_key = {
            key: torch.searchsorted(required_indices, raw_indices)
            for key, raw_indices in state_indices_by_key.items()
        }
        action_local_indices_by_key = {
            key: [
                torch.searchsorted(required_indices, raw_indices)
                for raw_indices in per_step_indices
            ]
            for key, per_step_indices in action_indices_by_key.items()
        }
        gc.collect()
        _stats_write(
            "✅ [Stats] Built sampled tensors in {:.2f}s (read plan {:.2f}s, row materialization {:.2f}s)".format(
                time.time() - read_start,
                process_start - read_start,
                time.time() - process_start,
            ),
            "green",
        )

        state_dict = {}
        _stats_write("🧠 [Stats] Building sampled state/action tensors...", "blue")
        for meta in state_meta_to_use:
            key = meta["key"]
            if key is None:
                continue
            state = self._slice_meta_feature(
                self._get_meta_source_data(raw_column_cache, meta), meta
            )
            state_dict[key] = state[state_local_indices_by_key[key]]

        action_dict = {}
        for meta in action_meta_to_use:
            key = meta["key"]
            if key is None:
                continue
            action = self._slice_meta_feature(
                self._get_meta_source_data(raw_column_cache, meta), meta
            )
            action_dict[key] = action

        # ── Statistics computation stage (V4 optimization) ─────────────────────

        stats_start = time.time()

        # Extract transforms from processor (if any)
        transforms = None
        if apply_processor and _preprocessor is not None:
            if (
                hasattr(_preprocessor, "action_state_transforms")
                and _preprocessor.action_state_transforms
            ):
                transforms = _preprocessor.action_state_transforms
                _stats_write(
                    f"🧩 [Stats] Will apply {len(transforms)} action_state_transforms",
                    "magenta",
                )
            else:
                _stats_write("🧩 [Stats] No action_state_transforms to apply", "magenta")

        state_stats_start = time.time()
        _stats_write(
            "📐 [Stats] Computing state stats for {} keys on {} sampled rows...".format(
                len(state_dict),
                _format_count(sampled_local_idx.numel()),
            ),
            "blue",
        )
        state_stats = compute_state_stats_with_transforms(
            state_dict,
            transforms=transforms,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            downsample_rate=1,
        )
        _stats_write(
            f"✅ [Stats] Computed state stats in {time.time() - state_stats_start:.2f}s",
            "green",
        )

        action_stats_start = time.time()
        _stats_write(
            "🎬 [Stats] Computing action stats for {} keys, action_size={}, sampled anchors={}...".format(
                len(action_dict),
                self.action_size,
                _format_count(sampled_local_idx.numel()),
            ),
            "blue",
        )
        action_stats = compute_action_stats_with_transforms(
            action_dict,
            state_dict,
            selected_ep_indices,
            self.action_size,
            transforms=transforms,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            downsample_rate=1,
            show_progress=_should_show_stats_pbar(),
            action_step_indices=action_local_indices_by_key,
        )
        _stats_write(
            f"✅ [Stats] Computed action stats in {time.time() - action_stats_start:.2f}s",
            "green",
        )

        stats = {"state": state_stats, "action": action_stats}

        if only_keys is not None:
            filtered_stats = {"state": {}, "action": {}}
            for category in ["action", "state"]:
                keys_to_keep = only_keys.get(category, set())
                for key in keys_to_keep:
                    if key in stats[category]:
                        filtered_stats[category][key] = stats[category][key]
            stats = filtered_stats

        gc.collect()
        _stats_write(f"🏁 [Stats] Computed all stats in {time.time() - stats_start:.2f}s", "green")
        _stats_write(
            f"🏁 [Stats] Total get_dataset_stats time: {time.time() - total_start:.2f}s",
            "green",
        )
        return stats

    def get_dataset_stats(
        self,
        _preprocessor=None,
        apply_processor: bool = True,
        use_fast_quantile: bool = True,
        num_workers: int = 32,
        only_keys: Optional[Dict[str, Set[str]]] = None,
    ):
        return self._fast_get_dataset_stats(
            _preprocessor=_preprocessor,
            apply_processor=apply_processor,
            use_fast_quantile=use_fast_quantile,
            num_workers=num_workers,
            only_keys=only_keys,
        )


# unit test for quantile speed comparison
def _compare_quantile_methods(
    N: int = 1_000_000,
    action_size: int = 32,
    dim: int = 7,
    num_workers: int = 8,
    repeat: int = 3,
):
    """
    Compare speed and accuracy between np.quantile and fast_quantile_parallel.

    Args:
        N: number of data points.
        action_size: action chunk size.
        dim: action dimension.
        num_workers: number of parallel workers.
        repeat: repeat count.
    """
    import time

    print("=" * 70)
    print(f"Quantile method comparison (N={N:,}, action_size={action_size}, dim={dim})")
    print("=" * 70)

    # Generate synthetic robot action data in [0, 1].
    np.random.seed(42)
    data = np.random.rand(N, action_size, dim).astype(np.float32)
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")

    # Original method.
    print("\n[Original method] np.quantile ...")
    times_orig = []
    for i in range(repeat):
        start = time.perf_counter()
        orig_q01 = np.quantile(data, 0.01, axis=0)
        orig_q99 = np.quantile(data, 0.99, axis=0)
        elapsed = time.perf_counter() - start
        times_orig.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
    avg_orig = np.mean(times_orig)
    print(f"  Average time: {avg_orig:.3f}s")

    # Fast method.
    print(f"\n[Fast method] fast_quantile_parallel (workers={num_workers}) ...")
    times_fast = []
    for i in range(repeat):
        start = time.perf_counter()
        fast_results = fast_quantile_parallel(data, [0.01, 0.99], num_workers)
        fast_q01 = fast_results[0.01]
        fast_q99 = fast_results[0.99]
        elapsed = time.perf_counter() - start
        times_fast.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
    avg_fast = np.mean(times_fast)
    print(f"  Average time: {avg_fast:.3f}s")

    # Compute differences.
    q01_diff = np.abs(orig_q01 - fast_q01)
    q99_diff = np.abs(orig_q99 - fast_q99)
    q01_pct = np.where(np.abs(orig_q01) > 1e-10, q01_diff / np.abs(orig_q01) * 100, 0)
    q99_pct = np.where(np.abs(orig_q99) > 1e-10, q99_diff / np.abs(orig_q99) * 100, 0)

    # Result summary.
    print("\n" + "=" * 70)
    print("Result summary")
    print("=" * 70)
    print(f"Speedup: {avg_orig / avg_fast:.1f}x")
    print("\nq01 accuracy:")
    print(f"  Max absolute difference: {q01_diff.max():.2e}")
    print(f"  Mean absolute difference: {q01_diff.mean():.2e}")
    print(f"  Max percentage error: {q01_pct.max():.4f}%")
    print(f"  Mean percentage error: {q01_pct.mean():.4f}%")
    print("\nq99 accuracy:")
    print(f"  Max absolute difference: {q99_diff.max():.2e}")
    print(f"  Mean absolute difference: {q99_diff.mean():.2e}")
    print(f"  Max percentage error: {q99_pct.max():.4f}%")
    print(f"  Mean percentage error: {q99_pct.mean():.4f}%")
    print(f"\nOutput shape: q01={fast_q01.shape}, q99={fast_q99.shape}")


if __name__ == "__main__":
    _compare_quantile_methods(
        N=1_000_000,
        action_size=32,
        dim=7,
        num_workers=8,
        repeat=3,
    )

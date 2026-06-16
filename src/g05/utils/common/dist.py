import itertools
import math
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_is_rank_sharded = bool(getattr(self.dataset, "is_rank_sharded", False))
        if self._dataset_is_rank_sharded:
            # Dataset already performed rank sharding, so sampler should operate in local mode.
            self.num_replicas = 1
            self.rank = 0
            self.num_samples = len(self.dataset)
            self.total_size = self.num_samples
        self.batch_size = batch_size
        self.start_batch_idx = 0  # per-rank dataloader batch index
        self._log_indices = False
        self._logger = None

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def __iter__(self):
        # super().__iter__() already returns the per-rank index stream (shuffled + padded)
        base_iter = super().__iter__()
        # Skip indices corresponding to already-consumed batches WITHOUT loading data
        skip = self.start_batch_idx * self.batch_size
        if skip > 0:
            base_iter = itertools.islice(base_iter, skip, None)
        return base_iter


class ResumableDistributedGroupedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for datasets with pre-defined group index ranges.
    It guarantees all indices within a batch come from the same group.

    Required dataset attribute:
        group_ranges: Dict[int, Tuple[int, int]], where each tuple is [start, end).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_batch_idx = 0

        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {self.batch_size}.")
        if self.num_replicas <= 0:
            raise ValueError(f"`num_replicas` must be > 0, got {self.num_replicas}.")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"`rank` must be in [0, {self.num_replicas}), got {self.rank}.")
        if not hasattr(self.dataset, "group_ranges"):
            raise TypeError(
                "Dataset must expose `group_ranges: Dict[int, Tuple[int, int]]` for grouped batching."
            )
        self._dataset_is_rank_sharded = bool(getattr(self.dataset, "is_rank_sharded", False))
        if self._dataset_is_rank_sharded:
            self.effective_num_replicas = 1
            self.effective_rank = 0
        else:
            self.effective_num_replicas = self.num_replicas
            self.effective_rank = self.rank

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch_idx: int):
        self.start_batch_idx = int(start_batch_idx)

    def _build_group_batches(self) -> List[List[int]]:
        group_ranges: Dict[int, Tuple[int, int]] = getattr(self.dataset, "group_ranges")
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        group_batches: List[List[int]] = []
        for group_id in sorted(group_ranges):
            start, end = group_ranges[group_id]
            if end <= start:
                continue

            indices = list(range(start, end))
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]

            if self.drop_last:
                total_size = (
                    len(indices) // self.effective_num_replicas
                ) * self.effective_num_replicas
                if total_size <= 0:
                    continue
                indices = indices[:total_size]
            else:
                total_size = (
                    math.ceil(len(indices) / self.effective_num_replicas)
                    * self.effective_num_replicas
                )
                if total_size > len(indices):
                    pad_size = total_size - len(indices)
                    # If pad_size > len(indices), we need to repeat indices to fully pad to total_size.
                    pad = (indices * math.ceil(pad_size / len(indices)))[:pad_size]
                    indices = indices + pad

            rank_indices = indices[self.effective_rank : total_size : self.effective_num_replicas]
            for i in range(0, len(rank_indices), self.batch_size):
                batch = rank_indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                group_batches.append(batch)

        if self.shuffle and len(group_batches) > 1:
            perm = torch.randperm(len(group_batches), generator=generator).tolist()
            group_batches = [group_batches[i] for i in perm]
        return group_batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build_group_batches()
        if self.start_batch_idx > 0:
            batches = batches[self.start_batch_idx :]
        yield from batches

    def __len__(self) -> int:
        group_ranges: Dict[int, Tuple[int, int]] = getattr(self.dataset, "group_ranges")
        total_batches = 0
        for start, end in group_ranges.values():
            n = max(0, end - start)
            if n == 0:
                continue
            if self.drop_last:
                total_size = (n // self.effective_num_replicas) * self.effective_num_replicas
                per_rank = total_size // self.effective_num_replicas
                total_batches += per_rank // self.batch_size
            else:
                total_size = (
                    math.ceil(n / self.effective_num_replicas) * self.effective_num_replicas
                )
                per_rank = total_size // self.effective_num_replicas
                total_batches += math.ceil(per_rank / self.batch_size)
        return max(0, total_batches - self.start_batch_idx)

from typing import List, DefaultDict, Dict, Any
import numpy as np

import torch

from galaxea_fm.data import __all__
from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset
from galaxea_fm.processors.mixture_processor import MixtureProcessor


def normalize_dataset_weights(lengths: List[int], given_weights: List[float]) -> List[float]:
    """
    sum_i (w_i * L_i) = sum_i L_i
    """
    assert len(lengths) == len(given_weights)
    total_len = sum(lengths)
    denom = sum(l * w for l, w in zip(lengths, given_weights))
    assert denom > 0
    k = total_len / denom
    normalized = [k * w for w in given_weights]
    return normalized


class MixtureLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        embodiment_datasets: Dict[str, Any], 
        use_weight_normalization: bool, 
        action_size: int, 
        past_action_size: int, 
        obs_size: int, 
        val_set_proportion: float, 
        is_training_set: bool, 
    ):
        self.embodiments = []
        self.weights = []
        self.datasets: List[BaseLerobotDataset] = []
        for emb in embodiment_datasets:
            emb_ds_cfg = embodiment_datasets[emb]
            dataset_groups = emb_ds_cfg.pop("dataset_groups")
            dataset_type = emb_ds_cfg.pop("type")
            if dataset_groups is None:
                continue
            
            for group in dataset_groups:
                weight = group.weight
                emb_ds_cfg["dataset_dirs"] = group.dataset_dirs
                dataset = __all__[dataset_type](
                    **emb_ds_cfg,
                    action_size=action_size,
                    past_action_size=past_action_size,
                    obs_size=obs_size,
                    val_set_proportion=val_set_proportion,
                    is_training_set=is_training_set,
                )
                self.embodiments.append(emb)
                self.weights.append(weight)
                self.datasets.append(dataset)

        self.is_training_set = is_training_set
        self.actual_lengths = [len(ds) for ds in self.datasets]

        if self.is_training_set:
            if use_weight_normalization:
                self.weights = normalize_dataset_weights(self.actual_lengths, self.weights)
            self.effective_lengths = [int(l * w) for l, w in zip(self.actual_lengths, self.weights)]
        else:
            self.weights = [1.0] * len(self.datasets)
            self.effective_lengths = self.actual_lengths

        self.effective_ends = np.cumsum(self.effective_lengths)
        self.effective_starts = np.concatenate([[0], self.effective_ends[:-1]])

    def __len__(self):
        return self.effective_ends[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.effective_ends, idx, side="right")
        if self.is_training_set:
            local_idx = np.random.randint(0, self.actual_lengths[dataset_idx])
        else:
            local_idx = idx - self.effective_starts[dataset_idx]
        sample = self.datasets[dataset_idx][local_idx]
        sample["embodiment"] = self.embodiments[dataset_idx]
        
        return sample
    
    def get_dataset_stats(self, processor: MixtureProcessor):
        emb_stats = DefaultDict(list)
        emb_weights = DefaultDict(list)
        for emb, w, ds in zip(self.embodiments, self.weights, self.datasets):
            emb_stats[emb].append(ds.get_dataset_stats(processor[emb]))
            emb_weights[emb].append(w)
                
        for emb in emb_stats:
            aggregated_stats = self._aggregate_weighted_stats(emb_weights[emb], emb_stats[emb])
            emb_stats[emb] = aggregated_stats
        
        return emb_stats

    def set_processor(self, processor: MixtureProcessor):
        for emb, ds in zip(self.embodiments, self.datasets):
            ds.set_processor(processor[emb])

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
            vars = torch.stack([s ** 2 for s in std_list])
            w_view = _weight_view(means[0])
            weighted_mean = (means * w_view).sum(dim=0)
            weighted_var = (vars + (means - weighted_mean) ** 2) * w_view
            weighted_var = weighted_var.sum(dim=0)
            return weighted_mean, weighted_var.sqrt()

        def _weighted_avg(tensor_list):
            stacked = torch.stack(tensor_list)
            w_view = _weight_view(stacked[0])
            return (stacked * w_view).sum(dim=0)

        aggregated_stats = {"state": DefaultDict(dict), "action": DefaultDict(dict)}

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
                aggregated_stats[field][key]["stepwise_mean"] = stepwise_mean
                aggregated_stats[field][key]["stepwise_std"] = stepwise_std
                aggregated_stats[field][key]["global_mean"] = global_mean
                aggregated_stats[field][key]["global_std"] = global_std

        return aggregated_stats


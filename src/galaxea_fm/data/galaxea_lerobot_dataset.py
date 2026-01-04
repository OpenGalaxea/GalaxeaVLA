import torch
import numpy as np
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset


class GalaxeaLerobotDataset(BaseLerobotDataset):

    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int, 
        past_action_size: int = 0,
        obs_size: int = 1,

        # signal
        ee_start_moving_thresh: float = 0.0,

        # train vs val
        val_set_proportion: float = 0.05, 
        is_training_set: bool = False,
    ):
        super().__init__(
            dataset_dirs=dataset_dirs, 
            shape_meta=shape_meta, 
            action_size=action_size, 
            past_action_size=past_action_size, 
            obs_size=obs_size, 
            val_set_proportion=val_set_proportion, 
            is_training_set=is_training_set
        )

        self.ee_start_moving_thresh = ee_start_moving_thresh
        if self.ee_start_moving_thresh > 1e-6:
            self.ee_pose_action_meta = [meta for meta in self.action_meta if "ee_pose" in meta["key"]]
            assert len(self.ee_pose_action_meta) > 0, "ee_start_moving_thresh is set but ee_pose is not in action_meta"
            self._get_ee_start_moving_step(max_workers=1)

    def _get_ee_start_moving_step_of_episode(self, episode_idx: int) -> int:
        episode_data: Dict[str, Any] = self._get_episode_data(episode_idx)
        ee_pose_action_meta: List[Dict[str, Any]] = [
            meta for meta in self.action_meta if "ee_pose" in meta.get("key", "")
        ]

        all_movement_distances = []
        for meta in ee_pose_action_meta:
            key = meta["key"]
            
            actions: torch.Tensor = episode_data["action"].get(key)
            
            if actions is None:
                continue
                
            # position_a: (T, 3) | position_b: (T, 3)
            position_a = actions[:, 0, :3]
            position_b = actions[:, 1, :3]
            
            difference_vector = position_a - position_b
            movement_distance = torch.linalg.norm(difference_vector, dim=1)
            
            all_movement_distances.append(movement_distance)

        if not all_movement_distances:
            return self.episode_data_index["to"][episode_idx] - 1

        total_movement = torch.stack(all_movement_distances).sum(dim=0)
        
        indices_of_movement = torch.nonzero(
            total_movement > self.ee_start_moving_thresh, 
            as_tuple=False
        ).squeeze(1)

        if indices_of_movement.numel() > 0:
            first_movement_relative_index = indices_of_movement[0].item()
            absolute_index = self.episode_data_index["from"][episode_idx] + first_movement_relative_index
            return absolute_index
        else:
            return self.episode_data_index["to"][episode_idx] - 1

    def _get_ee_start_moving_step(self, max_workers: Optional[int] = None):        
        tasks = list(range(len(self.episode_data_index["from"])))
        
        from_moving_step_idxs = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(self._get_ee_start_moving_step_of_episode, episode_idx): episode_idx 
                for episode_idx in tasks
            }
            
            for future in tqdm(as_completed(future_to_episode), total=len(future_to_episode), desc="Calculating from_moving_step indices"):
                episode_idx = future_to_episode[future]
                try:
                    result = future.result()
                    from_moving_step_idxs[episode_idx] = result
                except Exception as exc:
                    print(f'Episode {episode_idx} generated an exception: {exc}')
                    from_moving_step_idxs[episode_idx] = self.episode_data_index["from"][episode_idx]

        self.episode_data_index["from_moving_step"] = torch.tensor(from_moving_step_idxs)

        total_length = 0
        for i in range(len(self.episode_data_index["from"])):
            from_moving_idx = self.episode_data_index["from_moving_step"][i]
            to_idx = self.episode_data_index["to"][i]
            total_length += to_idx - from_moving_idx
        self.dataset_len = total_length

        self._episode_lengths = self.episode_data_index["to"] - self.episode_data_index["from_moving_step"]
        self._cumulative_lengths = torch.cat([
            torch.tensor([0]), 
            torch.cumsum(self._episode_lengths, dim=0)
        ])
        
    def get_original_index(self, idx):
        if not 'from_moving_step' in self.episode_data_index:
            return idx
        
        if not hasattr(self, '_episode_lengths'):
            return idx
        
        episode_idx = torch.searchsorted(self._cumulative_lengths, idx, right=True) - 1
        
        offset = idx - self._cumulative_lengths[episode_idx]
        original_idx = self.episode_data_index["from_moving_step"][episode_idx] + offset
        
        return original_idx.item()

    def __len__(self):
        if 'from_moving_step' in self.episode_data_index:
            return self.dataset_len
        else:
            return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        sample["coarse_task"] = lerobot_sample["coarse_task"]
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        original_idx = self.get_original_index(idx)
        sample = super().__getitem__(original_idx)

        return sample
    
    def get_init_positions(self):
        self._set_return_images(False)
        init_positions = {}
        for meta in self.init_state_meta:
            init_positions[meta["key"]] = []
        first_frame_indices = self.episode_data_index["from"].numpy().tolist()
        for index in tqdm(first_frame_indices, desc="Processing first frames for init positions", leave=False):
            for meta in self.init_state_meta:
                key = meta["key"]
                init_positions[key].append(self.multi_dataset[index][f"observation.state.{key}"][0])
        for key, val in init_positions.items():
            init_positions[key] = np.mean(val, axis=0)
        self._set_return_images(True)
        return init_positions


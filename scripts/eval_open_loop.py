from __future__ import annotations

import os

from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import rootutils
import torch

from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

# Add the project root directory to the Python path
rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from galaxea_fm.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.utils.pytorch_utils import dict_apply, dict_to_array, set_global_seed
from galaxea_fm.utils.visualize import plot_result
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.processors.base_processor import BaseProcessor


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed"):
        set_global_seed(cfg.seed, get_worker_init_fn=False)
    
    output_dir = Path(os.path.abspath(os.path.expanduser(cfg.output_dir)))
    output_dir.mkdir(exist_ok=True)
    print(f"Output dir: {output_dir}")

    # load model
    model: GalaxeaZeroPolicy = instantiate(cfg.model.model_arch)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    # HACK: ignore normalizer keys for testing using v1.0.0 checkpoints
    model.load_state_dict(state_dict, strict=False)
    policy = model.cuda().eval()
    
    dataset_val: GalaxeaLerobotDataset = instantiate(cfg.data, is_training_set=False)

    dataloader = DataLoader(
        dataset_val, 
        shuffle=False, 
        batch_size=cfg.batch_size_val, 
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=None, 
    )
    # NOTE: use pretrained norm stats
    checkpoint_path = Path(cfg.ckpt_path)
    dataset_stats = load_dataset_stats_from_json(checkpoint_path.parent.parent / "dataset_stats.json")
    processor: BaseProcessor = instantiate(cfg.model.processor)

    processor.set_normalizer_from_stats(dataset_stats)
    dataset_val.set_processor(processor)
    
    episode_from = dataset_val.episode_data_index["from"]
    episode_to = dataset_val.episode_data_index["to"]
    num_episodes = len(episode_from)

    if cfg.get("eval_episodes_num"):
        eval_episodes_num = cfg.eval_episodes_num
    else:
        eval_episodes_num = num_episodes
    eval_end_frame = episode_to[eval_episodes_num - 1]

    gt_actions = []
    pd_actions = []
    for i, batch in tqdm(enumerate(dataloader), desc="inferencing", total=len(dataloader)):
        batch = dict_apply(batch, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
        with torch.no_grad():
            batch = policy.predict_action(batch)

        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        batch = processor.postprocess(batch)
        cur_pd_action = dict_apply(batch["action"], lambda x: x.cpu().numpy())
        cur_gt_action = dict_apply(batch["gt_action"], lambda x: x.cpu().numpy())

        pd_actions.append(dict_to_array(cur_pd_action))
        gt_actions.append(dict_to_array(cur_gt_action))
        if i * cfg.batch_size_val >= eval_end_frame:
            break

    pd_actions = np.concatenate(pd_actions, axis=0)
    gt_actions = np.concatenate(gt_actions, axis=0)[:, 0, :]


    for idx in range(eval_episodes_num):
        cur_path = output_dir / f"{idx:06}"
        cur_path.mkdir(exist_ok=True)
        cur_pd_action = pd_actions[episode_from[idx]: episode_to[idx]]
        cur_gt_action = gt_actions[episode_from[idx]: episode_to[idx]]
        plot_result(cur_path, cur_gt_action, cur_pd_action)


if __name__ == "__main__":
    main()

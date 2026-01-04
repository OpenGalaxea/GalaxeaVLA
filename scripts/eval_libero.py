import logging
import math
import os
import pathlib
from pathlib import Path

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Optional

import hydra
import numpy as np
import rootutils
import torch

from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

# Add the project root directory to the Python path
rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.utils.pytorch_utils import dict_apply, dict_to_array, set_global_seed
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.processors.policy_processor import BasePolicyProcessor

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)

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
    # HACK: ignore normalizer keys for testing using v1.0.0 checkpoints
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
    policy = model.cuda().eval()

    # NOTE: use pretrained norm stats
    checkpoint_path = Path(cfg.ckpt_path)
    dataset_stats = load_dataset_stats_from_json(checkpoint_path.parent.parent / "dataset_stats.json")
    processor: BasePolicyProcessor = instantiate(cfg.model.processor)

    processor.set_normalizer_from_stats(dataset_stats)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.libero.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {cfg.libero.task_suite_name}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.libero.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif cfg.libero.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif cfg.libero.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif cfg.libero.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif cfg.libero.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {cfg.libero.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(cfg.libero.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            
            pbar = tqdm(total=max_steps + cfg.libero.num_steps_wait, desc=f"Episode {task_episodes + 1}")
            while t < max_steps + cfg.libero.num_steps_wait:
                pbar.update(1)
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.libero.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                sample = { # 
                    # torch.Size([1, 256, 256, 3]) to torch.Size([1,3, 256, 256])
                    "images": {
                            "image": torch.from_numpy(np.expand_dims(
                            img, axis=0
                        )).permute(0, 3, 1, 2),  # (H, W, C), dtype=unit8, range(0-255)
                            "wrist_image": torch.from_numpy(np.expand_dims(
                            wrist_img, axis=0
                        )).permute(0, 3, 1, 2),  # (H, W, C)
                    },
                    "state": {
                        "default": torch.from_numpy(np.expand_dims(state, axis=0)).to(torch.float32),
                    },
                    "task": str(task_description),
                    "state_is_pad": torch.tensor([False]),
                    "image_is_pad": torch.tensor([False]),
                    "action_is_pad": torch.tensor([False]*32),
                    "idx": torch.tensor(0),
                }

                if processor is not None:
                    sample = processor.preprocess(sample)

                batch = dict_apply(sample, lambda x: x.unsqueeze(0).cuda() if isinstance(x, torch.Tensor) else x)
                
                with torch.no_grad():
                    batch = policy.predict_action(batch)
                
                batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
                batch = processor.postprocess(batch)
                cur_pd_action = dict_apply(batch["action"], lambda x: x.cpu().numpy())

                action = cur_pd_action['default'][0][0]
                action[-1] = _binarize_gripper_open(action[-1])

                full_actions.append(action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1
            pbar.close()

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(output_dir)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from tqdm import tqdm

from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.eval.eval_utils import filter_embodiment

from experiments.libero.libero_eval_utils import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION as DEFAULT_ENV_RESOLUTION,
    LiberoGripperCommandState,
    action_dict_gripper_scalar,
    extract_libero_images,
    single_action_to_libero_action,
    build_libero_raw_obs,
    ensure_libero_config,
    ensure_libero_import,
    get_max_steps,
    save_rollout_video,
)
from experiments.libero.libero_vector_env import LiberoAsyncVectorEnv
from scripts.utils.policy_ws_client import MultiPolicyWSClient

register_default_resolvers()

logger = logging.getLogger(__name__)


def _json_default(obj: Any):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _cfg_get(cfg, path: str, default=None):
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def _make_env_fn(task, resolution: int, seed: int | None):
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    def _fn():
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(seed)
        return env

    return _fn


def _make_dummy_env_fn(task, resolution: int, seed: int | None):
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    def _fn():
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
            has_renderer=False,
        )
        env.seed(seed)
        return env

    return _fn


async def run_parallel_task(
    task_id: int,
    task_description: str,
    initial_states: list,
    num_trials: int,
    max_steps: int,
    num_steps_wait: int,
    num_parallel: int,
    client: MultiPolicyWSClient,
    vec_env: LiberoAsyncVectorEnv,
    embodiment_type: str | None,
    save_videos: bool = False,
    output_dir: Path = Path("outputs/libero_parallel_eval"),
) -> dict[str, Any]:
    """Run parallel evaluation.

    Logic per iteration:
      1. If all active envs are in dummy phase, step them with dummy actions.
      2. Otherwise, for each ready env:
         - If cache has actions -> consume locally (instant).
         - If needs recompute -> send request to server.
      3. Server decides when to batch-infer (max_batch_size OR max_wait_ms).
         Client just awaits the response.
      4. Step all envs that have actions.
    """
    trial_successes: dict[int, bool] = {}
    trial_infer_calls: dict[int, int] = {}
    trial_replay_images: dict[int, list] = {}
    completed = 0
    next_trial_idx = 0

    env_trial_id: list[int] = [0] * num_parallel
    env_step_count: list[int] = [0] * num_parallel
    env_action_step: list[int] = [0] * num_parallel
    env_infer_calls: list[int] = [0] * num_parallel
    env_replay: list[list] = [[] for _ in range(num_parallel)]

    def finish_episode(env_idx: int, success: bool):
        nonlocal next_trial_idx, completed
        tid = env_trial_id[env_idx]
        if tid < 0 or tid in trial_successes:
            return None
        trial_successes[tid] = success
        trial_infer_calls[tid] = env_infer_calls[env_idx]
        if save_videos:
            trial_replay_images[tid] = list(env_replay[env_idx])
        completed += 1
        env_step_count[env_idx] = 0
        env_action_step[env_idx] = 0
        env_infer_calls[env_idx] = 0
        env_replay[env_idx] = []
        client.clear_chunk(env_idx)
        if next_trial_idx < num_trials:
            _, state = assign_trial(env_idx)
            return state
        else:
            env_trial_id[env_idx] = -1
            return None

    def assign_trial(env_idx: int) -> tuple[int, Any]:
        nonlocal next_trial_idx
        trial_id = next_trial_idx
        state_idx = trial_id % len(initial_states)
        next_trial_idx += 1
        env_trial_id[env_idx] = trial_id
        env_step_count[env_idx] = 0
        env_action_step[env_idx] = 0
        env_infer_calls[env_idx] = 0
        env_replay[env_idx] = []
        return trial_id, initial_states[state_idx]

    init_states = []
    for i in range(num_parallel):
        _, state = assign_trial(i)
        init_states.append(state)

    await asyncio.to_thread(vec_env.reset)
    obs_batch = await asyncio.to_thread(vec_env.set_init_state_each, init_states)
    per_env_obs: list[dict | None] = list(obs_batch)
    gripper_states = [LiberoGripperCommandState.from_obs(obs) for obs in per_env_obs]

    await client.reset_all()

    total_steps = max_steps + num_steps_wait
    global_progress = tqdm(
        total=num_trials * total_steps,
        desc=f"Task {task_id}",
        leave=False,
    )

    while completed < num_trials:
        if next_trial_idx >= num_trials and all(env_trial_id[i] < 0 for i in range(num_parallel)):
            break

        # --- Handle episodes that exceeded max steps ---
        for i in range(num_parallel):
            if env_trial_id[i] >= 0 and env_step_count[i] >= total_steps:
                await client.reset_connection(i)
                new_state = finish_episode(i, False)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(
                        vec_env.set_init_state_single, i, new_state
                    )
                    gripper_states[i].reset_from_obs(per_env_obs[i])

        active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
        if not active_envs:
            continue

        # --- Phase: dummy steps (all envs still in warmup) ---
        all_in_dummy = all(env_step_count[i] < num_steps_wait for i in active_envs)
        if all_in_dummy:
            active_mask = [False] * num_parallel
            actions = [LIBERO_DUMMY_ACTION] * num_parallel
            for i in active_envs:
                active_mask[i] = True
                gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])
                if env_step_count[i] == 0:
                    logger.info(
                        "env %d trial %d warmup gripper env_command=%.1f",
                        i,
                        env_trial_id[i],
                        LIBERO_DUMMY_ACTION[-1],
                    )
            obs_batch, _, _, infos = await asyncio.to_thread(
                vec_env.step_selective, active_mask, actions
            )
            for i in active_envs:
                if obs_batch[i] is not None:
                    per_env_obs[i] = obs_batch[i]
                env_step_count[i] += 1
                global_progress.update(1)
                if infos[i].get("_done"):
                    await client.reset_connection(i)
                    client.clear_chunk(i)
                    new_state = finish_episode(i, True)
                    if new_state is not None:
                        per_env_obs[i] = await asyncio.to_thread(
                            vec_env.set_init_state_single, i, new_state
                        )
                        gripper_states[i].reset_from_obs(per_env_obs[i])
            continue

        # --- Phase: normal action steps ---
        # First, let all cached envs consume their remaining actions quickly
        # so they also enter recompute state before we send requests to server.
        # This maximizes the batch size on the server side.
        dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
        ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # Drain cache: step all envs that have cached actions until they
        # exhaust their cache or hit max_steps. This is fast (no server call).
        while True:
            cached_envs = [
                i
                for i in ready_envs
                if env_trial_id[i] >= 0
                and env_step_count[i] < total_steps
                and not client.needs_recompute(i)
            ]
            if not cached_envs:
                break

            # Step cached envs + dummy envs together
            active_mask = [False] * num_parallel
            actions = [LIBERO_DUMMY_ACTION] * num_parallel

            for i in dummy_envs:
                if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                    active_mask[i] = True
                    gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])

            for i in cached_envs:
                action = client.get_cached_action(i)
                if action is None:
                    continue
                actions[i] = single_action_to_libero_action(
                    action,
                    obs=per_env_obs[i],
                    gripper_state=gripper_states[i],
                )
                if env_action_step[i] == 0:
                    logger.info(
                        "env %d trial %d first cached policy_gripper=%s env_command=%.1f",
                        i,
                        env_trial_id[i],
                        action_dict_gripper_scalar(action),
                        actions[i][-1],
                    )
                active_mask[i] = True

            if not any(active_mask):
                break

            obs_batch, _, _, infos = await asyncio.to_thread(
                vec_env.step_selective, active_mask, actions
            )

            for i in range(num_parallel):
                if not active_mask[i]:
                    continue
                if save_videos and env_step_count[i] >= num_steps_wait:
                    images_hwc = extract_libero_images(per_env_obs[i])
                    env_replay[i].append(images_hwc)
                if obs_batch[i] is not None:
                    per_env_obs[i] = obs_batch[i]
                env_step_count[i] += 1
                if env_step_count[i] > num_steps_wait:
                    env_action_step[i] += 1
                global_progress.update(1)
                if infos[i].get("_done"):
                    await client.reset_connection(i)
                    client.clear_chunk(i)
                    new_state = finish_episode(i, True)
                    if new_state is not None:
                        per_env_obs[i] = await asyncio.to_thread(
                            vec_env.set_init_state_single, i, new_state
                        )
                        gripper_states[i].reset_from_obs(per_env_obs[i])

            # Update ready_envs for next drain iteration
            active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
            dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
            ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # Now all ready envs should need recompute (cache exhausted).
        # Collect and send all recompute requests at once.
        recompute_envs = [
            i
            for i in range(num_parallel)
            if env_trial_id[i] >= 0
            and env_step_count[i] >= num_steps_wait
            and env_step_count[i] < total_steps
            and client.needs_recompute(i)
        ]

        if not recompute_envs:
            # Nothing to do (all envs are in dummy or finished)
            await asyncio.sleep(0)
            continue

        n_dummy = len(
            [
                i
                for i in range(num_parallel)
                if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait
            ]
        )
        n_finished = len([i for i in range(num_parallel) if env_trial_id[i] < 0])
        logger.info(
            "Sending recompute: %d envs (dummy=%d, finished=%d, total_parallel=%d)",
            len(recompute_envs),
            n_dummy,
            n_finished,
            num_parallel,
        )

        # Send all recompute requests to server concurrently.
        # Server will batch them and return together.
        raw_obs_list = []
        current_steps_list = []
        for i in recompute_envs:
            raw_obs, _ = build_libero_raw_obs(
                obs=per_env_obs[i],
                task_description=task_description,
                embodiment_type=embodiment_type,
            )
            raw_obs_list.append(raw_obs)
            current_steps_list.append(env_action_step[i])

        await client.infer_batch(recompute_envs, raw_obs_list, current_steps_list)
        for i in recompute_envs:
            env_infer_calls[i] += 1

        # After recompute, step all recompute envs once (consume first action
        # from the fresh chunk) + dummy envs.
        active_mask = [False] * num_parallel
        actions = [LIBERO_DUMMY_ACTION] * num_parallel

        for i in dummy_envs:
            if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                active_mask[i] = True
                gripper_states[i].set_env_command(LIBERO_DUMMY_ACTION[-1])

        for i in recompute_envs:
            action = client.get_cached_action(i)
            if action is None:
                logger.warning("env %d has no cached action after infer, skipping step", i)
                continue
            actions[i] = single_action_to_libero_action(
                action,
                obs=per_env_obs[i],
                gripper_state=gripper_states[i],
            )
            if env_action_step[i] == 0:
                logger.info(
                    "env %d trial %d first recompute policy_gripper=%s env_command=%.1f",
                    i,
                    env_trial_id[i],
                    action_dict_gripper_scalar(action),
                    actions[i][-1],
                )
            active_mask[i] = True

        if not any(active_mask):
            await asyncio.sleep(0)
            continue

        obs_batch, _, _, infos = await asyncio.to_thread(
            vec_env.step_selective, active_mask, actions
        )

        for i in range(num_parallel):
            if not active_mask[i]:
                continue
            if save_videos and env_step_count[i] >= num_steps_wait:
                images_hwc = extract_libero_images(per_env_obs[i])
                env_replay[i].append(images_hwc)
            if obs_batch[i] is not None:
                per_env_obs[i] = obs_batch[i]
            env_step_count[i] += 1
            if env_step_count[i] > num_steps_wait:
                env_action_step[i] += 1
            global_progress.update(1)
            if infos[i].get("_done"):
                await client.reset_connection(i)
                client.clear_chunk(i)
                new_state = finish_episode(i, True)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(
                        vec_env.set_init_state_single, i, new_state
                    )
                    gripper_states[i].reset_from_obs(per_env_obs[i])

    global_progress.close()

    successes = sum(1 for v in trial_successes.values() if v)
    task_result = {
        "task_id": task_id,
        "task_description": task_description,
        "successes": successes,
        "total_episodes": num_trials,
        "success_rate": successes / num_trials if num_trials > 0 else 0.0,
        "infer_calls": sum(trial_infer_calls.values()),
    }

    if save_videos:
        for tid, success in trial_successes.items():
            images = trial_replay_images.get(tid, [])
            if images:
                save_rollout_video(
                    output_dir / "videos",
                    task_description,
                    tid,
                    success,
                    images,
                    fps=20,
                )

    return task_result


async def evaluate(args, cfg, embodiment_type: str | None) -> dict[str, Any]:
    ensure_libero_import()
    ensure_libero_config()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = get_max_steps(args.task_suite_name)

    if args.task_id is None:
        task_ids = list(range(task_suite.n_tasks))
    else:
        task_ids = [args.task_id]

    summary: dict[str, Any] = {
        "server_uri": args.server_uri,
        "ckpt_path": args.ckpt_path,
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "num_parallel": args.num_parallel,
        "num_steps_wait": args.num_steps_wait,
        "embodiment_type": embodiment_type,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "total_infer_calls": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    async with MultiPolicyWSClient(
        args.server_uri, args.num_parallel, timeout_s=args.timeout_s
    ) as client:
        logger.info(
            "Connected to policy server %s with %d parallel connections",
            args.server_uri,
            args.num_parallel,
        )
        summary["server_metadata"] = client.metadata

        for task_id in tqdm(task_ids, desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_description = task.language
            if len(initial_states) < args.num_trials_per_task:
                logger.warning(
                    "Task %s only has %d initial states, cycling to %d trials",
                    task_id,
                    len(initial_states),
                    args.num_trials_per_task,
                )

            env_fn = _make_env_fn(task, args.env_resolution, args.seed)
            dummy_fn = _make_dummy_env_fn(task, args.env_resolution, args.seed)

            vec_env = await asyncio.to_thread(
                LiberoAsyncVectorEnv,
                env_fns=[env_fn] * args.num_parallel,
                dummy_env_fn=dummy_fn,
            )

            try:
                task_result = await run_parallel_task(
                    task_id=task_id,
                    task_description=task_description,
                    initial_states=initial_states,
                    num_trials=args.num_trials_per_task,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    num_parallel=args.num_parallel,
                    client=client,
                    vec_env=vec_env,
                    embodiment_type=embodiment_type,
                    save_videos=args.save_videos,
                    output_dir=args.output_dir,
                )
            finally:
                await asyncio.to_thread(vec_env.close)

            summary["tasks"].append(task_result)
            summary["total_successes"] += task_result["successes"]
            summary["total_episodes"] += task_result["total_episodes"]
            summary["total_infer_calls"] += task_result["infer_calls"]
            logger.info(
                "Task %s success rate: %.3f (%d/%d)",
                task_id,
                task_result["success_rate"],
                task_result["successes"],
                task_result["total_episodes"],
            )

    summary["success_rate"] = (
        summary["total_successes"] / summary["total_episodes"] if summary["total_episodes"] else 0.0
    )
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate remote policy server on LIBERO (parallel)"
    )
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--server_uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_parallel", type=int, default=5)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--embodiment_type", default="libero")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout_s", type=float, default=300.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/libero_parallel_eval"))
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--env_resolution", type=int, default=DEFAULT_ENV_RESOLUTION)
    parser.add_argument("--log_level", default="INFO")
    args, remaining = parser.parse_known_args()
    return args, [item for item in remaining if "=" in item]


def main():
    args, overrides = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = None
    run_dir = None
    embodiment_type = args.embodiment_type
    if args.ckpt_path is not None:
        run_dir = find_run_dir(args.ckpt_path)
        cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

        if cfg.get("eval_embodiment") and "embodiment_datasets" in cfg.data:
            filter_embodiment(cfg, cfg.eval_embodiment)

        args.task_suite_name = _cfg_get(cfg, "libero.task_suite_name", args.task_suite_name)
        args.num_trials_per_task = _cfg_get(
            cfg,
            "libero.num_trials_per_task",
            _cfg_get(cfg, "EVALUATION.num_trials", args.num_trials_per_task),
        )
        args.num_steps_wait = _cfg_get(cfg, "libero.num_steps_wait", args.num_steps_wait)
        args.seed = args.seed if args.seed is not None else cfg.get("seed")
        if cfg.data.get("embodiment_datasets"):
            embodiment_type = cfg.data.embodiment_datasets.get(args.embodiment_type, {}).get(
                "embodiment_type", embodiment_type
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        logger.info("Resolved run dir: %s", run_dir)
    logger.info("Parallel envs: %d", args.num_parallel)

    summary = asyncio.run(evaluate(args, cfg, embodiment_type))
    summary_path = args.output_dir / f"{args.task_suite_name}_parallel_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info(
        "Total success rate: %.3f (%d/%d)",
        summary["success_rate"],
        summary["total_successes"],
        summary["total_episodes"],
    )
    logger.info("Results saved to %s", summary_path)


if __name__ == "__main__":
    main()

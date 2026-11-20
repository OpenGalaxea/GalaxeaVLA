"""Trial accounting of the parallel LIBERO evaluator when num_parallel > num_trials.

CPU only: the LIBERO vector env and the policy client are replaced by fakes, so
this exercises only the trial scheduling / counting logic of run_parallel_task.
"""

from __future__ import annotations

import asyncio
import importlib.util
import random
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _load_eval_module():
    path = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_parallel.py"
    spec = importlib.util.spec_from_file_location("eval_libero_parallel_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeGripperState:
    @classmethod
    def from_obs(cls, obs):
        return cls()

    def set_env_command(self, command):
        pass

    def reset_from_obs(self, obs):
        pass


class _FakeVecEnv:
    """Each episode either succeeds after a random number of steps or never
    succeeds (and is then ended by the evaluator's max-steps timeout)."""

    def __init__(self, num_envs: int, seed: int, success_prob: float = 0.8):
        self.num_envs = num_envs
        self.rng = random.Random(seed)
        self.success_prob = success_prob
        self.steps = [0] * num_envs
        self.success_at = [None] * num_envs

    def _start(self, i):
        self.steps[i] = 0
        if self.rng.random() < self.success_prob:
            self.success_at[i] = self.rng.randint(15, 60)
        else:
            self.success_at[i] = None
        return {"env": i}

    def reset(self):
        pass

    def set_init_state_each(self, states):
        assert len(states) == self.num_envs
        return [self._start(i) for i in range(self.num_envs)]

    def set_init_state_single(self, i, state):
        return self._start(i)

    def step_selective(self, active_mask, actions):
        obs = [None] * self.num_envs
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if active_mask[i]:
                self.steps[i] += 1
                obs[i] = {"env": i}
                target = self.success_at[i]
                infos[i]["_done"] = target is not None and self.steps[i] >= target
        return obs, None, None, infos


class _FakeClient:
    def __init__(self, num_envs: int, chunk: int = 8):
        self.chunk = chunk
        self.cache = [[] for _ in range(num_envs)]

    async def reset_all(self):
        pass

    async def reset_connection(self, i):
        pass

    def clear_chunk(self, i):
        self.cache[i] = []

    def needs_recompute(self, i):
        return not self.cache[i]

    def get_cached_action(self, i):
        return self.cache[i].pop(0) if self.cache[i] else None

    async def infer_batch(self, env_ids, raw_obs_list, current_steps_list):
        for i in env_ids:
            self.cache[i] = [DUMMY_ACTION] * self.chunk


@pytest.fixture
def eval_module(monkeypatch):
    module = _load_eval_module()
    monkeypatch.setattr(module, "LiberoGripperCommandState", _FakeGripperState)
    monkeypatch.setattr(module, "LIBERO_DUMMY_ACTION", DUMMY_ACTION)
    monkeypatch.setattr(
        module, "single_action_to_libero_action", lambda action, obs, gripper_state: list(action)
    )
    monkeypatch.setattr(module, "extract_libero_images", lambda obs: obs)
    monkeypatch.setattr(
        module,
        "build_libero_raw_obs",
        lambda obs, task_description, embodiment_type: ({}, None),
    )
    module._saved_trials = []
    monkeypatch.setattr(
        module,
        "save_rollout_video",
        lambda out_dir, desc, trial_id, success, images, fps: module._saved_trials.append(
            (trial_id, success)
        ),
    )
    return module


@pytest.mark.parametrize(
    "num_trials,num_parallel",
    [(1, 4), (2, 4), (3, 4), (4, 4), (10, 4), (1, 1), (7, 3)],
)
@pytest.mark.parametrize("seed", range(5))
def test_runs_exactly_num_trials(eval_module, tmp_path, num_trials, num_parallel, seed):
    result = asyncio.run(
        eval_module.run_parallel_task(
            task_id=0,
            task_description="fake task",
            initial_states=[object()] * 50,
            num_trials=num_trials,
            max_steps=70,
            num_steps_wait=5,
            num_parallel=num_parallel,
            client=_FakeClient(num_parallel),
            vec_env=_FakeVecEnv(num_parallel, seed),
            embodiment_type="libero",
            save_videos=True,
            output_dir=tmp_path,
        )
    )

    trial_ids = sorted(trial_id for trial_id, _ in eval_module._saved_trials)
    assert trial_ids == list(range(num_trials))
    assert result["total_episodes"] == num_trials
    assert 0 <= result["successes"] <= num_trials
    assert result["successes"] == sum(success for _, success in eval_module._saved_trials)

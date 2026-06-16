from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Any

import numpy as np
from gym.vector.utils import CloudpickleWrapper

__all__ = ["LiberoAsyncVectorEnv"]


class LiberoAsyncVectorEnv:
    def __init__(
        self,
        env_fns: list,
        dummy_env_fn=None,
        daemon: bool = True,
    ):
        try:
            mp.set_start_method("forkserver")
        except RuntimeError:
            pass

        ctx = mp.get_context("forkserver")
        self.num_envs = len(env_fns)
        self.closed = False

        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        dummy_env.close()
        del dummy_env

        self.parent_pipes: list = []
        self.processes: list = []
        self.error_queue = ctx.Queue()

        for idx, env_fn in enumerate(env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                name=f"LiberoWorker-{idx}",
                args=(idx, CloudpickleWrapper(env_fn), child_pipe, parent_pipe, self.error_queue),
            )
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)
            process.daemon = daemon
            process.start()
            child_pipe.close()

    def reset(self, **kwargs) -> list:
        for pipe in self.parent_pipes:
            pipe.send(("reset", kwargs))
        return list(self._recv_all())

    def step(self, actions) -> tuple[list, np.ndarray, np.ndarray, list]:
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        results = self._recv_all()
        observations_list, rewards, dones, infos = zip(*results)
        return (
            list(observations_list),
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            list(infos),
        )

    def step_selective(self, active_mask: list[bool], actions: list) -> tuple:
        for i in range(self.num_envs):
            if active_mask[i]:
                self.parent_pipes[i].send(("step", actions[i]))
            else:
                self.parent_pipes[i].send(("nop", None))
        raw_results = self._recv_all()
        obs_list = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            if active_mask[i]:
                obs, reward, done, info = raw_results[i]
                info = dict(info) if info else {}
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            else:
                obs_list.append(None)
                rewards.append(0.0)
                dones.append(False)
                infos.append({})
        return (
            obs_list,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def set_init_state_each(self, states: list) -> list:
        for pipe, state in zip(self.parent_pipes, states):
            pipe.send(("set_init_state", state))
        return list(self._recv_all())

    def set_init_state_single(self, idx: int, state) -> Any:
        self.parent_pipes[idx].send(("set_init_state", state))
        result, _success = self.parent_pipes[idx].recv()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pipe in self.parent_pipes:
            if pipe is not None and not pipe.closed:
                pipe.send(("close", None))
        for pipe in self.parent_pipes:
            if pipe is not None and not pipe.closed:
                try:
                    pipe.recv()
                except Exception:
                    pass
        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for p in self.processes:
            p.join(timeout=5)

    def _recv_all(self, timeout=None):
        results = []
        for i, pipe in enumerate(self.parent_pipes):
            if pipe is None or pipe.closed:
                raise RuntimeError(f"Worker {i} pipe is closed")
            if timeout is not None:
                if not pipe.poll(timeout):
                    raise mp.TimeoutError(f"Worker {i} response timed out after {timeout}s")
            result, success = pipe.recv()
            if not success:
                self._flush_errors()
            results.append(result)
        return results

    def _flush_errors(self):
        errors = []
        while not self.error_queue.empty():
            index, exctype, value = self.error_queue.get()
            errors.append((index, exctype, value))
            if index < len(self.parent_pipes) and self.parent_pipes[index] is not None:
                self.parent_pipes[index].close()
                self.parent_pipes[index] = None
        if errors:
            _, exctype, value = errors[-1]
            raise exctype(value)


def _worker(index, env_fn, pipe, parent_pipe, error_queue):
    try:
        env = env_fn()
    except Exception:
        with open(f"/tmp/libero_worker_{index}_error.log", "w") as f:
            import traceback

            f.write(f"Worker {index} env creation failed:\n")
            f.write(traceback.format_exc())
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
        return
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                obs = env.reset(**data)
                pipe.send((obs, True))
            elif command == "step":
                obs, reward, done, info = env.step(data)
                info = dict(info) if info else {}
                if done:
                    info["_done"] = True
                pipe.send(((obs, reward, done, info), True))
            elif command == "set_init_state":
                if data is not None:
                    env.reset()  # zero velocities from previous episode before applying init state
                    obs = env.set_init_state(data)
                    pipe.send((obs, True))
                else:
                    pipe.send((None, True))
            elif command == "nop":
                pipe.send((None, True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                fn = getattr(env, name)
                pipe.send((fn(*args, **kwargs) if callable(fn) else fn, True))
            else:
                raise RuntimeError(f"Unknown command: {command}")
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()

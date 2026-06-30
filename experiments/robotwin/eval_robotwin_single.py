"""RoboTwin single-task evaluation entrypoint (Hydra)."""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PHASE_TO_RESULT_FILENAME = {"demo_clean": "_result_clean.txt", "demo_randomized": "_result_random.txt"}
_ROBOTWIN_TEST_NUM_MARKER = "    test_num = 100\n"
_ROBOTWIN_TEST_NUM_PATCH = (
    '    test_num = int(usr_args["eval_num_episodes"])\n'
    "    if test_num <= 0:\n"
    '        raise ValueError("`eval_num_episodes` must be > 0.")\n'
)


def _collect_robotwin_result(
    robotwin_root: Path,
    task_name: str,
    policy_name: str,
    task_config: str,
    dest_dir: Path,
) -> None:
    """Find RoboTwin 2.0 result file and copy to the manager-expected location."""
    dest_filename = _PHASE_TO_RESULT_FILENAME.get(task_config)
    if dest_filename is None:
        return
    search_dir = robotwin_root / "eval_result" / task_name / policy_name / task_config
    if not search_dir.exists():
        return
    candidates = sorted(search_dir.rglob("_result.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(candidates[0]), str(dest_dir / dest_filename))


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried EVALUATION.dataset_stats_path and "
        "checkpoint parent directories. Please pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path, policy_name: str) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / policy_name
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _patch_robotwin_eval_policy_source(source: str) -> str:
    if _ROBOTWIN_TEST_NUM_PATCH in source:
        return source
    if _ROBOTWIN_TEST_NUM_MARKER not in source:
        raise RuntimeError(
            "Unable to patch RoboTwin eval_policy.py for `eval_num_episodes`: "
            "expected to find `test_num = 100`. Please update the patch for this RoboTwin version."
        )
    return source.replace(_ROBOTWIN_TEST_NUM_MARKER, _ROBOTWIN_TEST_NUM_PATCH, 1)


def _prepare_robotwin_eval_policy_script(robotwin_root: Path) -> Path:
    source_path = robotwin_root / "script" / "eval_policy.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"RoboTwin eval_policy.py not found: {source_path}")

    patched_source = _patch_robotwin_eval_policy_source(source_path.read_text(encoding="utf-8"))
    patched_path = source_path.with_name(f"_galaxeafm_eval_policy_{os.getpid()}.py")
    patched_path.write_text(patched_source, encoding="utf-8")
    return patched_path


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    sim_task = HydraConfig.get().runtime.choices["task"]
    if sim_task is None or str(sim_task).strip() == "":
        raise ValueError("`task` must be specified, e.g. task=pretrain/bench/robotwin_g05v2.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_name = str(cfg.EVALUATION.policy_name).strip()
    if policy_name == "":
        raise ValueError("`EVALUATION.policy_name` must not be empty.")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / policy_name).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(
        robotwin_root=robotwin_root,
        policy_source_dir=policy_source_dir,
        policy_name=policy_name,
    )

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = run_output_dir / str(cfg.EVALUATION.task_name)

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    eval_num_episodes = int(cfg.EVALUATION.eval_num_episodes)
    if eval_num_episodes <= 0:
        raise ValueError(f"`EVALUATION.eval_num_episodes` must be > 0, got {eval_num_episodes}")
    robotwin_eval_script = _prepare_robotwin_eval_policy_script(robotwin_root)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", "bf16" if cfg.model.enable_bf16_training else "no")
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )

    cmd = [
        sys.executable,
        "-u",
        robotwin_eval_script.relative_to(robotwin_root).as_posix(),
        "--config",
        f"policy/{policy_name}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_file, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                cwd=str(robotwin_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            return_code = process.wait()
    finally:
        robotwin_eval_script.unlink()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    _collect_robotwin_result(
        robotwin_root=robotwin_root,
        task_name=str(cfg.EVALUATION.task_name),
        policy_name=policy_name,
        task_config=str(cfg.EVALUATION.task_config),
        dest_dir=robotwin_eval_base,
    )
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()

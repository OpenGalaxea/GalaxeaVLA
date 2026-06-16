"""ckpt_utils.py

Checkpoint / run-directory utilities shared by eval and serve scripts.
Extracted from eval_open_loop.py to avoid circular imports and keep
entry-point scripts lean.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from g05.utils.hf import resolve_hf_model_path
from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

# HF processor/tokenizer config files — no model weights (.safetensors excluded).
_HF_PROCESSOR_WHITELIST = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "video_preprocessor_config.json",
    "configuration.json",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def copy_hf_processor_files(src: str | Path, dst: str | Path) -> None:
    """Copy HF processor/tokenizer sidecar files from src_dir to dst_dir by allowlist.

    Returns the list of actually copied filenames. Does not copy weights. Creates
    dst_dir if missing and overwrites existing files.
    """
    src = Path(resolve_hf_model_path(src, allow_patterns=_HF_PROCESSOR_WHITELIST))
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _HF_PROCESSOR_WHITELIST:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
            copied.append(name)
    logger.info(f"Copied {len(copied)} HF processor files to {dst}: {copied}")


def _apply_hf_processor_sidecar(cfg: DictConfig, run_dir: Path):
    """Map HF processor path to run_dir/hf_processor/ when present and null pretrained_model_path.

    Modified config keys:
      - model.model_arch.hf_processor_path
      - model.model_arch.pretrained_model_path          (set to null)
      - model.processor.tokenizer_params.pretrained_model_name_or_path  (if present)

    Returns whether a local sidecar was found. If not found, falls back to
    hf_processor_path <- original pretrained_model_path.
    """
    local_hf = Path(run_dir) / "hf_processor"
    has_local = local_hf.exists()

    pretrained_path = cfg.model.model_arch.get("pretrained_model_path", None)
    if pretrained_path is not None:
        cfg.model.model_arch.hf_processor_path = str(local_hf) if has_local else pretrained_path
        cfg.model.model_arch.pretrained_model_path = None

    if has_local and OmegaConf.select(cfg, "model.processor.tokenizer_params") is not None:
        cfg.model.processor.tokenizer_params.pretrained_model_name_or_path = str(local_hf)


def _patch_g05_compat(cfg: DictConfig) -> None:
    target = cfg.get("model", {}).get("model_arch", {}).get("_target_", "")
    if target != "g05.models.g05.g05_policy.G05Policy":
        return

    base_cfg_path = (
        _project_root() / "configs" / "model" / "vla" / "_base_g05_v2.yaml"
    )
    if not base_cfg_path.exists():
        logger.warning(f"G05 compat base config not found: {base_cfg_path}")
        return

    base_model_cfg = OmegaConf.load(base_cfg_path)
    cfg.model = OmegaConf.merge(base_model_cfg, cfg.model)
    logger.info("Patched G05 config with current base defaults for eval/serve compatibility")


def find_run_dir(ckpt_path: str) -> Path:
    """Find the run directory containing .hydra/config.yaml by searching upward from ckpt_path."""
    ckpt = Path(ckpt_path).absolute()
    candidate = ckpt.parent
    for _ in range(5):
        if (candidate / ".hydra" / "config.yaml").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(f"Could not find .hydra/config.yaml within 5 levels above {ckpt_path}")


def _register_hydra_builtin_resolvers() -> None:
    """Register Hydra built-in resolvers (`now`, `oc.env`) that aren't available outside Hydra."""
    OmegaConf.register_new_resolver(
        "now", lambda pattern, _tz="": datetime.now().strftime(pattern), replace=True
    )
    OmegaConf.register_new_resolver(
        "oc.env",
        lambda key, default=None: (
            os.environ[key]
            if key in os.environ
            else default
            if default is not None
            else (_ for _ in ()).throw(KeyError(f"Env var '{key}' not set"))
        ),
        replace=True,
    )


def load_config_from_run_dir(run_dir: Path, ckpt_path: str, overrides: list[str]) -> DictConfig:
    """Load config from a run directory's .hydra/config.yaml and patch Hydra-specific interpolations."""
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RESET = "\033[0m"

    logger.info(f"{_CYAN}📂 Loading config from {run_dir / '.hydra' / 'config.yaml'}{_RESET}")
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    _patch_g05_compat(cfg)

    # Derive eval subfolder from checkpoint filename (e.g. step_14000.pt -> eval_step_14000)
    ckpt_stem = Path(ckpt_path).stem  # e.g. "step_14000" or "last"
    eval_dir_name = f"eval_{ckpt_stem}"

    # Patch Hydra-specific interpolations
    cfg.run_dir = str(run_dir)
    cfg.output_dir = str(run_dir / eval_dir_name)
    cfg.exp_name = run_dir.name
    cfg.logger.task = "eval"
    cfg.logger.experiment_name = f"eval_{run_dir.name}"
    cfg.logger.mode = "disabled"
    cfg.ckpt_path = str(Path(ckpt_path).resolve())
    logger.info(f"{_CYAN}🔖 Checkpoint: {ckpt_stem}{_RESET}")

    _apply_hf_processor_sidecar(cfg, run_dir)

    # Auto-detect action_tokenizer.pt copied by finetune.py into run_dir.
    # .hydra/config.yaml is saved before finetune.py copies & updates ckpt_dir,
    # so the saved config still points to the original (possibly unreachable) path.
    local_at = run_dir / "action_tokenizer.pt"
    if local_at.exists():
        try:
            cfg.model.tokenizer.vq_config.ckpt_dir = str(local_at)
            logger.info(f"{_GREEN}🎯 Auto-resolved action_tokenizer → {local_at}{_RESET}")
        except Exception:
            pass  # config has no tokenizer.vq_config.ckpt_dir — skip
    else:
        logger.info(f"{_YELLOW}⚠️  No local action_tokenizer.pt found, using config default{_RESET}")

    # register_default_resolvers already handles oc.load, eval, split, etc.
    # We only need to add Hydra built-ins (now, oc.env) that aren't in register_default_resolvers.
    _register_hydra_builtin_resolvers()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        logger.info(f"{_GREEN}⚙️  Applied {len(overrides)} override(s): {overrides}{_RESET}")

    logger.info(f"{_GREEN}✅ Config loaded successfully{_RESET}")
    return cfg


def load_config_from_task_yaml(task_yaml: str, ckpt_path: str, overrides: list[str]) -> DictConfig:
    """Load eval config via Hydra composition from a task yaml file.

    Use when you want to evaluate a checkpoint with a *different* config than
    the one it was trained with (e.g. a newer task yaml).

    Args:
        task_yaml: Path to task yaml under configs/task/ (e.g.
                   "configs/task/pretrain/bench/foldbench_fmonly_g05v2.yaml")
        ckpt_path: Path to the checkpoint (.pt file) to evaluate.
        overrides:  Extra key=value Hydra overrides.
    """
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RESET = "\033[0m"

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from g05.utils.config.config_resolvers import register_default_resolvers

    register_default_resolvers()
    _register_hydra_builtin_resolvers()

    # Derive task name (e.g. "pretrain/bench/foldbench_fmonly_g05v2")
    project_root = _project_root()
    task_config_root = project_root / "configs" / "task"
    task_path = Path(task_yaml).resolve()
    task_name = str(task_path.relative_to(task_config_root).with_suffix(""))
    logger.info(f"{_CYAN}📂 Loading task config: {task_name}{_RESET}")

    ckpt_stem = Path(ckpt_path).stem
    eval_dir_name = f"eval_{ckpt_stem}"
    output_dir = str(Path(ckpt_path).parent.parent / eval_dir_name)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(project_root / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={task_name}",
                f"hydra.run.dir={output_dir}",
                f"output_dir={output_dir}",  # patches ${hydra:runtime.output_dir}
                f"logger.task={task_name}",  # patches ${hydra:runtime.choices.task}
                "exp_name=eval",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        logger.info(f"{_GREEN}⚙️  Applied {len(overrides)} override(s): {overrides}{_RESET}")
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)

    # Eval-specific patches
    cfg.ckpt_path = str(Path(ckpt_path).resolve())
    cfg.output_dir = output_dir
    cfg.run_dir = str(Path(ckpt_path).parent.parent)
    cfg.logger.task = "eval"
    cfg.logger.experiment_name = f"eval_{Path(ckpt_path).parent.parent.name}"
    cfg.logger.mode = "disabled"
    logger.info(f"{_CYAN}🔖 Checkpoint: {ckpt_stem}{_RESET}")

    _apply_hf_processor_sidecar(cfg, Path(cfg.run_dir))

    logger.info(f"{_GREEN}✅ Config loaded from task yaml{_RESET}")
    return cfg

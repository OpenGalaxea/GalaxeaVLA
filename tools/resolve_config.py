#!/usr/bin/env python3
"""Resolve Hydra task config and optionally diff two configs.

Usage:
    # Resolve a single task config (prints YAML):
    python tools/resolve_config.py pretrain/bench/foldbench_fmonly_g05v2

    # Resolve with overrides:
    python tools/resolve_config.py pretrain/bench/foldbench_fmonly_g05v2 model.batch_size=16

    # Diff two task configs (colored diff):
    python tools/resolve_config.py --diff pretrain/bench/foldbench_fmonly_g05v2 pretrain/bench/foldbench_fmonly

    # Only show a sub-key:
    python tools/resolve_config.py pretrain/bench/foldbench_fmonly_g05v2 --key model.model_arch

    # Diff a sub-key:
    python tools/resolve_config.py --diff taskA taskB --key model.processor

    # Output to file:
    python tools/resolve_config.py pretrain/bench/foldbench_fmonly_g05v2 -o resolved.yaml

    # Show only keys that differ:
    python tools/resolve_config.py --diff taskA taskB --only-diff

    # Sort keys alphabetically (like old behavior):
    python tools/resolve_config.py pretrain/bench/foldbench_fmonly_g05v2 --sort-keys
"""

import argparse
import difflib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_task_config(task_name: str, overrides: list[str] | None = None) -> "DictConfig":
    """Resolve a task config via Hydra compose."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf

    # Register custom resolvers before compose
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from g05.utils.config.config_resolvers import register_default_resolvers

    register_default_resolvers()

    config_dir = str(PROJECT_ROOT / "configs")
    all_overrides = [f"task={task_name}"]
    if overrides:
        all_overrides.extend(overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=all_overrides)

    return cfg


def cfg_to_yaml(
    cfg,
    key: str | None = None,
    sort_keys: bool = False,
    collapse_datasets: bool = False,
    post_build: bool = False,
) -> str:
    """Convert resolved config (or sub-key) to YAML string.

    Resolves interpolations per-node: resolvable ones are expanded,
    unresolvable ones (e.g. ${hydra:runtime.*}, ${now:...}) are kept as-is.

    If collapse_datasets is True, any `dataset_dirs: [...]` list with more than
    one entry is collapsed to the first entry plus a "... (+N more, total M)"
    annotation, for readability.

    If post_build is True, the cfg is transformed to show what dataset and
    processor actually receive (mirrors processor_utils.build_processors and
    processor_utils.instantiate_dataset). See _apply_post_build for details.
    """
    from omegaconf import OmegaConf

    container = _deep_resolve(cfg)
    if post_build:
        _apply_post_build(container)
    if collapse_datasets:
        _collapse_dataset_dirs_inplace(container)

    if key:
        for part in key.split("."):
            if not isinstance(container, dict) or part not in container:
                raise KeyError(f"key path `{key}` not found at sub-key `{part}`")
            container = container[part]

    return OmegaConf.to_yaml(OmegaConf.create(container), sort_keys=sort_keys)


def _collapse_dataset_dirs_inplace(node) -> None:
    """Recursively walk container, collapsing `dataset_dirs` lists to one entry."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "dataset_dirs" and isinstance(v, list) and len(v) > 1:
                total = len(v)
                node[k] = [v[0], f"... (+{total - 1} more, total {total})"]
            else:
                _collapse_dataset_dirs_inplace(v)
    elif isinstance(node, list):
        for item in node:
            _collapse_dataset_dirs_inplace(item)


def _apply_post_build(container: dict) -> None:
    """Mutate `container` so model.processor / data reflect what dataset and
    processor constructors actually receive at runtime.

    Mirrors:
      - `g05.utils.data.processor_utils.build_processors`  (lines 38-151)
      - `g05.utils.data.processor_utils.instantiate_dataset` (lines 154-179)

    Steps:
      1. If `data.processors` exists (mixture case), replace `model.processor`
         with a dict keyed by embodiment_type, each value = merged per-emb cfg
         (OmegaConf.merge(emb, base) + action_state_merger whole-replace +
          allow_emb_target_override + processor_overrides + embodiment_type).
      2. Strip mixture control keys from `data`:
         processors / processor_overrides / allow_emb_target_override.

    Does NOT instantiate anything — purely a static view.
    """
    from omegaconf import OmegaConf

    if not isinstance(container, dict):
        return

    data = container.get("data")
    model = container.get("model")
    if not isinstance(data, dict) or not isinstance(model, dict):
        return

    processors_cfg = data.get("processors")
    processor_base = model.get("processor")

    if processors_cfg and isinstance(processors_cfg, dict) and "embodiment_datasets" in data:
        # Determine allow_emb_target_override priority: data-level > base-level.
        data_flag = data.get("allow_emb_target_override")
        base_flag = (
            (processor_base or {}).get("allow_emb_target_override", False)
            if isinstance(processor_base, dict)
            else False
        )
        allow_emb_target = data_flag if data_flag is not None else base_flag

        mixture_overrides = data.get("processor_overrides") or {}

        per_emb_merged: dict = {}
        seen_signatures: dict = {}
        embodiment_datasets = data.get("embodiment_datasets") or {}

        for emb_name, emb_ds_cfg in embodiment_datasets.items():
            if not isinstance(emb_ds_cfg, dict):
                continue
            emb_type = str(emb_ds_cfg.get("embodiment_type"))
            if emb_type not in processors_cfg:
                continue
            if emb_type in per_emb_merged:
                # duplicate embodiment_type — skip; build_processors errors only
                # when signatures differ at runtime. Here we just keep first.
                continue

            emb_cfg = processors_cfg[emb_type]
            # OmegaConf.merge semantics: later overrides earlier (base wins).
            if isinstance(processor_base, dict):
                merged = OmegaConf.to_container(
                    OmegaConf.merge(OmegaConf.create(emb_cfg), OmegaConf.create(processor_base)),
                    resolve=True,
                )
                # action_state_merger: whole-replace from base.
                if processor_base.get("action_state_merger") is not None:
                    merged["action_state_merger"] = processor_base["action_state_merger"]
                # allow_emb_target_override: emb _target_ wins over base _target_.
                if allow_emb_target:
                    emb_target = emb_cfg.get("_target_") if isinstance(emb_cfg, dict) else None
                    if emb_target is not None:
                        merged["_target_"] = emb_target
                # Control key never reaches downstream.
                merged.pop("allow_emb_target_override", None)
            else:
                merged = dict(emb_cfg) if isinstance(emb_cfg, dict) else emb_cfg

            # Mixture-level overrides.
            if isinstance(mixture_overrides, dict):
                for k, v in mixture_overrides.items():
                    merged[k] = v

            # Inject embodiment_type.
            if isinstance(merged, dict):
                merged["embodiment_type"] = emb_type

            per_emb_merged[emb_type] = merged

        model["processor"] = per_emb_merged

    # Strip dataset-side mixture control keys (instantiate_dataset behavior).
    for k in ("processors", "processor_overrides", "allow_emb_target_override"):
        data.pop(k, None)


def _deep_resolve(cfg) -> object:
    """Recursively resolve an OmegaConf config, tolerating per-node failures."""
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, DictConfig):
        result = {}
        for k in cfg:
            result[k] = _resolve_child(cfg, k)
        return result
    elif isinstance(cfg, ListConfig):
        result = []
        for i in range(len(cfg)):
            result.append(_resolve_child(cfg, i))
        return result
    else:
        return cfg


def _resolve_child(parent, key) -> object:
    """Resolve a single child; on failure, keep the raw interpolation string."""
    from omegaconf import DictConfig, ListConfig, OmegaConf, ValueNode

    try:
        val = parent[key]
        # If it's a container, recurse
        if isinstance(val, (DictConfig, ListConfig)):
            return _deep_resolve(val)
        return val
    except Exception:
        # Could not resolve — preserve the original ${...} expression
        try:
            raw = parent._get_node(key)
            if isinstance(raw, ValueNode):
                # Return the raw interpolation expression (e.g. "${hydra:runtime.output_dir}")
                return raw._value()
            raw_val = OmegaConf.to_container(raw, resolve=False, throw_on_missing=False)
            return raw_val
        except Exception:
            return f"<UNRESOLVED: {key}>"


def colored_diff(
    text_a: str, text_b: str, label_a: str, label_b: str, only_diff: bool = False
) -> str:
    """Generate a unified diff with ANSI colors."""
    lines_a = text_a.splitlines(keepends=False)
    lines_b = text_b.splitlines(keepends=False)

    diff = list(
        difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm="")
    )
    if not diff:
        return "\033[32m✓ No differences found.\033[0m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    colored = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"{CYAN}{line}{RESET}")
        elif line.startswith("-"):
            colored.append(f"{RED}{line}{RESET}")
        elif line.startswith("+"):
            colored.append(f"{GREEN}{line}{RESET}")
        elif line.startswith("@@"):
            colored.append(f"{CYAN}{line}{RESET}")
        else:
            colored.append(line)

    return "\n".join(colored)


def key_level_diff(yaml_a: str, yaml_b: str, label_a: str, label_b: str) -> str:
    """Show only keys that have different values (flat comparison)."""
    from omegaconf import OmegaConf

    dict_a = OmegaConf.to_container(OmegaConf.create(yaml_a), resolve=False)
    dict_b = OmegaConf.to_container(OmegaConf.create(yaml_b), resolve=False)

    def flatten(d, prefix=""):
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                items.update(flatten(v, new_key))
        elif isinstance(d, list):
            items[prefix] = d
        else:
            items[prefix] = d
        return items

    flat_a = flatten(dict_a)
    flat_b = flatten(dict_b)
    all_keys = sorted(set(flat_a) | set(flat_b))

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    lines = []
    for key in all_keys:
        va = flat_a.get(key, "<MISSING>")
        vb = flat_b.get(key, "<MISSING>")
        if va != vb:
            lines.append(f"{YELLOW}{key}{RESET}")
            lines.append(f"  {RED}{label_a}: {va}{RESET}")
            lines.append(f"  {GREEN}{label_b}: {vb}{RESET}")

    if not lines:
        return f"{GREEN}✓ No differences found.{RESET}"

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve Hydra task config and optionally diff two configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task", help="Task config name (e.g. pretrain/bench/foldbench_fmonly_g05v2)"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides (e.g. model.batch_size=16)")
    parser.add_argument("--diff", metavar="TASK_B", help="Second task to diff against")
    parser.add_argument("--key", "-k", help="Only show sub-key (e.g. model.model_arch)")
    parser.add_argument(
        "--only-diff", action="store_true", help="In diff mode, only show differing keys"
    )
    parser.add_argument("-o", "--output", help="Write resolved YAML to file")
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort keys alphabetically (default: preserve original order)",
    )
    parser.add_argument(
        "--collapse-datasets",
        action="store_true",
        help="Collapse `dataset_dirs` lists to first entry + count (for readability)",
    )
    parser.add_argument(
        "--post-build",
        action="store_true",
        help="Show cfg as dataset/processor constructors actually receive it "
        "(per-embodiment processor merge + data control-key strip).",
    )

    args = parser.parse_args()

    cfg_a = resolve_task_config(args.task, args.overrides or None)
    yaml_a = cfg_to_yaml(
        cfg_a,
        args.key,
        sort_keys=args.sort_keys,
        collapse_datasets=args.collapse_datasets,
        post_build=args.post_build,
    )

    if args.diff:
        cfg_b = resolve_task_config(args.diff, args.overrides or None)
        yaml_b = cfg_to_yaml(
            cfg_b,
            args.key,
            sort_keys=args.sort_keys,
            collapse_datasets=args.collapse_datasets,
            post_build=args.post_build,
        )

        label_a = args.task
        label_b = args.diff

        if args.only_diff:
            output = key_level_diff(yaml_a, yaml_b, label_a, label_b)
        else:
            output = colored_diff(yaml_a, yaml_b, label_a, label_b)

        print(output)
    else:
        if args.output:
            Path(args.output).write_text(yaml_a)
            print(f"Written to {args.output}")
        else:
            print(yaml_a)


if __name__ == "__main__":
    main()

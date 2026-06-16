"""processor_utils.py

Unified processor & dataset instantiation utilities.

All scripts (finetune, train_vq, eval_open_loop, ...) should use these two
functions instead of manually stripping / merging / instantiating.
"""
from __future__ import annotations

import copy

import tqdm as tqdm_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g05.data_processor.processor.base_processor import BaseProcessor
from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.logging.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cache: avoids re-instantiating identical processors across
# train/eval build_processors() calls (e.g. 17 embodiments × 2 calls = 34→17).
# Key: OmegaConf YAML string of the merged per-embodiment config.
_processor_instance_cache: dict[str, object] = {}


def build_processors(
    cfg: DictConfig,
    **per_emb_overrides,
) -> BaseProcessor | MixtureProcessor:
    """Build processor(s) from Hydra config.

    Mixture case (cfg.data.processors exists):
        For each embodiment_type, merge per-type config with cfg.model.processor (base),
        apply per_emb_overrides, then wrap in MixtureProcessor.

    Single case:
        Directly instantiate cfg.model.processor.

    Args:
        cfg: Full Hydra config (needs cfg.data, cfg.model.processor).
        **per_emb_overrides: Extra fields injected into every per-emb processor
            config before instantiation (e.g. num_obs_steps=1 for train_vq).

    Returns:
        MixtureProcessor or BaseProcessor.
    """
    processors_cfg = cfg.data.get("processors", None)

    if not processors_cfg:
        return instantiate(cfg.model.processor)

    assert "embodiment_datasets" in cfg.data, \
        "MixtureProcessor requires embodiment_datasets in cfg.data!"

    processor_base = None
    if cfg.model.get("processor", None):
        processor_base = OmegaConf.create(
            OmegaConf.to_container(cfg.model.processor, resolve=True)
        )
        OmegaConf.set_struct(processor_base, False)

    required_types = {
        str(emb_ds_cfg.embodiment_type)
        for emb_ds_cfg in cfg.data.embodiment_datasets.values()
    }
    processor_keys = {str(k) for k in processors_cfg.keys()}
    if processor_keys != required_types:
        raise ValueError(
            "cfg.data.processors must be keyed exactly by embodiment_type. "
            f"Missing: {sorted(required_types - processor_keys)}; "
            f"extra: {sorted(processor_keys - required_types)}."
        )

    result = {}
    signatures = {}
    _items = list(cfg.data.embodiment_datasets.items())
    for emb_name, emb_ds_cfg in tqdm_module.tqdm(
        _items, desc="Building processors", leave=False, disable=len(_items) <= 1,
    ):
        emb_type = str(emb_ds_cfg.embodiment_type)
        emb_processor_cfg = processors_cfg[emb_type]

        OmegaConf.set_struct(emb_processor_cfg, False)
        if processor_base is not None:
            merged = OmegaConf.merge(emb_processor_cfg, processor_base)
            # Some nested component configs (especially action_state_merger) should be
            # replaced as a whole when overridden from task-level model.processor,
            # instead of recursively merging dict keys and keeping stale parts_meta.
            if processor_base.get("action_state_merger", None) is not None:
                merged.action_state_merger = OmegaConf.create(
                    OmegaConf.to_container(processor_base.action_state_merger, resolve=True)
                )
            # Default follows OmegaConf merge semantics, where base overrides emb.
            # Only allow data-level _target_ to override base in reverse when mixture
            # or task explicitly declares allow_emb_target_override: true. This is
            # used for per-embodiment processor class selection.
            # Prefer mixture-level cfg.data.allow_emb_target_override, then task-level
            # cfg.model.processor.allow_emb_target_override. Default is false.
            allow_emb_target = (
                cfg.data.get("allow_emb_target_override", None)
                if cfg.data.get("allow_emb_target_override", None) is not None
                else processor_base.get("allow_emb_target_override", False)
            )
            if allow_emb_target:
                emb_target = emb_processor_cfg.get("_target_", None)
                if emb_target is not None:
                    merged["_target_"] = emb_target
            # Control-flow key; do not pass to downstream instantiate.
            merged.pop("allow_emb_target_override", None)
        else:
            merged = emb_processor_cfg

        # Apply mixture-level processor_overrides (e.g. norm_default_mode from at_aug_tail.yaml)
        mixture_overrides = cfg.data.get("processor_overrides", None)
        if mixture_overrides:
            for k, v in OmegaConf.to_container(mixture_overrides, resolve=True).items():
                merged[k] = v

        # Apply per-embodiment overrides (e.g. num_obs_steps=1)
        for k, v in per_emb_overrides.items():
            merged[k] = v

        # Inject embodiment_type into config before instantiation so that
        # nested objects (e.g. samples_builder) receive the correct value
        # during __init__ rather than via a post-hoc attribute assignment.
        merged["embodiment_type"] = emb_type

        signature = OmegaConf.to_container(merged, resolve=True)
        if emb_type in signatures:
            if signatures[emb_type] != signature:
                raise ValueError(
                    f"embodiment_type `{emb_type}` is used by multiple sources with "
                    f"non-equivalent processor configs; latest source is `{emb_name}`."
                )
            continue
        signatures[emb_type] = signature

        _cache_key = OmegaConf.to_yaml(merged)
        if _cache_key in _processor_instance_cache:
            p = copy.deepcopy(_processor_instance_cache[_cache_key])
            logger.debug(f"[build_processors] Cache hit for {emb_name}, reusing processor instance")
        else:
            p = instantiate(merged)
            _processor_instance_cache[_cache_key] = p
            p = copy.deepcopy(p)

        result[emb_type] = p

    return MixtureProcessor(result)


def instantiate_dataset(cfg: DictConfig, **kwargs):
    """Instantiate dataset from cfg.data, stripping processors first.

    cfg.data may contain a ``processors`` key that is consumed by
    :func:`build_processors` but is not a valid argument for the dataset
    constructor.  This function strips it before calling ``instantiate``.

    Args:
        cfg: Full Hydra config (needs cfg.data).
        **kwargs: Extra arguments forwarded to ``instantiate``
            (e.g. is_training_set=True).

    Returns:
        MixtureLerobotDataset or BaseLerobotDataset.
    """
    data_cfg = cfg.data
    # Strip mixture-level control keys that aren't valid dataset constructor args.
    # `processors` / `processor_overrides`: consumed by build_processors.
    # `allow_emb_target_override`: consumed by build_processors (per-emb _target_ flag).
    _STRIP_KEYS = ("processors", "processor_overrides", "allow_emb_target_override")
    if any(data_cfg.get(k, None) is not None for k in _STRIP_KEYS):
        OmegaConf.set_struct(data_cfg, False)
        data_cfg = OmegaConf.create(
            {k: v for k, v in data_cfg.items() if k not in _STRIP_KEYS}
        )
    return instantiate(data_cfg, **kwargs)

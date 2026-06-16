import torch
import logging
import itertools
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from g05.utils.logging.log_box import log_box

logger = logging.getLogger(__name__)


@contextmanager
def _init_empty_weights_safe():
    """Run init_empty_weights while patching nn.Module.to and nn.Module.load_state_dict.

    accelerate.init_empty_weights creates all parameters as meta tensors, which has
    three compatibility issues:

    1. nn.Module.to is not patched: transformers/tokenizer base classes call
       self.to(device) inside __init__, triggering "Cannot copy out of meta tensor".
       Fix: directly return self when meta parameters are detected.

    2. load_state_dict calls inside __init__ are moved back to meta again by the
       register_parameter patch from init_empty_weights, so they become no-ops even
       with assign=True. Fix: defer these calls and let the caller replay them at
       the right time.

    3. auto-replay happens at the wrong time: G05Policy.__init__ first defers the
       VLM pretrained load (vocab=257216), then resizes embeddings to 261252; auto
       replay crashes on a shape conflict. Fix: yield deferred_loads and let
       load_model_from_checkpoint() replay only modules that still have meta
       parameters after the outer VLA checkpoint is loaded.

    Usage:
        with _init_empty_weights_safe() as deferred:
            model = instantiate(cfg)
        # ... load the outer VLA checkpoint first ...
        # replay modules that still have meta parameters, i.e. independent
        # checkpoints not covered by the VLA checkpoint.
        for module, sd, strict, kwargs in deferred:
            if any(p.is_meta for p in itertools.chain(
                    module.parameters(), module.buffers())):
                module.load_state_dict(sd, strict=strict, assign=True, **kwargs)
    """
    from accelerate import init_empty_weights
    from torch.nn.modules.module import _IncompatibleKeys
    import torch.nn as nn

    original_module_to = nn.Module.to
    original_load_state_dict = nn.Module.load_state_dict
    deferred_loads: list = []  # [(module, state_dict_copy, strict, extra_kwargs)]

    def _to_noop_for_meta(self, *args, **kwargs):
        first = next(itertools.chain(self.parameters(), self.buffers()), None)
        if first is not None and first.is_meta:
            return self
        return original_module_to(self, *args, **kwargs)

    def _defer_load_for_meta(self, state_dict, strict=True, assign=False, **kwargs):
        first = next(itertools.chain(self.parameters(), self.buffers()), None)
        if first is not None and first.is_meta:
            deferred_loads.append((self, dict(state_dict), strict, kwargs))
            return _IncompatibleKeys([], [])
        return original_load_state_dict(self, state_dict, strict=strict, assign=assign, **kwargs)

    with init_empty_weights():
        nn.Module.to = _to_noop_for_meta
        nn.Module.load_state_dict = _defer_load_for_meta
        try:
            yield deferred_loads  # caller receives the list; no auto-replay
        finally:
            nn.Module.to = original_module_to
            nn.Module.load_state_dict = original_load_state_dict
    # No auto-replay: load_model_from_checkpoint controls replay timing.


def _partial_load_named_param(
    key: str, local_param: torch.Tensor, ckpt_param: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Partially load a shape-mismatched parameter.

    Two strategies:
    - dim-0 (rows): copy as many leading rows as possible; zero-fill or truncate the rest.
      Used for vocab embeddings, output projections, decoders, biases.
    - last-dim (cols): copy as many leading cols on the last axis as possible.
      Used for input projections and encoders whose action_dim may change.

    Key patterns and their strategy:
      dim-0  : vlm.input_proj / vlm.output_proj, embed_tokens, lm_head,
               action_expert.output_proj, action_decoder (weight & bias)
      last-dim: action_expert.input_proj, action_encoder (weight & bias)

    Returns:
        (patched_tensor, warning_message) or (None, None) if key is not recognized.
    """

    def _msg(dim_name: str, local_dim: int, ckpt_dim: int) -> str:
        n = min(local_dim, ckpt_dim)
        if ckpt_dim > local_dim:
            return (
                f"Partial load for {key}: loaded {n}/{local_dim} {dim_name} "
                f"(ckpt={ckpt_dim}, truncated extra {ckpt_dim - local_dim})"
            )
        return (
            f"Partial load for {key}: loaded {n}/{local_dim} {dim_name} "
            f"(ckpt={ckpt_dim}, zero-initialized remaining {local_dim - ckpt_dim})"
        )

    # Patterns handled by dim-0 copy
    DIM0_PATTERNS = (
        "vlm.input_proj.weight",
        "vlm.output_proj.weight",
        "embed_tokens",
        "lm_head",
        "action_expert.output_proj",  # covers .weight and .bias
        "action_decoder",  # covers weight [action_dim, hidden] and bias [action_dim]
    )
    # Patterns handled by last-dim copy
    LAST_DIM_PATTERNS = (
        "action_expert.input_proj.weight",
        "action_encoder",  # covers weight [hidden, action_dim] and bias [hidden]
        "proprio_embedder.mlp.0.weight",  # [hidden, proprio_dim]
    )

    if any(p in key for p in DIM0_PATTERNS):
        patched = torch.zeros(local_param.shape, dtype=local_param.dtype, device="cpu")
        n = min(local_param.shape[0], ckpt_param.shape[0])
        patched[:n] = ckpt_param[:n]
        return patched, _msg("rows", local_param.shape[0], ckpt_param.shape[0])

    if any(p in key for p in LAST_DIM_PATTERNS):
        patched = torch.zeros(local_param.shape, dtype=local_param.dtype, device="cpu")
        n = min(local_param.shape[-1], ckpt_param.shape[-1])
        patched[..., :n] = ckpt_param[..., :n]
        return patched, _msg("cols", local_param.shape[-1], ckpt_param.shape[-1])

    return None, None


def remap_vocab_rows_by_token(
    ckpt_weight: torch.Tensor,
    local_weight: torch.Tensor,
    old_token_to_id: dict[str, int],
    new_token_to_id: dict[str, int],
) -> tuple[torch.Tensor, int]:
    """Partially load vocab rows, then repair shifted added-token rows by token identity.

    This is for checkpoints where the added-token tail changed order, e.g. enabling
    BAR inserts <bos_blk>/<eos_blk>/<pad_action_token> before existing <EOV>/<state>.
    Prefix rows still load normally so base vocab/action rows are preserved. Matching
    named tokens are then restored to their current IDs; tokens absent from the old map
    keep the current model initialization.
    """
    if local_weight.is_meta:
        local_init = torch.zeros(local_weight.shape, dtype=local_weight.dtype, device="cpu")
    else:
        local_init = local_weight.detach().cpu().clone()

    patched = local_init.clone()
    n_prefix = min(local_weight.shape[0], ckpt_weight.shape[0])
    patched[:n_prefix].copy_(ckpt_weight[:n_prefix].to(patched.dtype).cpu())

    for token, new_id in new_token_to_id.items():
        if token in old_token_to_id:
            continue
        if new_id < 0 or new_id >= patched.shape[0]:
            continue
        patched[new_id].copy_(local_init[new_id])

    copied = 0
    for token, old_id in old_token_to_id.items():
        new_id = new_token_to_id.get(token)
        if new_id is None:
            continue
        if old_id < 0 or old_id >= ckpt_weight.shape[0]:
            continue
        if new_id < 0 or new_id >= patched.shape[0]:
            continue
        patched[new_id].copy_(ckpt_weight[old_id].to(patched.dtype).cpu())
        copied += 1
    return patched, copied


def _build_bar_token_row_remap_config(model, state_dict: Dict[str, torch.Tensor]) -> dict | None:
    """Build token-row remap config for loading non-BAR checkpoints into BAR models."""
    action_tokenizer = getattr(model, "action_tokenizer", None)
    if action_tokenizer is None or not getattr(
        action_tokenizer, "block_wise_autoregressive", False
    ):
        return None

    processor = getattr(model, "processor", None)
    get_token_map = getattr(processor, "get_added_token_id_map", None)
    if get_token_map is None:
        return None

    current_token_to_id = get_token_map()
    bar_tokens = ("<bos_blk>", "<eos_blk>", "<pad_action_token>")
    if not all(tok in current_token_to_id for tok in bar_tokens):
        return None

    local_state = model.state_dict()
    candidate_keys = [
        "model.vlm.input_proj.weight",
        "model.vlm.output_proj.weight",
    ]
    keys = [
        key
        for key in candidate_keys
        if key in state_dict
        and key in local_state
        and len(state_dict[key].shape) >= 2
        and len(local_state[key].shape) >= 2
        and state_dict[key].shape[0] != local_state[key].shape[0]
    ]
    if not keys:
        return None

    ckpt_vocab = int(state_dict[keys[0]].shape[0])
    current_vocab = int(local_state[keys[0]].shape[0])
    inserted_count = len(bar_tokens)
    if current_vocab - ckpt_vocab != inserted_count:
        logger.info(
            "Token-row remap skipped: BAR model detected but vocab delta is %s, expected %s",
            current_vocab - ckpt_vocab,
            inserted_count,
        )
        return None

    insertion_id = min(current_token_to_id[tok] for tok in bar_tokens)
    old_token_to_id: dict[str, int] = {}
    for token, new_id in current_token_to_id.items():
        if token in bar_tokens:
            continue
        old_id = new_id - inserted_count if new_id > insertion_id else new_id
        if 0 <= old_id < ckpt_vocab:
            old_token_to_id[token] = old_id

    moved = []
    for token in ("<EOV>", "<state>"):
        if token in old_token_to_id and token in current_token_to_id:
            moved.append(f"{token} {old_token_to_id[token]}->{current_token_to_id[token]}")
    moved_msg = "; ".join(moved) if moved else "no shifted named tokens"
    logger.warning(
        "Token row remap enabled for BAR checkpoint load: ckpt_vocab=%s current_vocab=%s; "
        "%s; BAR tokens are new",
        ckpt_vocab,
        current_vocab,
        moved_msg,
    )

    return {
        "old_token_to_id": old_token_to_id,
        "new_token_to_id": current_token_to_id,
        "keys": keys,
    }


def load_state_dict_safely(
    model,
    state_dict,
    extra_prefixes=None,
    load_config: Optional[Dict[str, Any]] = None,
    return_info: bool = False,
):
    """
    Optimized version: minimize memory overhead by avoiding full model_dict creation.

    Args:
        model: The model to load weights into
        state_dict: The checkpoint state dict
        extra_prefixes: List of key prefixes to load even if not in model (e.g., ['normalizer.'])
        load_config: Optional dict with loading configuration:
            - key_prefix_to_remove: str, remove this prefix from checkpoint keys (e.g., 'model.')
            - ignore_key_prefixes: List[str], ignore keys starting with these prefixes
            - partial_load_keys: List[str], keys that support partial loading (dim mismatch)
            - verbose: bool, print detailed diagnostics
        return_info: If True, return a dict with diagnostics (model/loaded_count/truly_missing/
            partial_loaded_keys/mismatched_keys/unexpected_keys/extra_loaded_keys) instead of
            just the model. Use this in offline eval to assert no unexpected missing keys.

    Returns:
        nn.Module by default; if return_info=True, a dict with keys:
            model, loaded_count, truly_missing, partial_loaded_keys,
            mismatched_keys, unexpected_keys, extra_loaded_keys.
    """
    if extra_prefixes is None:
        extra_prefixes = ["normalizer."]

    if load_config is None:
        load_config = {}

    key_prefix_to_remove = load_config.get("key_prefix_to_remove", None)
    ignore_key_prefixes = load_config.get("ignore_key_prefixes", [])
    partial_load_keys = load_config.get(
        "partial_load_keys",
        [
            "embed_tokens",
            "action_encoder",
            "action_decoder",
            "lm_head",
            "vlm.input_proj.weight",
            "vlm.output_proj.weight",
            "action_expert.input_proj.weight",
            "action_expert.output_proj.weight",
            "action_expert.output_proj.bias",
            "proprio_embedder.mlp.0.weight",
        ],
    )
    verbose = load_config.get("verbose", False)
    token_row_remap = load_config.get("token_row_remap")
    token_row_remap_keys = set(token_row_remap.get("keys", [])) if token_row_remap else set()

    # 0. Preprocess checkpoint keys
    if verbose:
        logger.info("=== Checkpoint Preprocessing ===")
        logger.info(f"Total keys in checkpoint before processing: {len(state_dict)}")

    # Defensive copy to avoid mutating the caller's dict
    state_dict = dict(state_dict)

    # Remove prefix if specified
    if key_prefix_to_remove:
        new_state_dict = {}
        removed_prefix_count = 0
        for k, v in state_dict.items():
            if k.startswith(key_prefix_to_remove):
                new_key = k[len(key_prefix_to_remove) :]
                new_state_dict[new_key] = v
                removed_prefix_count += 1
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        if verbose:
            logger.info(f"Removed prefix '{key_prefix_to_remove}' from {removed_prefix_count} keys")

    # Filter ignored keys
    if ignore_key_prefixes:
        filtered_keys = []
        for k in list(state_dict.keys()):
            if any(k.startswith(prefix) for prefix in ignore_key_prefixes):
                filtered_keys.append(k)
        for k in filtered_keys:
            del state_dict[k]
        if verbose:
            logger.info(
                f"Filtered out {len(filtered_keys)} keys with ignored prefixes: {ignore_key_prefixes}"
            )

    if verbose:
        logger.info(f"Total keys in checkpoint after processing: {len(state_dict)}")

    # 1. Get the full model state_dict for key lookup and shape comparison.
    local_state = model.state_dict()

    if verbose:
        logger.info("=== Checkpoint Key Analysis ===")
        logger.info(f"Total keys in model: {len(local_state)}")

        # Analyze checkpoint keys by category
        ckpt_categories = {
            "action_encoder": [],
            "action_decoder": [],
            "embed_tokens": [],
            "lm_head": [],
            "mixtures_vlm": [],
            "mixtures_action": [],
            "mixtures_proprio": [],
            "vision": [],
            "projector": [],
            "other": [],
        }
        for k in state_dict.keys():
            categorized = False
            for category in ckpt_categories:
                if category in k:
                    ckpt_categories[category].append(k)
                    categorized = True
                    break
            if not categorized:
                ckpt_categories["other"].append(k)

        for category, keys in ckpt_categories.items():
            if keys:
                logger.info(f"  {category}: {len(keys)} keys")
                if category in ["action_encoder", "action_decoder", "embed_tokens", "lm_head"]:
                    for k in keys[:3]:
                        logger.info(f"    {k}: {state_dict[k].shape}")

    # 2. Prepare a clean dict containing only tensors we want to update.
    update_dict = {}

    mismatched_keys = []
    partial_loaded_keys = []
    unexpected_keys = []
    extra_loaded_keys = []
    loaded_keys_count = 0

    # 3. Iterate over the checkpoint on CPU.
    for key, ckpt_param in state_dict.items():
        if key in local_state:
            local_param = local_state[key]

            # Shape matches: add directly to the update list.
            if local_param.shape == ckpt_param.shape:
                update_dict[key] = ckpt_param
                loaded_keys_count += 1
            else:
                # Shape mismatch: handle by type.
                mismatched_keys.append((key, local_param.shape, ckpt_param.shape))

                # Check if this key supports partial loading
                is_partial_load_key = any(partial_key in key for partial_key in partial_load_keys)

                if is_partial_load_key:
                    if token_row_remap and key in token_row_remap_keys:
                        patched, copied = remap_vocab_rows_by_token(
                            ckpt_param,
                            local_param,
                            token_row_remap["old_token_to_id"],
                            token_row_remap["new_token_to_id"],
                        )
                        logger.warning(
                            f"Token-row remap for {key}: copied {copied} token rows by token "
                            "string; remaining rows keep current initialization"
                        )
                        update_dict[key] = patched
                        partial_loaded_keys.append(key)
                        loaded_keys_count += 1
                        continue

                    patched, partial_message = _partial_load_named_param(
                        key, local_param, ckpt_param
                    )
                    if patched is None:
                        logger.warning(
                            f"Shape mismatch for {key}: model={local_param.shape}, "
                            f"ckpt={ckpt_param.shape} - keeping random init"
                        )
                        continue
                    logger.warning(partial_message)
                    update_dict[key] = patched
                    partial_loaded_keys.append(key)
                    loaded_keys_count += 1
                else:
                    logger.warning(
                        f"Shape mismatch for {key}: model={local_param.shape}, ckpt={ckpt_param.shape} - keeping random init"
                    )
        else:
            # Present in checkpoint but absent from model.
            # Check whether it belongs to extra_prefixes.
            if any(key.startswith(prefix) for prefix in extra_prefixes):
                update_dict[key] = ckpt_param
                extra_loaded_keys.append(key)
                logger.info(f"Loading extra key: {key}")
            else:
                unexpected_keys.append(key)

    # 4. Perform loading.
    # strict=False automatically ignores keys absent from update_dict but present in
    # the model, i.e. missing keys. This avoids copying unchanged parameters.
    # assign=True is required for meta-device models; otherwise copy-in-place is a no-op.
    has_meta = any(p.is_meta for p in model.parameters())

    # Dtype alignment: on the assign=True path, PyTorch no longer performs implicit
    # casts. Mixed-dtype checkpoint tensors, such as a pi05 base with ~70% bf16, would
    # be assigned into model params as-is, causing large attention/MLP weights to be
    # updated by Adam in bf16 with severe precision loss. It also breaks downstream
    # invariants such as apply_fp32_params expecting the model to have one dtype before
    # it runs. Reproduce the old assign=False copy_into semantics by casting checkpoint
    # tensors to the dtype expected by the model; meta param dtype is defined at
    # instantiation time and defaults to fp32 for nn.Module.
    for k in list(update_dict.keys()):
        if k in local_state and update_dict[k].dtype != local_state[k].dtype:
            update_dict[k] = update_dict[k].to(local_state[k].dtype)

    msg = model.load_state_dict(update_dict, strict=False, assign=has_meta)
    # Ensure lm_head weights are shared with embed_tokens for old/new model structures.
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "embed_tokens") and hasattr(inner, "lm_head"):
            # Old: GalaxeaJoint
            inner.lm_head.weight = inner.embed_tokens.weight
        elif hasattr(inner, "vlm"):
            # New: G05Model — vlm.output_proj tied to vlm.input_proj
            inner.vlm.output_proj.weight = inner.vlm.input_proj.weight

    # 5. Manual cleanup.
    del update_dict
    torch.cuda.empty_cache()

    # 6. Logging — structured box
    manually_handled = {k for k, _, _ in mismatched_keys}
    truly_missing = [k for k in msg.missing_keys if k not in manually_handled]
    total_model_keys = len(local_state)

    _loaded_icon = "✅" if loaded_keys_count == total_model_keys else "⚠️ "
    _rows = [
        ("Loaded", f"{loaded_keys_count} / {total_model_keys}  {_loaded_icon}"),
        (
            "Partial (handled)",
            f"{len(partial_loaded_keys)} params  (dim mismatch, auto-fixed)"
            if partial_loaded_keys
            else "0",
        ),
        ("Missing (rand init)", f"{len(truly_missing)} params" if truly_missing else "0"),
        (
            "Unexpected (skip)",
            f"{len(unexpected_keys)} params  (not in model)" if unexpected_keys else "0",
        ),
    ]
    if extra_loaded_keys:
        _rows.append(("Extra keys (loaded)", f"{len(extra_loaded_keys)} params  (e.g. normalizer)"))

    # Detail lines for partial keys (always show all — typically ≤10)
    if partial_loaded_keys:
        _rows.append(None)
        _rows.append("Partial keys:")
        for k in partial_loaded_keys:
            _rows.append(f"  {k}")

    # Detail lines for missing keys (cap at 5)
    if truly_missing:
        _rows.append(None)
        _rows.append(f"Missing keys (first {min(5, len(truly_missing))}):")
        for k in truly_missing[:5]:
            _rows.append(f"  {k}")
        if len(truly_missing) > 5:
            _rows.append(f"  ... and {len(truly_missing) - 5} more")

    # unexpected keys detail (cap at 5)
    if unexpected_keys:
        _rows.append(None)
        _rows.append(f"Unexpected keys (first {min(5, len(unexpected_keys))}):")
        for k in list(unexpected_keys)[:5]:
            _rows.append(f"  {k}")
        if len(unexpected_keys) > 5:
            _rows.append(f"  ... and {len(unexpected_keys) - 5} more")

    # Shape-mismatched (not loaded)
    remaining_mismatched = [item for item in mismatched_keys if item[0] not in partial_loaded_keys]
    if remaining_mismatched:
        _rows.append(None)
        _rows.append("Shape mismatch (not loaded):")
        for k, ms, cs in remaining_mismatched[:5]:
            _rows.append(f"  {k}: model={ms} ckpt={cs}")

    log_box(logger, "📊  VLA Checkpoint — Load Summary", _rows)

    if return_info:
        return {
            "model": model,
            "loaded_count": loaded_keys_count,
            "truly_missing": truly_missing,
            "partial_loaded_keys": partial_loaded_keys,
            "mismatched_keys": mismatched_keys,
            "unexpected_keys": unexpected_keys,
            "extra_loaded_keys": extra_loaded_keys,
        }
    return model


def load_model_from_checkpoint(
    model_arch_cfg,
    ckpt_path: str,
    device: str = "cuda",
    state_dict_key: str = "model_state_dict",
    extra_prefixes=None,
    load_config=None,
    use_meta_device: bool = False,
    eval_mode: bool = True,
    return_full_checkpoint: bool = False,
):
    """Load a model and apply checkpoint weights, optionally using meta-device acceleration.

    Compared with the old three-step path (instantiate -> torch.load ->
    load_state_dict_safely -> .cuda()), this function provides:
    - use_meta_device=True (default): use accelerate.init_empty_weights to skip random
      initialization, then combine mmap=True and map_location=device to load weights
      directly onto the target device and avoid a CPU staging buffer. It also nulls
      pretrained_model_path to skip unnecessary HF safetensors I/O.
    - use_meta_device=False: fall back to legacy behavior with full backward compatibility.

    Args:
        model_arch_cfg: Hydra model_arch config passed to hydra.utils.instantiate()
        ckpt_path: checkpoint file path in .pt format containing state_dict_key
        device: target device, default "cuda"
        state_dict_key: key for model weights in the checkpoint dict, default "model_state_dict"
        extra_prefixes: forwarded to load_state_dict_safely, default ["normalizer."]
        load_config: forwarded to load_state_dict_safely
        use_meta_device: True enables the fast path; False falls back to legacy behavior
        eval_mode: True (default) calls model.eval() before returning; finetune passes False
        return_full_checkpoint: True also returns the full checkpoint dict for optimizer state

    Returns:
        eval_mode=True: nn.Module, already .eval(), with weights on device
        return_full_checkpoint=True: (nn.Module, dict)
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if use_meta_device:
        # Null out pretrained_model_path to skip ~6GB HF safetensors I/O.
        # Weights come entirely from the VLA checkpoint; HF backbone weights are not needed.
        # Mirror the logic in ckpt_utils.py:89-91 used by the eval/serve path.
        _original_pretrained = model_arch_cfg.get("pretrained_model_path", None)
        if _original_pretrained:
            OmegaConf.set_struct(model_arch_cfg, False)
            model_arch_cfg.hf_processor_path = (
                model_arch_cfg.get("hf_processor_path", None) or _original_pretrained
            )
            model_arch_cfg.pretrained_model_path = None
            OmegaConf.set_struct(model_arch_cfg, True)
            logger.info(
                "load_model_from_checkpoint: nullified pretrained_model_path "
                "(weights come from VLA ckpt, skipping HF safetensors I/O)"
            )

        with _init_empty_weights_safe() as deferred_loads:
            model = instantiate(model_arch_cfg)
    else:
        deferred_loads = []
        model = instantiate(model_arch_cfg)

    full_ckpt = torch.load(
        ckpt_path,
        map_location=device if use_meta_device else "cpu",
        mmap=True,
        weights_only=False,
    )
    state_dict = full_ckpt[state_dict_key]

    if load_config is None:
        load_config = {}
    else:
        load_config = dict(load_config)
    if load_config.get("auto_token_row_remap", True) and "token_row_remap" not in load_config:
        token_row_remap = _build_bar_token_row_remap_config(model, state_dict)
        if token_row_remap is not None:
            load_config["token_row_remap"] = token_row_remap

    load_state_dict_safely(
        model,
        state_dict,
        extra_prefixes=extra_prefixes,
        load_config=load_config,
    )

    # After the VLA checkpoint is loaded, replay submodules that still have meta
    # parameters, such as ActionCodecV2Wrapper. Modules covered by the VLA checkpoint
    # (VLM backbone, vision tower, etc.) no longer have meta parameters and are
    # skipped automatically, avoiding overwriting correctly resized embeddings with
    # pretrained weights from the old vocab size.
    if use_meta_device and deferred_loads:
        for module, sd, strict, kwargs in deferred_loads:
            has_meta = any(
                p.is_meta for p in itertools.chain(module.parameters(), module.buffers())
            )
            if has_meta:
                module.load_state_dict(sd, strict=strict, assign=True, **kwargs)

    # Fail fast: remaining meta parameters mean the checkpoint lacks weights for new
    # modules. Continuing to .to(device) will crash with NotImplementedError: Cannot
    # copy out of meta tensor, and warnings are easy to miss.
    remaining_meta = [name for name, param in model.named_parameters() if param.is_meta]
    if remaining_meta:
        # Group by top-level module so it is obvious which submodule is not covered
        # by the checkpoint.
        from collections import defaultdict

        groups: Dict[str, List[str]] = defaultdict(list)
        for name in remaining_meta:
            top = ".".join(name.split(".")[:3])  # e.g. model.proprio_embedder.mlp
            groups[top].append(name)
        group_lines = "\n".join(
            f"  [{len(keys):>3d}] {top}  (e.g. {keys[0]})"
            for top, keys in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        )
        raise RuntimeError(
            f"load_model_from_checkpoint: {len(remaining_meta)} parameters remain "
            f"on meta device after loading VLA ckpt — these modules exist in the "
            f"current model but have NO weights in the checkpoint:\n"
            f"{group_lines}\n"
            f"Likely causes:\n"
            f"  1. ckpt is older than current model code (new nn.Module added since)\n"
            f"  2. config enables a sub-module the ckpt doesn't have "
            f"(e.g. proprio_encoder=mlp, use_qk_norm=true) — override in task yaml\n"
            f"  3. OR pass use_meta_device=False to fall back to default init for "
            f"missing keys (slower load, larger CPU peak)"
        )

    # Buffers with persistent=False, such as siglip position_ids, are not in the
    # state_dict. model.to(device) moves them to the target device; it is a no-op for
    # parameters already on device.
    model = model.to(device)

    if eval_mode:
        model = model.eval()

    if return_full_checkpoint:
        return model, full_ckpt
    return model


# ---------------------------------------------------------------------------
# Training checkpoint save / resume (used by scripts/finetune.py)
# ---------------------------------------------------------------------------


def save_training_checkpoint(
    output_dir,
    step: int,
    epoch: int,
    batch_idx: int,
    model=None,
    optimizer=None,
    scheduler=None,
    ema_model=None,
    **extra_state,
):
    """Save training state to ``output_dir/checkpoints/step_{step}.pt``.

    Also refreshes the ``output_dir/last.pt`` symlink to point at the new
    checkpoint. Pass ``optimizer=None`` / ``scheduler=None`` for a final,
    inference-only checkpoint. Extra keyword args (e.g. ``action_batch_idx``)
    are stored verbatim in the checkpoint dict.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    path = output_dir / "checkpoints" / f"step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict() if model is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "ema_model_state_dict": ema_model.ema_model.state_dict() if ema_model is not None else None,
    }
    state.update(extra_state)
    torch.save(state, path)

    last_pt_path = output_dir / "last.pt"
    if last_pt_path.exists() or last_pt_path.is_symlink():
        last_pt_path.unlink()
    last_pt_path.symlink_to(path.relative_to(output_dir))
    return path


def fix_optimizer_state_after_resume(optimizer) -> int:
    """Repair optimizer state loaded from a checkpoint with mismatched params.

    The checkpoint may come from a model with a different architecture
    (e.g., different vocab size → different embedding shape) or dtype
    (e.g., bf16 state vs fp32 params). Fused Adam bundles all params in a
    group into one CUDA kernel and requires matching dtype AND shape between
    each param and its state tensors.

    Casts dtype-mismatched state tensors in place and reinitializes state for
    shape-mismatched params. Returns the number of params whose state was reset.
    """
    reset_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None:
                continue
            need_reset = False
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point() and v.dtype != p.dtype:
                        state[k] = v.to(dtype=p.dtype)
                    if v.shape != p.shape:
                        need_reset = True
            if need_reset:
                reset_count += 1
                optimizer.state[p] = {}
    return reset_count

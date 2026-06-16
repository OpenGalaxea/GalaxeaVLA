import time
import torch
import logging
from collections import OrderedDict

from g05.tokenizer.interface.base_action_tokenizer import DecodeResult

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator

logger = logging.getLogger(__name__)


def eval_tokenizer_first_batch(action_tokenizer, batch, device_id, *, hf_tokenizer=None):
    """Encode→Decode the first batch with the action tokenizer and log reconstruction error.

    Prefers vlm_action (from batch["samples"]) as tokenizer input, since it has been
    through rotation transform + dummy normalization — matching tokenizer training data.
    Falls back to gt_action for ActionProcessor-based training where gt_action is already
    post-transform.

    Prints:
    - First sample's action tokens
    - Per-part L1 reconstruction error (in tokenizer input space)
    """
    # --- Resolve tokenizer input: prefer vlm_action over gt_action ---
    # In VLA training (GalaxeaCoTProcessor), gt_action is saved BEFORE rotation transform,
    # so it's in raw euler format — incompatible with the tokenizer (trained on rotation_6d).
    # vlm_action (from batch["samples"]["action"]["value"]) is the actual tokenizer input:
    # post-rotation-transform + dummy-normalized.
    # In action-only training (ActionProcessor), gt_action is already post-transform, so it works.
    tokenizer_input = None
    input_source = None
    batch_input_parts_meta = None

    # Try vlm_action from batch["samples"] (VLA path)
    samples = batch.get("samples")
    if samples is not None and isinstance(samples, list) and len(samples) > 0:
        first_sample = samples[0]
        if isinstance(first_sample, dict) and "action" in first_sample:
            action_entry = first_sample["action"]
            if isinstance(action_entry, dict) and "value" in action_entry:
                vlm_actions = [s["action"]["value"] for s in samples]
                tokenizer_input = torch.stack(vlm_actions, dim=0)
                input_source = "vlm_action"
                # Extract input_parts_meta from action entry (set by GalaxeaCoTProcessor)
                batch_input_parts_meta = action_entry.get("parts_meta", None)

    # Fallback to gt_action
    if tokenizer_input is None:
        gt_action = batch.get("gt_action")
        if gt_action is None:
            logger.warning(
                "[First Batch Tokenizer Eval] No vlm_action or gt_action found, skipping"
            )
            return

        if isinstance(gt_action, list):
            if isinstance(gt_action[0], dict):
                gt_action = torch.stack(
                    [torch.cat([v for v in d.values()], dim=-1) for d in gt_action], dim=0
                )
            else:
                gt_action = torch.stack(gt_action, dim=0)

        tokenizer_input = gt_action
        input_source = "gt_action"
        # Fallback: try top-level action_parts_meta (ActionProcessor path)
        batch_input_parts_meta = batch.get("action_parts_meta", None)

    tokenizer_input = tokenizer_input.to(device_id)

    action_op_mask = batch.get("action_op_mask")
    action_dim_is_pad = batch.get("action_dim_is_pad")
    time_horizon = tokenizer_input.shape[1]
    action_dim = tokenizer_input.shape[2]
    batch_size = tokenizer_input.shape[0]

    # Get parts_meta from tokenizer
    parts_meta = None
    if hasattr(action_tokenizer, "parts_meta") and action_tokenizer.parts_meta is not None:
        parts_meta = action_tokenizer.parts_meta
    elif hasattr(action_tokenizer, "action_tokenizer"):
        backend = action_tokenizer.action_tokenizer
        if hasattr(backend, "parts_meta") and backend.parts_meta is not None:
            parts_meta = backend.parts_meta
        elif hasattr(backend, "key_dims") and backend.key_dims:
            parts_meta = backend.key_dims
        elif hasattr(backend, "canonical_parts_meta") and backend.canonical_parts_meta is not None:
            parts_meta = backend.canonical_parts_meta
        elif hasattr(backend, "_parts_meta") and backend._parts_meta:
            parts_meta = backend._parts_meta
        elif hasattr(backend, "_original_component_dims"):
            parts_meta = backend._original_component_dims

    if parts_meta is None:
        # Fallback: infer from action_dim_is_pad or use single part
        parts_meta = OrderedDict([("action", action_dim)])

    # Build encode/decode kwargs
    frequency = batch.get("frequency")

    backend_encode_kw = {}
    backend_decode_kw = {}
    if frequency is not None:
        backend_encode_kw["frequency"] = frequency.to(device_id)
        backend_decode_kw["frequency"] = frequency.to(device_id)
    if action_dim_is_pad is not None:
        backend_encode_kw["action_dim_is_pad"] = action_dim_is_pad.to(device_id)
        backend_decode_kw["action_dim_is_pad"] = action_dim_is_pad.to(device_id)
    if action_op_mask is not None:
        backend_encode_kw["action_op_mask"] = action_op_mask.to(device_id)
    if batch_input_parts_meta is not None:
        backend_encode_kw["input_parts_meta"] = batch_input_parts_meta

    with torch.no_grad():
        backend = getattr(action_tokenizer, "action_tokenizer", action_tokenizer)

        import inspect

        try:
            backend_sig = set(inspect.signature(backend.encode).parameters.keys())
        except (ValueError, TypeError):
            backend_sig = set()
        backend_safe_kw = {k: v for k, v in backend_encode_kw.items() if k in backend_sig}
        codes = backend.encode(tokenizer_input, **backend_safe_kw)

        frontend_encode_kw = {"return_token_dict": True}
        frontend_encode_kw.update(backend_encode_kw)
        token_dicts = action_tokenizer._encode_action_indices(
            tokenizer_input, encode_kwargs=frontend_encode_kw
        )

        # Decode: use backend.decode which handles external→internal key mapping.
        # Strip _input_action_dim to prevent naive prefix unpadding — for cross-embodiment
        # subsets, unpad_action_to_input_dim slices the first N dims which is wrong
                # (for example when right_ee_pose lives at a nonzero offset in the model layout).
        decode_codes = {k: v for k, v in codes.items() if k != "_input_action_dim"}
        # In subset mode use _repr_info from codes rather than action_dim_is_pad:
        # passing a 10D dim_is_pad to decode_from_codes would misdetect representation.
        if batch_input_parts_meta is not None and "_repr_info" in codes:
            decoded = backend.decode(decode_codes)
        else:
            backend_safe_decode_kw = {
                k: v for k, v in backend_decode_kw.items() if k in backend_sig
            }
            decoded = backend.decode(decode_codes, **backend_safe_decode_kw)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        decoded = decoded.to(device_id)

        # If input was a cross-embodiment subset, pad tokenizer_input to model space so
        # both tensors have matching dims for comparison.
        compare_input = tokenizer_input
        compare_dim_is_pad = (
            action_dim_is_pad.to(device_id) if action_dim_is_pad is not None else None
        )
        if decoded.shape[-1] != tokenizer_input.shape[-1]:
            if decoded.shape[-1] > compare_input.shape[-1] and batch_input_parts_meta is not None:
                # Input is a subset of decoded space — pad input up to decoded dim.
                if hasattr(backend, "parts_meta"):
                    from g05.tokenizer.utils.parts_meta_padding import (
                        pad_action_to_model_dim,
                    )

                    padded_input, padded_dim_is_pad, _, is_subset = pad_action_to_model_dim(
                        tokenizer_input,
                        batch_input_parts_meta,
                        backend.parts_meta,
                        compare_dim_is_pad,
                    )
                    if is_subset:
                        compare_input = padded_input
                        compare_dim_is_pad = padded_dim_is_pad
                        action_dim = compare_input.shape[2]
            elif decoded.shape[-1] < compare_input.shape[-1]:
                # Decoded is smaller than input (e.g. merge_spec tokenizer only handles a
                # subset of the full VLA action space).  Extract the tokenizer's known dims
                # from compare_input using semantic key matching.
                if (
                    batch_input_parts_meta is not None
                    and hasattr(backend, "parts_meta")
                    and backend.parts_meta is not None
                ):
                    try:
                        from g05.tokenizer.utils.parts_meta_padding import (
                            unpad_action_by_parts_meta,
                        )

                        compare_input = unpad_action_by_parts_meta(
                            compare_input,
                            backend.parts_meta,
                            batch_input_parts_meta,
                        )
                    except Exception:
                        compare_input = compare_input[..., : decoded.shape[-1]]
                else:
                    compare_input = compare_input[..., : decoded.shape[-1]]
                compare_dim_is_pad = None
                action_dim = compare_input.shape[2]

        # Compute overall losses (excluding padded dims)
        if compare_dim_is_pad is not None:
            valid_mask = ~compare_dim_is_pad.bool()  # (B, D)
            valid_mask_3d = valid_mask.unsqueeze(1).expand_as(compare_input)
            abs_diff = torch.abs(compare_input - decoded)
            sq_diff = (compare_input - decoded) ** 2
            l1_loss = abs_diff[valid_mask_3d].mean()
            mse_loss = sq_diff[valid_mask_3d].mean()
        else:
            l1_loss = torch.abs(compare_input - decoded).mean()
            mse_loss = ((compare_input - decoded) ** 2).mean()

        # Print first sample's action tokens
        logger.info("=" * 80)
        logger.info(f"[First Batch Tokenizer Eval] (input_source={input_source})")
        logger.info("=" * 80)

        if isinstance(token_dicts, list) and len(token_dicts) > 0:
            first_sample_tokens = token_dicts[0]
            if isinstance(first_sample_tokens, dict):
                # WBC format: dict of tensors
                logger.info(f"Sample 0 tokens (per-part):")
                for key, val in first_sample_tokens.items():
                    if key == "_repr_info":
                        continue
                    if isinstance(val, torch.Tensor):
                        logger.info(f"  {key}: {val.tolist()}")
            elif isinstance(first_sample_tokens, (list, torch.Tensor)):
                # Flat token list
                tokens = (
                    first_sample_tokens.tolist()
                    if isinstance(first_sample_tokens, torch.Tensor)
                    else first_sample_tokens
                )
                logger.info(f"Sample 0 tokens: {tokens}")
        elif isinstance(token_dicts, torch.Tensor):
            logger.info(f"Sample 0 tokens: {token_dicts[0].tolist()}")

        # Compute and print per-part L1 error (in tokenizer input space)
        # Skip parts where all dims are padded (embodiment doesn't have this part)
        logger.info("-" * 80)
        logger.info(
            f"Per-part L1 reconstruction error (batch_size={batch_size}, tokenizer input space):"
        )

        input_np = compare_input[0].cpu().numpy()  # First sample
        decoded_np = decoded[0].cpu().numpy()

        # Get action_dim_is_pad for filtering padded parts
        dim_is_pad = None
        if compare_dim_is_pad is not None:
            dim_is_pad = compare_dim_is_pad[0].cpu().numpy()  # (D,) bool

        offset = 0
        total_l1 = 0.0
        total_dim = 0
        for part_name, dim in parts_meta.items():
            if offset + dim > action_dim:
                break

            if dim_is_pad is not None and dim_is_pad[offset : offset + dim].all():
                offset += dim
                continue

            part_gt = input_np[:, offset : offset + dim]
            part_pred = decoded_np[:, offset : offset + dim]

            # Only compute L1 on non-padded dims within this part
            if dim_is_pad is not None:
                part_pad = dim_is_pad[offset : offset + dim]
                active_dims = ~part_pad
                if active_dims.any():
                    part_gt = part_gt[:, active_dims]
                    part_pred = part_pred[:, active_dims]
                    active_count = int(active_dims.sum())
                else:
                    offset += dim
                    continue
            else:
                active_count = dim

            part_l1 = abs(part_gt - part_pred).mean()
            total_l1 += part_l1 * active_count
            total_dim += active_count
            logger.info(f"  {part_name:<20}: L1={part_l1:.6f} (dim={active_count})")
            offset += dim

        if total_dim > 0:
            avg_l1 = total_l1 / total_dim
            logger.info(f"  {'Sample-0 per-part avg':<20}: L1={avg_l1:.6f}")

        logger.info("-" * 80)
        logger.info(
            f"Batch-avg (tokenizer input space): L1={l1_loss.item():.6f}, MSE={mse_loss.item():.6f}"
        )
        logger.info("=" * 80)

        # --- AR Text Round-Trip (mirrors processor.decode_ar eval path) ---
        if hf_tokenizer is not None and isinstance(token_dicts, list) and len(token_dicts) > 0:
            try:
                # action-index space → HF token-ID space
                hf_ids_batch = action_tokenizer.action_indices_to_token_ids(token_dicts)

                # Diagnostic: verify text round-trip is lossless on sample 0
                _text0 = hf_tokenizer.decode(hf_ids_batch[0])
                _re0 = hf_tokenizer.encode(_text0, add_special_tokens=False)
                _trip_ok = hf_ids_batch[0] == _re0
                if not _trip_ok:
                    _n_orig, _n_re = len(hf_ids_batch[0]), len(_re0)
                    if _n_orig == _n_re:
                        _first_diff = next(
                            (i for i, (a, b) in enumerate(zip(hf_ids_batch[0], _re0)) if a != b),
                            None,
                        )
                        logger.warning(
                            f"[AR] Text round-trip LOSSY at sample 0: "
                            f"same len={_n_orig}, first diff at pos {_first_diff}, "
                            f"orig={hf_ids_batch[0][_first_diff]}, re={_re0[_first_diff]}"
                        )
                    else:
                        logger.warning(
                            f"[AR] Text round-trip LOSSY at sample 0: "
                            f"orig_len={_n_orig}, re_encoded_len={_n_re}"
                        )

                # Diagnostic: batch decode from token_dicts directly (no text step).
                # This should match the first-block direct decode L1; if it doesn't,
                # the serializer round-trip (codes → indices → decode) has a bug.
                _batch_decode_kw: dict = {}
                if action_dim_is_pad is not None:
                    _batch_decode_kw["action_dim_is_pad"] = action_dim_is_pad.to(device_id)
                _direct_np, _ = action_tokenizer._decode_action_indices(
                    token_dicts,
                    time_horizon=time_horizon,
                    action_dim=action_dim,
                    decode_kwargs=_batch_decode_kw or None,
                )
                _direct_t = torch.from_numpy(_direct_np).to(device_id)
                _direct_cmp = (
                    _direct_t[..., : compare_input.shape[-1]]
                    if _direct_t.shape[-1] != compare_input.shape[-1]
                    else _direct_t
                )

                if compare_dim_is_pad is not None:
                    _vm = (~compare_dim_is_pad.bool()).unsqueeze(1).expand_as(compare_input)
                    _direct_l1 = torch.abs(compare_input - _direct_cmp)[_vm].mean()
                else:
                    _direct_l1 = torch.abs(compare_input - _direct_cmp).mean()

                # Main: per-sample text round-trip decode (full decode_ar simulation)
                ar_decoded_list: list = []
                ar_absent_keys: list = []
                for i in range(batch_size):
                    text = hf_tokenizer.decode(hf_ids_batch[i])
                    raw_ids = hf_tokenizer.encode(text, add_special_tokens=False)
                    raw_tokens = torch.tensor(raw_ids, dtype=torch.long, device=device_id)

                    per_decode_kw: dict = {}
                    if frequency is not None:
                        per_decode_kw["frequency"] = frequency[i : i + 1].to(device_id)
                    if action_dim_is_pad is not None:
                        per_decode_kw["action_dim_is_pad"] = action_dim_is_pad[i : i + 1].to(
                            device_id
                        )

                    decoded_i = action_tokenizer.decode_token_ids_to_actions(
                        raw_tokens,
                        time_horizon=time_horizon,
                        action_dim=action_dim,
                        decode_kwargs=per_decode_kw or None,
                    )
                    if isinstance(decoded_i, DecodeResult):
                        ar_absent_keys.append(decoded_i.absent_keys)
                        decoded_i = decoded_i.action
                    else:
                        ar_absent_keys.append(set())
                    ar_decoded_list.append(decoded_i.cpu())

                ar_decoded = torch.stack(ar_decoded_list, dim=0).to(device_id)

                ar_compare = ar_decoded
                if ar_decoded.shape[-1] != compare_input.shape[-1]:
                    ar_compare = ar_decoded[..., : compare_input.shape[-1]]

                # Build absent-key dim masks for AR metric exclusion
                def _absent_keys_to_dim_mask(
                    absent_keys_per_sample: list, parts_meta_dict, bs: int, adim: int, dev
                ) -> torch.Tensor:
                    mask = torch.ones(bs, adim, dtype=torch.bool, device=dev)
                    for b, absent in enumerate(absent_keys_per_sample):
                        if b >= bs:
                            break
                        off = 0
                        for pname, d in parts_meta_dict.items():
                            if off + d > adim:
                                break
                            if pname in absent:
                                mask[b, off : off + d] = False
                            off += d
                    return mask

                _absent_mask_ar = _absent_keys_to_dim_mask(
                    ar_absent_keys, parts_meta, batch_size, action_dim, device_id
                )

                if compare_dim_is_pad is not None:
                    valid_mask_ar = ~compare_dim_is_pad.bool() & _absent_mask_ar
                    valid_mask_3d_ar = valid_mask_ar.unsqueeze(1).expand_as(compare_input)
                    ar_l1 = torch.abs(compare_input - ar_compare)[valid_mask_3d_ar].mean()
                    ar_mse = ((compare_input - ar_compare) ** 2)[valid_mask_3d_ar].mean()
                else:
                    _vm_3d_ar = _absent_mask_ar.unsqueeze(1).expand_as(compare_input)
                    ar_l1 = torch.abs(compare_input - ar_compare)[_vm_3d_ar].mean()
                    ar_mse = ((compare_input - ar_compare) ** 2)[_vm_3d_ar].mean()

                logger.info("=" * 80)
                logger.info(
                    "[First Batch Tokenizer Eval - AR Round-Trip] (mirrors processor.decode_ar)"
                )
                logger.info("=" * 80)
                logger.info("-" * 80)
                logger.info(
                    f"Per-part L1 (AR round-trip, batch_size={batch_size}, tokenizer input space):"
                )
                ar_decoded_np = ar_compare[0].cpu().numpy()
                _ar_absent_sample0 = ar_absent_keys[0] if len(ar_absent_keys) > 0 else set()
                offset = 0
                total_l1_ar = 0.0
                total_dim_ar = 0
                for part_name, dim in parts_meta.items():
                    if offset + dim > action_dim:
                        break
                    if dim_is_pad is not None and dim_is_pad[offset : offset + dim].all():
                        offset += dim
                        continue
                    if part_name in _ar_absent_sample0:
                        logger.info(f"  {part_name:<20}: L1=0.000000 (absent, no-op)")
                        offset += dim
                        continue
                    part_gt = input_np[:, offset : offset + dim]
                    part_ar = ar_decoded_np[:, offset : offset + dim]
                    if dim_is_pad is not None:
                        part_pad = dim_is_pad[offset : offset + dim]
                        active_dims = ~part_pad
                        if active_dims.any():
                            part_gt = part_gt[:, active_dims]
                            part_ar = part_ar[:, active_dims]
                            active_count = int(active_dims.sum())
                        else:
                            offset += dim
                            continue
                    else:
                        active_count = dim
                    part_l1_ar = abs(part_gt - part_ar).mean()
                    total_l1_ar += part_l1_ar * active_count
                    total_dim_ar += active_count
                    logger.info(f"  {part_name:<20}: L1={part_l1_ar:.6f} (dim={active_count})")
                    offset += dim
                if total_dim_ar > 0:
                    logger.info(
                        f"  {'Sample-0 per-part avg':<20}: L1={total_l1_ar / total_dim_ar:.6f}"
                    )
                logger.info("-" * 80)
                logger.info(f"  Text round-trip sample-0: {'LOSSLESS' if _trip_ok else 'LOSSY'}")
                logger.info(
                    f"  Direct encode-decode L1:    {l1_loss.item():.6f}  (reference, first block)"
                )
                logger.info(
                    f"  Direct from token_dicts L1: {_direct_l1.item():.6f}  (bypass text step)"
                )
                logger.info(
                    f"  AR text round-trip L1:      {ar_l1.item():.6f}, MSE={ar_mse.item():.6f}"
                )
                logger.info("=" * 80)

            except Exception as e:
                logger.warning(f"[AR Round-Trip Eval] failed: {type(e).__name__}: {e}")


@torch.no_grad()
def log_sample_text_diagnostic(
    model,
    train_dataset,
    train_processor,
    device="cuda",
    max_display_len=300,
):
    """Decode one sample per embodiment + one VQA sample to verify AR input/target text.

    Called from finetune.py before training starts (main process only).
    """
    from g05.utils.logging.log_box import log_box, _write_to_file
    from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset

    IGNORE_INDEX = -100
    # Resolve tokenizer (GemmaTokenizerFast) and InputPreprocessor
    tokenizer = model.processor.tokenizer
    processor = model.processor  # InputPreprocessor with encode_train
    max_chunk = getattr(model, "max_chunk_token_length", 2048)

    rows = []
    file_lines = []  # plain-text full version for log file (not box-formatted)

    def _add_entry(label, input_ids_t, labels_t, split_idx):
        """Decode one (input_ids, labels) pair and append rows."""
        ids = input_ids_t.cpu().tolist()
        label_ids = labels_t.cpu().tolist()

        input_text = tokenizer.decode(ids, skip_special_tokens=False)
        target_ids = [t for t in label_ids if t != IGNORE_INDEX]
        target_text = (
            tokenizer.decode(target_ids, skip_special_tokens=False) if target_ids else "(empty)"
        )

        n_masked = sum(1 for t in label_ids if t == IGNORE_INDEX)
        n_target = len(label_ids) - n_masked

        header = f"[{label}] seq_len={len(label_ids)}, masked={n_masked}, target={n_target}, split={split_idx}"
        input_line = f"  INPUT:  {input_text[:max_display_len]}{'...' if len(input_text) > max_display_len else ''}"
        target_line = f"  TARGET: {target_text[:max_display_len]}{'...' if len(target_text) > max_display_len else ''}"
        rows.extend([header, input_line, target_line, None])

        # Full version for file — plain text, no box truncation
        file_lines.append(header)
        file_lines.append(f"  INPUT:  {input_text}")
        file_lines.append(f"  TARGET: {target_text}")
        file_lines.append("")

    # --- Action embodiments ---
    # Use train_dataset (MixtureLerobotDataset) directly: its __getitem__ loads
    # images from video, adds embodiment info, and calls the processor — returning
    # a fully-processed sample dict with "samples" already built.
    if isinstance(train_dataset, MixtureLerobotDataset):
        seen_embs = set()
        # Walk through dataset groups to find one index per unique embodiment.
        # In overfit mode, _overfit_effective_starts exists; otherwise use effective_starts.
        overfit = hasattr(train_dataset, "_overfit_len")
        starts = (
            train_dataset._overfit_effective_starts if overfit else train_dataset.effective_starts
        )
        for i, emb in enumerate(train_dataset.embodiments):
            if emb in seen_embs:
                continue
            seen_embs.add(emb)
            try:
                # Get one fully-processed sample from this embodiment group
                sample = train_dataset[int(starts[i])]
                sample_dict = sample["samples"]

                # Encode through InputPreprocessor
                input_ids, labels, attn_mask, split_idx = processor.encode_train(
                    [sample_dict],
                    device=torch.device(device),
                    training=False,
                    max_chunk_token_length=max_chunk,
                )

                _add_entry(emb, input_ids[0], labels[0], split_idx)
            except Exception as e:
                rows.append(f"[{emb}] ERROR: {e}")
                rows.append(None)
                file_lines.append(f"[{emb}] ERROR: {e}")
                file_lines.append("")

    if rows:
        # Remove trailing separator
        if rows and rows[-1] is None:
            rows.pop()
        # Terminal: box with truncation
        log_box(
            logger,
            "Token Decode Diagnostic (1 sample per embodiment)",
            rows,
            inner_width=120,
        )
        # File: plain text with full content (no box truncation)
        if file_lines:
            _write_to_file("\n=== Token Decode Diagnostic (1 sample per embodiment) ===")
            for line in file_lines:
                _write_to_file(line)
            _write_to_file("=== End Token Decode Diagnostic ===\n")


_GLOBAL_MONITOR = None


class GlobalMonitor:
    """
    Hack for inner model layers to pass some monitoring info to the trainer.
    """

    def __init__(self):
        self.log_dict = {}
        self.step = 0
        self._attn_sample_layers: set = set()
        self._should_collect_attn: bool = False
        # Registry-controlled flags (set by MetricsCollector.on_forward_start)
        self._should_log_x_out: bool = False
        self._should_log_maxlogits: bool = False

    def reset(self):
        self.log_dict = {}

    def set_step(self, step: int):
        self.step = step

    def should_collect_attn(self, layer_idx: int) -> bool:
        """Whether the current step should collect attention stats for this layer."""
        return self._should_collect_attn and layer_idx in self._attn_sample_layers

    def configure_attn_sampling(self, num_layers: int, max_samples: int = 5):
        """Automatically choose representative layers for attention-stat sampling.

        Uniformly select max_samples layers, including first and last.
        """
        if num_layers <= 0:
            return
        if num_layers <= max_samples:
            self._attn_sample_layers = set(range(num_layers))
        else:
            indices = set()
            for i in range(max_samples):
                indices.add(round(i * (num_layers - 1) / (max_samples - 1)))
            self._attn_sample_layers = indices

    def log(self, update_dict):
        self.log_dict.update(update_dict)

    def get_metrics(self):
        return self.log_dict


def set_global_monitor():
    global _GLOBAL_MONITOR
    _GLOBAL_MONITOR = GlobalMonitor()


def get_global_monitor() -> GlobalMonitor:
    return _GLOBAL_MONITOR


def init_experiment_tracker(cfg: DictConfig, accelerator: Accelerator, output_dir: Path):
    """
    Initialize experiment tracker (SwanLab or WandB) using Accelerator's unified API.

    Args:
        cfg: Hydra configuration
        accelerator: Accelerator instance
        output_dir: Output directory for logs

    Returns:
        tracker_type: Type of tracker initialized ('swanlab', 'wandb', or 'none')
    """
    tracker_type = cfg.logger.type.lower()

    if tracker_type == "none":
        logger.info("Logger disabled (type=none)")
        return tracker_type

    # Set project and experiment name from task if not specified
    task_name = cfg.logger.task if cfg.logger.task else cfg.hydra.runtime.choices.task
    project_name = cfg.logger.project if cfg.logger.project else task_name.split("/")[0]
    experiment_name = (
        cfg.logger.experiment_name if cfg.logger.experiment_name else task_name.split("/")[-1]
    )
    dir = (
        cfg.logger.dir
        if cfg.logger.dir
        else str(output_dir / "swanlab")
        if tracker_type == "swanlab"
        else str(output_dir / "wandb")
    )

    init_kwargs = {}

    if tracker_type == "swanlab":
        init_kwargs["swanlab"] = {
            "workspace": cfg.logger.workspace,
            "experiment_name": experiment_name,
            "logdir": dir,
            "mode": cfg.logger.mode,
        }
    elif tracker_type == "wandb":
        init_kwargs["wandb"] = {
            "name": experiment_name,
            "dir": dir,
            "mode": cfg.logger.mode,
        }
        # For wandb, workspace field is entity
        if cfg.logger.workspace:
            init_kwargs["wandb"]["entity"] = cfg.logger.workspace
    elif tracker_type is None:
        logger.info("Logger disabled (type=none)")
        return tracker_type
    else:
        raise ValueError(
            f"Unsupported logger type: {tracker_type}. Choose 'swanlab', 'wandb', or 'none'."
        )

    accelerator.init_trackers(
        project_name=project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs=init_kwargs,
    )
    logger.info(f"Initialized {tracker_type} tracker")

    # Save git diff of Python files for code tracing
    if tracker_type == "wandb" and accelerator.is_main_process:
        import subprocess
        import wandb

        try:
            diff = subprocess.check_output(
                ["git", "diff", "--", "*.py", "*/*.py", "*/*/*.py", "*/*/*/*.py"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")
            if diff:
                wandb.run.log_code(name="git_diff.patch", include_fn=lambda _: False)
                with open(f"{dir}/git_diff.patch", "w") as f:
                    f.write(diff)
                wandb.save(f"{dir}/git_diff.patch", base_path=dir, policy="now")
                logger.info(f"Saved git diff ({len(diff)} bytes) to wandb")
            else:
                logger.info("No uncommitted Python changes to save")
        except Exception as e:
            logger.warning(f"Failed to save git diff: {e}")

    return tracker_type


class MFUTracker:
    """
    Model FLOPS Utilization (MFU) Tracker

    Tracks and calculates the hardware utilization during training by comparing
    actual FLOPS achieved vs theoretical peak FLOPS of the GPU.
    """

    def __init__(
        self,
        model,
        batch_size,
        device_id=0,
        update_interval=10,
        world_size=1,
        dtype=None,
    ):
        """
        Initialize MFU tracker.

        Args:
            model: The model to track
            batch_size: Effective batch size (batch_size * grad_accumulation * world_size)
            device_id: GPU device ID
            seq_length: Sequence length (optional, for sequence models)
            world_size: Number of GPUs (default: 1)
            dtype: Training dtype for MFU calculation (torch.float32, torch.bfloat16, torch.float16)

        Important Notes on dtype:
            - For pure FP32/BF16 training: Pass the model's dtype
            - For AMP training: Pass the autocast dtype (e.g., torch.bfloat16), NOT the weight dtype
              * In AMP, weights are stored in FP32, but compute uses lower precision
              * ~95%+ of FLOPs come from matmul/conv which use the autocast precision
              * Example: with torch.autocast("cuda", dtype=torch.bfloat16), pass torch.bfloat16
        """
        self.device_id = device_id
        self.batch_size = batch_size
        self.world_size = world_size

        # Auto-detect dtype from model if not specified
        if dtype is None:
            dtype = next(model.parameters()).dtype
            logger.warning(
                f"Auto-detected training dtype from model weights: {dtype}. "
                f"For AMP training, please explicitly pass amp_dtype for accurate MFU calculation."
            )
        self.dtype = dtype

        # Get GPU peak FLOPS (single GPU) for the specified dtype
        # Note: For AMP training, this should be the autocast dtype (e.g., bf16),
        # not the model weight dtype, since compute-intensive ops use the autocast dtype
        self.gpu_peak_flops = self._get_gpu_peak_flops(dtype)

        # Total peak FLOPS across all GPUs
        self.total_peak_flops = self.gpu_peak_flops * world_size

        # Estimate model FLOPS per step
        self.model_flops_per_step = self._estimate_model_flops(model)

        # Tracking variables
        self.start_time = time.time()
        self.start_step = 0
        self.update_interval = (
            update_interval  # Update MFU metrics every N steps for recent performance
        )

        # Detect if this is likely AMP training (weights FP32 but compute dtype is lower precision)
        weight_dtype = next(model.parameters()).dtype
        is_amp_training = weight_dtype == torch.float32 and dtype in [torch.bfloat16, torch.float16]
        training_mode = f"AMP ({dtype})" if is_amp_training else f"{dtype}"

        # Store for Training Configuration box (printed by finetune.py before training starts)
        self._training_mode = training_mode
        self._world_size = world_size

    def _get_gpu_peak_flops(self, dtype):
        """
        Estimate peak FLOPS for the GPU based on dtype.

        Args:
            dtype: torch.float32, torch.bfloat16, or torch.float16
        """
        device_name = torch.cuda.get_device_name(self.device_id)
        device_capability = torch.cuda.get_device_capability(self.device_id)

        # Peak FLOPS estimates for common GPUs
        # Format: {GPU_name: {'bf16': TFLOPS, 'fp16': TFLOPS, 'fp32': TFLOPS}}
        gpu_peak_flops_db = {
            "B200": {"bf16": 2250e12, "fp16": 2250e12, "fp32": 75e12, "tf32": 1125e12},
            "B20Z": {"bf16": 2250e12, "fp16": 2250e12, "fp32": 75e12, "tf32": 1125e12},
            "B100": {"bf16": 1750e12, "fp16": 1750e12, "fp32": 60e12, "tf32": 875e12},
            "H100": {"bf16": 1979e12, "fp16": 1979e12, "fp32": 67e12, "tf32": 989e12},
            "H20": {"bf16": 148e12, "fp16": 148e12, "fp32": 44e12, "tf32": 74e12},
            "A100": {"bf16": 624e12, "fp16": 624e12, "fp32": 19.5e12, "tf32": 312e12},
            "A800": {"bf16": 624e12, "fp16": 624e12, "fp32": 19.5e12, "tf32": 312e12},
            "4090": {
                "bf16": 165.2e12,
                "fp16": 165.2e12,
                "fp32": 82.6e12,
                "tf32": 82.6e12,
            },  # RTX 4090
        }

        # Determine precision type
        if dtype == torch.bfloat16:
            dtype_key = "bf16"
        elif dtype == torch.float16:
            dtype_key = "fp16"
        elif dtype == torch.float32:
            dtype_key = "fp32"
        else:
            logger.warning(f"Unknown dtype {dtype}, defaulting to fp32")
            dtype_key = "fp32"

        # Try to match GPU name
        for key, flops_dict in gpu_peak_flops_db.items():
            if key in device_name:
                peak_flops = flops_dict.get(dtype_key)
                if peak_flops is None:
                    # Fallback to bf16 value (e.g. H20 has no fp16 spec)
                    peak_flops = flops_dict.get("bf16", 0)
                    logger.warning(
                        f"GPU {device_name} has no {dtype} peak FLOPS spec, "
                        f"falling back to bf16: {peak_flops / 1e12:.1f} TFLOPS"
                    )
                else:
                    logger.info(
                        f"Detected GPU: {device_name}, dtype: {dtype}, "
                        f"peak FLOPS: {peak_flops / 1e12:.1f} TFLOPS"
                    )
                self._gpu_name = device_name
                return peak_flops

        # Default estimate based on compute capability
        if device_capability[0] >= 10:  # Blackwell (B200, B100)
            default_flops = {"bf16": 2000e12, "fp16": 2000e12, "fp32": 70e12}
        elif device_capability[0] >= 9:  # Hopper (H100, H800)
            default_flops = {"bf16": 1000e12, "fp16": 1000e12, "fp32": 67e12}
        elif device_capability[0] >= 8:  # Ampere (A100, A30, RTX 30xx/40xx)
            default_flops = {"bf16": 150e12, "fp16": 150e12, "fp32": 20e12}
        elif device_capability[0] >= 7:  # Volta/Turing (V100, T4)
            default_flops = {"bf16": 50e12, "fp16": 100e12, "fp32": 15e12}
        else:
            default_flops = {"bf16": 25e12, "fp16": 50e12, "fp32": 10e12}

        peak_flops = default_flops[dtype_key]
        self._gpu_name = f"{device_name} (estimated)"
        return peak_flops

    def _estimate_model_flops(self, model):
        """Estimate FLOPs per training step (forward + backward).

        Delegates to model.estimate_training_flops_per_sample() if available
        (each Policy subclass implements its own FLOPs estimation).
        Returns 0 if the model doesn't support FLOPs estimation.
        """
        flops_per_sample = 0
        if hasattr(model, "estimate_training_flops_per_sample"):
            flops_per_sample = model.estimate_training_flops_per_sample()

        if flops_per_sample == 0:
            self._num_params_M = 0.0
            self._flops_per_step_T = 0.0
            return 0

        num_params = sum(p.numel() for p in model.parameters())
        flops_per_step = flops_per_sample * self.batch_size
        self._num_params_M = num_params / 1e6
        self._flops_per_step_T = flops_per_step / 1e12
        return flops_per_step

    def reset(self, current_step):
        """Reset the timer for tracking recent performance."""
        self.start_time = time.time()
        self.start_step = current_step

    def compute_metrics(self, current_step):
        """
        Compute MFU and throughput metrics.

        Returns:
            dict: Dictionary containing mfu, samples_per_sec, steps_per_sec
        """
        elapsed_time = time.time() - self.start_time
        steps_completed = current_step - self.start_step

        if elapsed_time > 0 and steps_completed > 0:
            # Actual FLOPS = (FLOPs per step * steps) / time
            actual_flops = (self.model_flops_per_step * steps_completed) / elapsed_time
            # Use total_peak_flops since model_flops_per_step accounts for all GPUs
            mfu = actual_flops / self.total_peak_flops

            # Throughput metrics
            samples_per_sec = (self.batch_size * steps_completed) / elapsed_time
            steps_per_sec = steps_completed / elapsed_time
        else:
            mfu = 0.0
            samples_per_sec = 0.0
            steps_per_sec = 0.0

        # Reset timer periodically for recent performance tracking
        if steps_completed >= self.update_interval:
            self.reset(current_step)

        return {
            "performance/mfu": mfu,
            "performance/samples_per_sec": samples_per_sec,
            "performance/steps_per_sec": steps_per_sec,
        }

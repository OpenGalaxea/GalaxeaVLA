import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


def move_to_device(data, device):
    """Recursively move all tensors in a nested structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data


def _get_sample_processor(processor, embodiment):
    if processor is None:
        return None
    if hasattr(processor, "processors"):
        if embodiment is None:
            return None
        return processor[embodiment]
    return processor


def _extract_parts_meta_from_batch(batch):
    if not isinstance(batch, dict):
        return None

    action_parts_meta = batch.get("action_parts_meta")
    if not isinstance(action_parts_meta, dict):
        return None

    parts_meta = {}
    for key, value in action_parts_meta.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            parts_meta[key] = int(value.reshape(-1)[0].item())
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            parts_meta[key] = int(value[0])
        else:
            parts_meta[key] = int(value)

    return parts_meta if parts_meta else None


def resolve_parts_meta(processor=None, batch=None):
    """Resolve action parts meta from explicit processor or batch payload."""
    if processor is not None:
        if hasattr(processor, "processors") and isinstance(processor.processors, dict):
            for sub_processor in processor.processors.values():
                merger = getattr(sub_processor, "action_state_merger", None)
                parts_meta = getattr(merger, "max_action_shape_meta", None)
                if isinstance(parts_meta, dict) and parts_meta:
                    return dict(parts_meta)
        merger = getattr(processor, "action_state_merger", None)
        parts_meta = getattr(merger, "max_action_shape_meta", None)
        if isinstance(parts_meta, dict) and parts_meta:
            return dict(parts_meta)

    return _extract_parts_meta_from_batch(batch)


def _denormalize_action_batch(action_batch, proprio_batch, processor=None, embodiments=None):
    if processor is None:
        return action_batch
    if not isinstance(action_batch, torch.Tensor) or not isinstance(proprio_batch, torch.Tensor):
        return action_batch

    denorm_actions = []
    batch_size = action_batch.shape[0]
    for idx in range(batch_size):
        embodiment = embodiments[idx] if embodiments is not None else None
        sample_processor = _get_sample_processor(processor, embodiment)
        if sample_processor is None:
            denorm_actions.append(action_batch[idx : idx + 1])
            continue

        sample = {
            "action": action_batch[idx : idx + 1].clone(),
            "state": proprio_batch[idx : idx + 1].clone(),
        }
        sample = sample_processor.action_state_merger.backward(sample)

        # action_state_merger.backward may return tensors with a leading batch
        # dimension (shape [1, T, D]). For per-sample re-forward, merger.forward
        # expects unbatched tensors [T, D].
        for field in ("action", "state"):
            if field in sample and isinstance(sample[field], dict):
                squeezed = {}
                for key, value in sample[field].items():
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1:
                        squeezed[key] = value.squeeze(0)
                    else:
                        squeezed[key] = value
                sample[field] = squeezed

        sample = sample_processor.normalizer.backward(sample)
        sample = sample_processor.action_state_merger.forward(sample)

        action_denorm = sample["action"]
        if isinstance(action_denorm, torch.Tensor) and action_denorm.ndim == 2:
            action_denorm = action_denorm.unsqueeze(0)
        denorm_actions.append(action_denorm)

    return torch.cat(denorm_actions, dim=0)


def _denormalize_tokenizer_action_batch(
    action_batch, proprio_batch, processor=None, embodiments=None
):
    """Denormalize actions from tokenizer decode space to physical space.

    When vlm_input_action_normalizer is configured (e.g., dummy clip for KI training),
    the tokenizer decode output is in that space, NOT the main normalizer space.
    Falls back to the main normalizer if vlm_input_action_normalizer is not available
    (i.e., tokenizer and main normalizer share the same space).
    """
    if processor is None:
        return action_batch
    if not isinstance(action_batch, torch.Tensor) or not isinstance(proprio_batch, torch.Tensor):
        return action_batch

    denorm_actions = []
    batch_size = action_batch.shape[0]
    for idx in range(batch_size):
        embodiment = embodiments[idx] if embodiments is not None else None
        sample_processor = _get_sample_processor(processor, embodiment)
        if sample_processor is None:
            denorm_actions.append(action_batch[idx : idx + 1])
            continue

        sample = {
            "action": action_batch[idx : idx + 1].clone(),
            "state": proprio_batch[idx : idx + 1].clone(),
        }
        sample = sample_processor.action_state_merger.backward(sample)

        for field in ("action", "state"):
            if field in sample and isinstance(sample[field], dict):
                squeezed = {}
                for key, value in sample[field].items():
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1:
                        squeezed[key] = value.squeeze(0)
                    else:
                        squeezed[key] = value
                sample[field] = squeezed

        # Use vlm_input_action_normalizer if available (tokenizer space),
        # otherwise fall back to main normalizer (same space as tokenizer)
        vlm_norm = getattr(sample_processor, "_vlm_input_action_normalizer", None)
        if vlm_norm is not None:
            sample = vlm_norm.backward(sample)
        else:
            sample = sample_processor.normalizer.backward(sample)

        sample = sample_processor.action_state_merger.forward(sample)

        action_denorm = sample["action"]
        if isinstance(action_denorm, torch.Tensor) and action_denorm.ndim == 2:
            action_denorm = action_denorm.unsqueeze(0)
        denorm_actions.append(action_denorm)

    return torch.cat(denorm_actions, dim=0)


def _masked_l1(pred, gt, valid_dim_mask):
    num_valid = valid_dim_mask.sum().clamp(min=1)
    diff = torch.abs(pred - gt)
    return diff.masked_select(valid_dim_mask).sum() / num_valid.float()


def build_valid_dim_mask(action_dim_is_pad, target):
    """Build a boolean mask that marks valid (non-padded) action dimensions.

    Args:
        action_dim_is_pad: (B, D) bool tensor, True = padded dim.
        target: (B, T, D) tensor whose shape the mask will be expanded to.

    Returns:
        valid_dim_mask: (B, T, D) bool tensor, True = valid dim.
    """
    return (~action_dim_is_pad.bool()).unsqueeze(1).expand_as(target)


def compute_fm_metrics(action_pred, action_gt, valid_dim_mask, threshold=1 / 256):
    """Compute FM action accuracy and L1 in normalized space.

    Args:
        action_pred: (B, T, D) predicted actions.
        action_gt: (B, T, D) ground-truth actions.
        valid_dim_mask: (B, T, D) bool mask, True = valid dim.
        threshold: accuracy threshold (default 1/256).

    Returns:
        (fm_acc, fm_l1) as Python floats.
    """
    diff = torch.abs(action_pred - action_gt)
    num_valid = valid_dim_mask.sum().clamp(min=1)
    fm_acc = ((diff < threshold) & valid_dim_mask).sum().float() / num_valid.float()
    fm_l1 = diff.masked_select(valid_dim_mask).sum() / num_valid.float()
    return fm_acc.item(), fm_l1.item()


def compute_per_key_l1(action_pred, action_gt, valid_dim_mask, parts_meta):
    """Compute denormalized L1 for each action part key defined by parts_meta.

    Args:
        action_pred: (B, T, D) denormalized predicted actions.
        action_gt: (B, T, D) denormalized ground truth actions.
        valid_dim_mask: (B, T, D) bool mask for valid dimensions.
        parts_meta: Ordered mapping {part_name: dim_count}.

    Returns:
        Dict[str, float]: {"per_key/<part>/l1": value}
    """
    if not isinstance(parts_meta, dict) or len(parts_meta) == 0:
        return {}

    result = {}
    offset = 0
    total_dim = action_pred.shape[-1]

    for part, dim in parts_meta.items():
        dim = int(dim)
        if dim <= 0:
            continue
        if offset + dim > total_dim:
            break

        pred_part = action_pred[..., offset : offset + dim]
        gt_part = action_gt[..., offset : offset + dim]
        mask_part = valid_dim_mask[..., offset : offset + dim]

        if not mask_part.any():
            offset += dim
            continue

        diff = torch.abs(pred_part - gt_part)
        num_valid = mask_part.sum().clamp(min=1).float()
        l1 = diff.masked_select(mask_part).sum() / num_valid
        result[f"per_key/{part}/l1"] = l1.item()

        offset += dim

    return result


def compute_per_key_binary_accuracy(
    action_pred, action_gt, valid_dim_mask, parts_meta, rule_key_names, threshold=0.5
):
    """Compute binary accuracy for rule-based keys in denormalized space.

    Binarize both pred and gt at threshold, then compute agreement fraction.

    Args:
        action_pred: (B, T, D) denormalized predicted actions.
        action_gt: (B, T, D) denormalized ground truth actions.
        valid_dim_mask: (B, T, D) bool mask for valid dimensions.
        parts_meta: Ordered mapping {part_name: dim_count}.
        rule_key_names: Set or list of key names that are rule-based.
        threshold: Value above which the binary state is considered "1".

    Returns:
        Dict[str, float]: {"per_key/<part>/binary_acc": value}
    """
    if not isinstance(parts_meta, dict) or len(parts_meta) == 0:
        return {}
    if not rule_key_names:
        return {}

    rule_set = set(rule_key_names) if not isinstance(rule_key_names, set) else rule_key_names
    result = {}
    offset = 0
    total_dim = action_pred.shape[-1]

    for part, dim in parts_meta.items():
        dim = int(dim)
        if dim <= 0 or part not in rule_set:
            offset += dim
            continue
        if offset + dim > total_dim:
            break

        pred_part = action_pred[..., offset : offset + dim]
        gt_part = action_gt[..., offset : offset + dim]
        mask_part = valid_dim_mask[..., offset : offset + dim]

        if not mask_part.any():
            offset += dim
            continue

        pred_binary = (pred_part > threshold).float()
        gt_binary = (gt_part > threshold).float()
        correct = ((pred_binary - gt_binary).abs() < 0.5) & mask_part
        num_valid = mask_part.sum().clamp(min=1).float()
        acc = correct.sum().float() / num_valid
        result[f"per_key/{part}/binary_acc"] = acc.item()

        offset += dim

    return result


def _compute_per_emb_raw(
    embodiments,
    action_preds_denorm,
    action_gt_denorm,
    action_preds_cpu,
    action_gt,
    valid_dim_mask,
    ar_pred=None,
    ar_gt=None,
    ar_valid_dim_mask=None,
    threshold=1 / 256,
):
    """Compute per-embodiment raw sums for distributed aggregation.

    Returns:
        dict: {emb_name: {"fm_l1_sum", "fm_acc_correct", "fm_valid_count",
                          "ar_l1_sum", "ar_valid_count", "sample_count"}}
    """
    emb_groups = defaultdict(list)
    for i, emb in enumerate(embodiments):
        emb_groups[emb].append(i)

    per_emb_raw = {}
    for emb, indices in emb_groups.items():
        emb_mask = valid_dim_mask[indices]
        fm_valid = emb_mask.sum().item()

        entry = {
            "sample_count": len(indices),
            "fm_valid_count": fm_valid,
        }

        # FM L1 (denormalized)
        fm_diff = torch.abs(action_preds_denorm[indices] - action_gt_denorm[indices])
        entry["fm_l1_sum"] = fm_diff.masked_select(emb_mask).sum().item()

        # FM accuracy (normalized space)
        fm_diff_norm = torch.abs(action_preds_cpu[indices] - action_gt[indices])
        entry["fm_acc_correct"] = ((fm_diff_norm < threshold) & emb_mask).sum().float().item()

        # AR L1 (denormalized, optional)
        if ar_pred is not None and ar_gt is not None:
            ar_emb_mask = ar_valid_dim_mask[indices] if ar_valid_dim_mask is not None else emb_mask
            ar_diff = torch.abs(ar_pred[indices] - ar_gt[indices])
            entry["ar_l1_sum"] = ar_diff.masked_select(ar_emb_mask).sum().item()
            entry["ar_valid_count"] = ar_emb_mask.sum().item()

        per_emb_raw[emb] = entry

    return per_emb_raw


def reduce_per_emb_metrics(per_emb_raw):
    """All-gather per-emb raw sums across DDP ranks and compute weighted averages.

    Args:
        per_emb_raw: dict from _compute_per_emb_raw()

    Returns:
        dict: {"per_emb/{emb}/eval/fm_action_l1": float, ...}
    """
    import torch.distributed as dist

    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, per_emb_raw)
    else:
        gathered = [per_emb_raw]

    # Merge sums across ranks
    merged = defaultdict(lambda: defaultdict(float))
    for rank_data in gathered:
        for emb, entry in rank_data.items():
            for k, v in entry.items():
                merged[emb][k] += v

    # Compute weighted averages
    metrics = {}
    for emb, m in merged.items():
        prefix = f"per_emb/{emb}"
        fm_valid = m.get("fm_valid_count", 0)
        if fm_valid > 0:
            metrics[f"{prefix}/eval/fm_action_l1"] = m["fm_l1_sum"] / fm_valid
            metrics[f"{prefix}/eval/fm_action_acc"] = m["fm_acc_correct"] / fm_valid
        ar_valid = m.get("ar_valid_count", 0)
        if ar_valid > 0:
            metrics[f"{prefix}/eval/ar_action_l1"] = m["ar_l1_sum"] / ar_valid
        metrics[f"{prefix}/eval/sample_count"] = int(m["sample_count"])

    return metrics


def reduce_payload(payload):
    """All-gather a flat {key: float} payload across DDP ranks and aggregate.

    Counts (``*sample_count`` / ``meta/num_samples``) are summed, everything
    else is averaged. Uses all_gather_object so ranks with different key sets
    cannot deadlock.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return payload

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)

    key_values = defaultdict(list)
    for rank_payload in gathered:
        for key, value in rank_payload.items():
            key_values[key].append(float(value))

    reduced = {}
    for key, values in key_values.items():
        if key.endswith("sample_count") or key == "meta/num_samples":
            reduced[key] = sum(values)
        else:
            reduced[key] = sum(values) / len(values)
    return reduced


def rollout_and_calculate_metrics(
    batch,
    model=None,
    accelerator=None,
    processor=None,
    overfit=False,
    return_per_emb_raw=False,
    parts_meta=None,
    return_preds=False,
    embodiments_override=None,
):
    """Rollout + teacher-forcing forward and collect metrics.

    Args:
        embodiments_override: when provided, per-embodiment metrics use this list
            as the key. A typical case is eval_checkpoints using
            ``batch["embodiment_fine"]`` to split fine-grained embodiments, while
            the processor side (denormalization) still uses ``batch["embodiment"]``
            as the coarse-grained type. When omitted, both paths match training
            and use ``batch["embodiment"]``.
    """
    log_dict = {}
    recent_metrics = {}

    if parts_meta is None:
        parts_meta = resolve_parts_meta(processor=processor, batch=batch)

    # normalized action (ground truth)
    action_gt = batch["action"].cpu()
    proprio_gt = batch.get("proprio", None)
    if isinstance(proprio_gt, torch.Tensor):
        proprio_gt = proprio_gt.cpu()

    # embodiments_for_processor always uses batch["embodiment"], which must be the
    # coarse-grained name known by the processor.
    # embodiments_for_per_emb uses the override or falls back to the processor path.
    embodiments_for_processor = None
    if isinstance(batch.get("embodiment", None), list):
        embodiments_for_processor = batch["embodiment"]
    if embodiments_override is not None:
        embodiments_for_per_emb = list(embodiments_override)
    else:
        embodiments_for_per_emb = embodiments_for_processor
    # Keep the legacy variable name for _denormalize_*_batch; it must be coarse-grained.
    embodiments = embodiments_for_processor

    # Move all tensors in batch to the correct device
    batch = move_to_device(batch, accelerator.device)

    if overfit:
        action_hash = (
            batch["action"].sum().item() if isinstance(batch.get("action"), torch.Tensor) else "N/A"
        )
        logger.info(
            f"[OVERFIT DEBUG] eval batch action_sum={action_hash:.6f}, shape={batch['action'].shape}"
        )

    with torch.no_grad(), accelerator.autocast():
        was_training = model.training
        if overfit:
            model.eval()

        # --- Teacher-forcing forward pass FIRST ---
        # Run before inference to avoid any inference side-effects on model state.
        loss, loss_value_dict = model(batch)
        teacher_forcing_acc = model.train_action_accuracy
        teacher_forcing_cot_acc = getattr(model, "train_cot_accuracy", 0.0)
        if "ce_loss" in loss_value_dict:
            teacher_forcing_ce_loss = loss_value_dict["ce_loss"].cpu().item()
        else:
            teacher_forcing_ce_loss = -1.0

        if "fm_loss" in loss_value_dict:
            fm_loss = loss_value_dict["fm_loss"].cpu().item()
        else:
            fm_loss = -1.0

        # --- Inference rollout SECOND ---
        # Use a shallow copy so that inference's batch.update(generated) does NOT
        # overwrite the original batch["action"] (which is the GT action needed
        # by metric computation below).
        inference_batch = {**batch}
        action_preds = model(
            inference_batch,
            inference_mode=True,
        )

        if overfit:
            model.train(was_training)

    recent_metrics["teacher_forcing/ar_action_acc"] = teacher_forcing_acc
    recent_metrics["teacher_forcing/ar_cot_acc"] = teacher_forcing_cot_acc
    recent_metrics["teacher_forcing/ar_action_ce_loss"] = teacher_forcing_ce_loss
    recent_metrics["teacher_forcing/fm_loss"] = fm_loss

    if isinstance(action_preds, dict):
        ar_action_token_acc = action_preds.get("action_accuracy", -1.0)
        ar_action_l1_model = action_preds.get("action_l1_loss", -1.0)
        ar_cot_acc_model = action_preds.get("cot_accuracy", -1.0)
        recent_metrics["rollout/ar_action_token_acc"] = ar_action_token_acc
        recent_metrics["rollout/ar_cot_acc"] = ar_cot_acc_model
        recent_metrics["rollout/ar_action_l1_normalized"] = ar_action_l1_model

        # CoT text available in action_preds["cot_text"] for caller to log separately
        # (not put in recent_metrics because reduce_payload requires numeric values)

        action_dim_is_pad = batch["action_dim_is_pad"].cpu().bool()  # (B, D_action), True = padded
        valid_dim_mask = build_valid_dim_mask(action_dim_is_pad, action_gt)

        noop_dim_mask = torch.zeros_like(valid_dim_mask)

        action_op_mask_cpu = batch.get("action_op_mask")
        if isinstance(action_op_mask_cpu, torch.Tensor):
            action_op_mask_cpu = action_op_mask_cpu.cpu().bool()
        if action_op_mask_cpu is not None:
            # Backward-compat: stale cached batches may still carry the old (B, 1, D) shape.
            if action_op_mask_cpu.dim() == 3 and action_op_mask_cpu.shape[1] == 1:
                action_op_mask_cpu = action_op_mask_cpu.squeeze(1)
            inactive_dims = ~action_op_mask_cpu & ~action_dim_is_pad
            noop_dim_mask |= inactive_dims.unsqueeze(1).expand_as(valid_dim_mask)

        ar_absent_keys = (
            action_preds.get("ar_absent_keys") if isinstance(action_preds, dict) else None
        )
        if ar_absent_keys is not None and parts_meta is not None:
            key_order = list(parts_meta.keys())
            dim_counts = [int(d) for d in parts_meta.values()]
            offsets = [0]
            for d in dim_counts[:-1]:
                offsets.append(offsets[-1] + d)
            for b, absent in enumerate(ar_absent_keys):
                if b >= noop_dim_mask.shape[0]:
                    break
                for part, dim_count, off in zip(key_order, dim_counts, offsets):
                    if part in absent:
                        noop_dim_mask[b, :, off : off + dim_count] = True

        ar_valid_dim_mask = valid_dim_mask & ~noop_dim_mask

        # Denormalize GT once — shared by both AR and FM metric computation
        # and downstream snapshot visualization.
        action_gt_denorm = action_gt
        if processor is not None and isinstance(proprio_gt, torch.Tensor):
            action_gt_denorm = _denormalize_action_batch(
                action_batch=action_gt,
                proprio_batch=proprio_gt,
                processor=processor,
                embodiments=embodiments,
            )

        # AR action from model output (must be ar_action; no fallback)
        ar_action_preds_cpu = None
        if isinstance(action_preds.get("ar_action"), torch.Tensor):
            ar_action_preds_cpu = action_preds["ar_action"].cpu()

        # AR accuracy & L1 — compare in denormalized (physical) space.
        # AR predictions are in tokenizer space (vlm_input_action normalized),
        # GT is in main normalized space. Denormalize each with the correct
        # normalizer so both end up in physical space for fair comparison.
        ar_pred = None
        ar_action_l1_denorm = -1.0
        ar_action_acc = -1.0
        if (
            isinstance(ar_action_preds_cpu, torch.Tensor)
            and ar_action_preds_cpu.shape == action_gt.shape
        ):
            ar_pred = ar_action_preds_cpu
            if processor is not None and isinstance(proprio_gt, torch.Tensor):
                ar_pred = _denormalize_tokenizer_action_batch(
                    action_batch=ar_action_preds_cpu,
                    proprio_batch=proprio_gt,
                    processor=processor,
                    embodiments=embodiments,
                )
            ar_action_acc, _ = compute_fm_metrics(ar_pred, action_gt_denorm, ar_valid_dim_mask)
            ar_action_l1_denorm = _masked_l1(ar_pred, action_gt_denorm, ar_valid_dim_mask).item()
        recent_metrics["rollout/ar_action_acc"] = ar_action_acc
        recent_metrics["rollout/ar_action_l1"] = ar_action_l1_denorm

        # FM accuracy & normalized L1
        action_preds_cpu = action_preds["action"].cpu()  # (B, T, D_action)
        fm_acc, _ = compute_fm_metrics(action_preds_cpu, action_gt, valid_dim_mask)
        recent_metrics["rollout/fm_action_acc"] = fm_acc

        action_preds_denorm = action_preds_cpu
        if processor is not None and isinstance(proprio_gt, torch.Tensor):
            action_preds_denorm = _denormalize_action_batch(
                action_batch=action_preds_cpu,
                proprio_batch=proprio_gt,
                processor=processor,
                embodiments=embodiments,
            )

        action_l1_loss = _masked_l1(action_preds_denorm, action_gt_denorm, valid_dim_mask)
        recent_metrics["rollout/fm_action_l1"] = action_l1_loss.item()

        if parts_meta is not None:
            per_key_fm = compute_per_key_l1(
                action_pred=action_preds_denorm,
                action_gt=action_gt_denorm,
                valid_dim_mask=valid_dim_mask,
                parts_meta=parts_meta,
            )
            for key, value in per_key_fm.items():
                recent_metrics[f"rollout/{key}/fm_action_l1"] = value

            if isinstance(ar_pred, torch.Tensor):
                per_key_ar = compute_per_key_l1(
                    action_pred=ar_pred,
                    action_gt=action_gt_denorm,
                    valid_dim_mask=ar_valid_dim_mask,
                    parts_meta=parts_meta,
                )
                for key, value in per_key_ar.items():
                    recent_metrics[f"rollout/{key}/ar_action_l1"] = value

                rule_key_names = set()
                tokenizer_obj = getattr(processor, "action_tokenizer", None)
                if tokenizer_obj is not None:
                    decode_meta = getattr(tokenizer_obj, "_decode_meta", None)
                    if decode_meta is not None:
                        rule_key_names = set(getattr(decode_meta, "rule_key_names", []))
                per_key_binary_acc = compute_per_key_binary_accuracy(
                    action_pred=ar_pred,
                    action_gt=action_gt_denorm,
                    valid_dim_mask=ar_valid_dim_mask,
                    parts_meta=parts_meta,
                    rule_key_names=rule_key_names,
                )
                for key, value in per_key_binary_acc.items():
                    recent_metrics[f"rollout/{key}/ar_action_binary_acc"] = value

        if parts_meta is not None:
            _parts_key_order = list(parts_meta.keys())
            _parts_dim_counts = [int(d) for d in parts_meta.values()]
            _parts_offsets = [0]
            for d in _parts_dim_counts[:-1]:
                _parts_offsets.append(_parts_offsets[-1] + d)
            _total_parts = len(_parts_key_order)
            _B = action_dim_is_pad.shape[0]

            _pred_ratio_sum = 0.0
            _pred_ratio_count = 0
            for b in range(_B):
                _num_actual = 0
                _op_b = action_op_mask_cpu[b] if action_op_mask_cpu is not None else None
                _pad_b = action_dim_is_pad[b]
                _absent_b = (
                    ar_absent_keys[b]
                    if ar_absent_keys is not None and b < len(ar_absent_keys)
                    else set()
                )
                for _part, _dc, _off in zip(_parts_key_order, _parts_dim_counts, _parts_offsets):
                    if _off + _dc > _pad_b.shape[0]:
                        continue
                    _part_pad = _pad_b[_off : _off + _dc]
                    _is_padded = _part_pad.all()
                    _is_inactive = False
                    if _op_b is not None and not _is_padded:
                        _part_op = _op_b[_off : _off + _dc]
                        _is_inactive = not _part_op.any()
                    if not _is_padded and not _is_inactive:
                        _num_actual += 1
                _num_predicted = _total_parts - len(_absent_b & set(_parts_key_order))
                _pred_ratio_sum += _num_predicted / max(_num_actual, 1)
                _pred_ratio_count += 1

            if _pred_ratio_count > 0:
                recent_metrics["rollout/predicted_parts_ratio"] = (
                    _pred_ratio_sum / _pred_ratio_count
                )

    log_dict.update(
        {k: v.cpu().item() if isinstance(v, torch.Tensor) else v for k, v in recent_metrics.items()}
    )

    # Optionally attach raw pred tensors for downstream use (e.g. trajectory metrics)
    # to avoid redundant inference forward passes.
    preds_out = None
    if return_preds and isinstance(action_preds, dict):
        preds_out = {
            "action": action_preds["action"].cpu(),
            "action_gt": action_gt,
            "action_denorm": action_preds_denorm,
            "action_gt_denorm": action_gt_denorm,
            "valid_dim_mask": valid_dim_mask,
        }
        if isinstance(ar_pred, torch.Tensor):
            preds_out["ar_action_denorm"] = ar_pred
            preds_out["ar_gt_denorm"] = action_gt_denorm
            preds_out["ar_valid_dim_mask"] = ar_valid_dim_mask
            preds_out["ar_action_norm"] = ar_action_preds_cpu
            preds_out["ar_gt_norm"] = action_gt

    if not return_per_emb_raw:
        if return_preds:
            return log_dict, preds_out
        return log_dict

    # Compute per-embodiment raw sums for distributed aggregation.
    # Use embodiments_for_per_emb (override-aware) so eval can key by fine-grained emb
    # while denormalization above still uses the coarse processor-compatible name.
    per_emb_raw = {}
    if embodiments_for_per_emb is not None and isinstance(action_preds, dict):
        per_emb_raw = _compute_per_emb_raw(
            embodiments_for_per_emb,
            action_preds_denorm,
            action_gt_denorm,
            action_preds_cpu,
            action_gt,
            valid_dim_mask,
            ar_pred=ar_pred,
            ar_gt=action_gt_denorm if ar_pred is not None else None,
            ar_valid_dim_mask=ar_valid_dim_mask,
        )

    if return_preds:
        return log_dict, per_emb_raw, preds_out
    return log_dict, per_emb_raw

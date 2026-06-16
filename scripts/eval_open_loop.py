"""Open-loop evaluation / GT-only visualization.

With --no_model: loads dataset + processor, runs postprocess on GT actions and
states, then visualizes per-episode GT action curves and proprioceptive state.

With --client_mode: connects to a serve_policy.py WebSocket server for inference
instead of loading the model locally. Useful for end-to-end testing of the
serve_policy pipeline (preprocess → inference → postprocess).

Without --no_model (default): same as eval_open_loop.py — loads model, runs
inference, computes metrics, and visualizes both GT and predicted actions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

from collections import defaultdict
from pathlib import Path

import numpy as np
import rootutils
import torch

from functools import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root directory to the Python path
rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from g05.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset
from g05.utils.common.pytorch_utils import dict_apply, set_global_seed
from g05.utils.eval.visualize import visualize_episode
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.data_processor.processor.base_processor import BaseProcessor
from g05.utils.data.data_utils import collate_fn_pad_sequences
from g05.utils.logging.logging_config import setup_logging, get_logger
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.eval.eval_utils import (
    compute_valid_episode_indices,
    concat_dict_list,
    filter_embodiment,
    to_scalar,
    truncate_datasets,
)
from g05.utils.checkpoint.ckpt_utils import (
    find_run_dir,
    load_config_from_run_dir,
    load_config_from_task_yaml,
)

logger = get_logger(__name__)


def _extract_raw_obs(sample: dict, embodiment: str | None = None) -> dict:
    """Extract raw_obs from a dataset sample (without processor) for WebSocket client mode.

    Dataset __getitem__ (with processor=None) returns:
      images: {key: Tensor[obs_size, C, H, W]}  (uint8 after image_transforms)
      state:  {key: Tensor[obs_size, D]}         (float32)
      task:   str

    Server expects:
      images: {key: ndarray[C, H, W]}  (single frame, uint8)
      state:  {key: ndarray[D]}        (single frame)
      task:   str
    """
    raw_obs: dict = {
        "images": {},
        "state": {},
        "task": sample["task"],
    }
    # Take last observation step as current frame
    for key, val in sample["images"].items():
        arr = val[-1].numpy() if isinstance(val, torch.Tensor) else np.asarray(val[-1])
        raw_obs["images"][key] = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    for key, val in sample["state"].items():
        arr = val[-1].numpy() if isinstance(val, torch.Tensor) else np.asarray(val[-1])
        raw_obs["state"][key] = arr.astype(np.float32) if arr.dtype != np.float32 else arr

    if embodiment is not None:
        raw_obs["embodiment_type"] = embodiment
    if "frequency" in sample:
        raw_obs["frequency"] = float(sample["frequency"])
    if "coarse_task" in sample:
        raw_obs["coarse_task"] = sample["coarse_task"]
    return raw_obs


def evaluate_single_dataset(
    policy,  # model policy or None (--no_model mode)
    dataset: GalaxeaLerobotDataset,
    processor: BaseProcessor,
    output_dir: Path,
    cfg: DictConfig,
):
    """Run open-loop evaluation on a single-embodiment dataset.

    When policy is None (--no_model), only GT actions and states are visualized
    without any model inference or metric computation.

    Pipeline per batch:
      1. Save normalized GT (before predict_action overwrites batch["action"])
      2. Run model inference on GPU (skipped when policy is None)
      3. Compute AR/FM metrics in **normalized** space (skipped when policy is None)
      4. postprocess -> denormalize + split into per-body-part dicts
      5. Accumulate per-part dicts for visualization

    After the loop:
      - Concatenate all batches, slice by episode, and plot per-episode HTML
      - Save aggregated metrics to metrics.json (skipped when policy is None)
    """
    has_model = policy is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the same collate_fn as training: keeps batch['samples'] as list of dicts
    # (the model's internal processor expects per-sample template strings, not collated)
    action_collate_fn = partial(collate_fn_pad_sequences, padding_input_id=processor.pad_token_id)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batch_size_val,
        num_workers=cfg.model.num_workers,
        pin_memory=cfg.model.get("pin_memory", True),
        persistent_workers=cfg.model.persistent_workers,
        worker_init_fn=None,
        collate_fn=action_collate_fn,
    )

    # ---- Determine which episodes belong to the current split ----
    frame_offset, frame_end, valid_episode_indices = compute_valid_episode_indices(dataset)
    num_episodes = len(valid_episode_indices)
    all_ep_from = dataset.episode_data_index["from"]
    all_ep_to = dataset.episode_data_index["to"]
    logger.info(
        f"Split frame range: [{frame_offset}, {frame_end}), "
        f"valid episodes: {num_episodes} (out of {len(all_ep_from)} total)"
    )

    eval_episodes_num = (
        min(cfg.eval_episodes_num, num_episodes) if cfg.get("eval_episodes_num") else num_episodes
    )
    # eval_end_frame is relative to the dataloader (which starts from frame_offset)
    last_eval_ep = valid_episode_indices[eval_episodes_num - 1]
    eval_end_frame = int(all_ep_to[last_eval_ep]) - frame_offset

    num_eval_batches = (eval_end_frame + cfg.batch_size_val - 1) // cfg.batch_size_val
    logger.info(
        f"eval_end_frame={eval_end_frame}, num_eval_batches={num_eval_batches}, "
        f"total_dataloader_batches={len(dataloader)}"
    )

    # ==================== Inference loop ====================
    gt_actions = []  # list[dict{part: (B, chunk, dim)}]
    pd_actions = []  # list[dict{part: (B, chunk, dim)}]
    ar_actions = []  # list[dict{part: (B, chunk, dim)}], only if model produces ar_action
    gt_states = []  # list[dict{part: (B, obs, dim)}] — proprioceptive state
    op_masks = []  # list[dict{part: (B, dim)}] bool — action_op_mask per dim per sample
    has_op_mask = False
    has_ar_action = False
    all_metrics = defaultdict(list)

    # CoT video accumulation
    save_cot_video = cfg.get("save_cot_video", False) and has_model
    all_pixel_values = []  # list[(B, 1, C, H, W)] CPU
    all_pred_cot = []  # list[str], flat
    all_gt_cot = []  # list[str], flat

    # Normalized (pre-denorm) flat tensors for normalized-space visualization
    raw_gt_actions = []  # list[(B, T, D)]
    raw_pd_actions = []  # list[(B, T, D)]
    action_dim_is_pad_mask = None  # (D,) bool, captured from first batch

    _to_np = lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    desc = "inferencing" if has_model else "loading GT"

    for i, batch in tqdm(enumerate(dataloader), desc=desc, total=num_eval_batches):
        # predict_action overwrites batch["action"] with its prediction,
        # so we save the normalized GT first for metric computation.
        action_gt = batch["action"].clone()  # (B, T, D) normalized GT, on CPU
        proprio_gt = batch["proprio"].clone()  # (B, obs, D) normalized GT, on CPU

        if has_model:
            batch = dict_apply(batch, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                batch = policy.predict_action(batch)
            batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)

            # ---- Metrics in normalized space (before postprocess/denorm) ----
            # Metrics must be computed before postprocess because postprocess denormalizes
            # the actions and splits them into per-body-part dicts, losing the flat tensor.

            # AR token accuracy (produced by the model during inference; -1 if not available)
            all_metrics["rollout/ar_action_token_acc"].append(
                to_scalar(batch.get("action_accuracy", -1.0))
            )
            all_metrics["rollout/ar_action_l1"].append(to_scalar(batch.get("action_l1_loss", -1.0)))

            # Shared mask for AR and FM metrics
            action_dim_is_pad = batch["action_dim_is_pad"].bool()  # (B, D)
            valid_dim_mask = (~action_dim_is_pad).unsqueeze(1).expand_as(action_gt)
            num_valid = valid_dim_mask.sum().clamp(min=1)

            # AR action accuracy (normalized space, threshold-based, same as FM)
            ar_action_pred = batch.get("action_ar")
            if isinstance(ar_action_pred, torch.Tensor) and ar_action_pred.shape == action_gt.shape:
                ar_diff = torch.abs(ar_action_pred - action_gt)
                ar_acc = ((ar_diff < 1 / 256) & valid_dim_mask).sum().float() / num_valid.float()
                all_metrics["rollout/ar_action_acc"].append(ar_acc.item())
            else:
                all_metrics["rollout/ar_action_acc"].append(-1.0)

            # FM metrics (mask out padded action dims for mixture training)
            action_pred = batch["action"]  # (B, T, D)
            diff = torch.abs(action_pred - action_gt)
            fm_acc = ((diff < 1 / 256) & valid_dim_mask).sum().float() / num_valid.float()
            fm_l1 = diff.masked_select(valid_dim_mask).sum() / num_valid.float()

            all_metrics["rollout/fm_action_acc"].append(fm_acc.item())
            all_metrics["rollout/fm_action_l1"].append(fm_l1.item())

            # ---- Accumulate normalized flat tensors (before denorm) ----
            raw_gt_actions.append(action_gt.numpy())  # (B, T, D) normalized
            raw_pd_actions.append(batch["action"].cpu().numpy())  # (B, T, D) normalized
            if action_dim_is_pad_mask is None:
                action_dim_is_pad_mask = action_dim_is_pad[0].numpy()  # (D,) bool

        # ---- CoT video: accumulate pixel_values and text ----
        if save_cot_video:
            # Keep only the first camera view, move to CPU immediately
            all_pixel_values.append(batch["pixel_values"][:, 0:1].cpu())

            # Model-predicted CoT text (G05 uses "cot_text"; galaxea_zero uses "generated_cot")
            pred_cot = batch.get("generated_cot")
            if pred_cot is None:
                pred_cot = batch.get("cot_text")
            if pred_cot is not None:
                all_pred_cot.extend(pred_cot)
            else:
                all_pred_cot.extend([""] * batch["pixel_values"].shape[0])

            # GT plan step text (extracted by processor into samples)
            for sample_dict in batch["samples"]:
                gt_step = sample_dict.get("plan_step", "")
                if gt_step is None:
                    gt_step = sample_dict.get("atomic_task", "")
                all_gt_cot.append(str(gt_step) if gt_step else "")

        # ---- Postprocess: denormalize + split into per-body-part dicts ----
        # GT action + state through full postprocess
        gt_post_data = {
            "action": action_gt.clone(),
            "proprio": proprio_gt.clone(),
            "action_dim_is_pad": batch.get("action_dim_is_pad"),
            "state_dim_is_pad": batch.get("proprio_dim_is_pad"),
        }
        # Include action_op_mask so action_state_merger.backward splits it into parts.
        # op_mask shape may be (B, D) or (B, 1, D); normalize to (B, D) for splitting.
        _op_mask = batch.get("action_op_mask")
        if isinstance(_op_mask, torch.Tensor):
            _op_mask = _op_mask.cpu().bool()
            if _op_mask.dim() == 3 and _op_mask.shape[1] == 1:
                _op_mask = _op_mask.squeeze(1)
            gt_post_data["action_op_mask"] = _op_mask.clone()
        gt_post_data = processor.postprocess(gt_post_data)
        gt_actions.append(dict_apply(gt_post_data["action"], _to_np))
        gt_st = dict_apply(gt_post_data["state"], lambda x: x[:, -1, :] if x.ndim == 3 else x)
        gt_states.append(dict_apply(gt_st, _to_np))
        if isinstance(gt_post_data.get("action_op_mask"), dict):
            has_op_mask = True
            op_masks.append(dict_apply(gt_post_data["action_op_mask"], _to_np))

        if has_model:
            if "action_ar" in batch:
                has_ar_action = True
                # Only copy the keys postprocess actually needs — avoids deepcopy of pixel_values
                ar_batch = {
                    "action": batch["action_ar"],
                    "proprio": batch.get("proprio"),
                    "action_dim_is_pad": batch.get("action_dim_is_pad"),
                    "proprio_dim_is_pad": batch.get("proprio_dim_is_pad"),
                }
                ar_batch = processor.postprocess(ar_batch)
                ar_actions.append(dict_apply(ar_batch["action"], _to_np))

            # Only copy the keys postprocess actually needs — avoids deepcopy of pixel_values
            pd_batch = {
                "action": batch["action"],
                "action_fm": batch.get("action_fm"),
                "action_ar": batch.get("action_ar"),
                "proprio": batch.get("proprio"),
                "action_dim_is_pad": batch.get("action_dim_is_pad"),
                "proprio_dim_is_pad": batch.get("proprio_dim_is_pad"),
                "action_op_mask": batch.get("action_op_mask"),
            }
            pd_batch = {k: v for k, v in pd_batch.items() if v is not None}
            pd_batch = processor.postprocess(pd_batch)
            pd_actions.append(dict_apply(pd_batch["action"], _to_np))

        if (i + 1) * cfg.batch_size_val >= eval_end_frame:
            break

    # ==================== Per-episode visualization ====================
    has_gt_parts = len(gt_actions) > 0
    if has_gt_parts:
        gt_actions = concat_dict_list(gt_actions)
        gt_step0 = {k: v[:, 0, :] for k, v in gt_actions.items()}
    else:
        gt_step0 = None

    if has_model:
        pd_actions = concat_dict_list(pd_actions)  # {part: (N, chunk, dim)}
    if has_ar_action:
        ar_actions = concat_dict_list(ar_actions)

    has_gt_states = len(gt_states) > 0
    if has_gt_states:
        gt_states = concat_dict_list(gt_states)  # {part: (N, dim)}

    if has_op_mask:
        op_masks = concat_dict_list(op_masks)  # {part: (N, dim)} bool

    # Concat normalized flat tensors and filter pad dims (model mode only)
    raw_gt_step0 = None
    raw_pd_concat = None
    if has_model:
        raw_pd_actions = np.concatenate(raw_pd_actions, axis=0)  # (N, T, D)
        raw_gt_actions = np.concatenate(raw_gt_actions, axis=0)  # (N, T, D)
        raw_gt_step0 = raw_gt_actions[:, 0, :]  # (N, D) -- chunk step 0
        valid_dims = ~action_dim_is_pad_mask  # (D,) bool
        raw_pd_concat = raw_pd_actions[:, :, valid_dims]  # (N, T, valid_D)
        raw_gt_step0 = raw_gt_step0[:, valid_dims]  # (N, valid_D)

    # ---- Per-episode visualization (unified) ----
    if has_gt_parts and gt_step0 is not None:
        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        N = next(iter(gt_step0.values())).shape[0]
        num_vis = min(eval_episodes_num, len(valid_episode_indices))

        for idx in range(num_vis):
            ep_idx = valid_episode_indices[idx]
            fr = int(all_ep_from[ep_idx]) - frame_offset
            to = int(all_ep_to[ep_idx]) - frame_offset
            if to > N:
                break

            ep_dir = episodes_dir / f"{idx:06}"
            cur_gt = {k: v[fr:to] for k, v in gt_step0.items()}
            cur_pred = {k: v[fr:to] for k, v in pd_actions.items()} if has_model else None
            cur_secondary = {k: v[fr:to] for k, v in ar_actions.items()} if has_ar_action else None

            extra_pkl = {}
            if cur_secondary is not None:
                extra_pkl["ar_pd"] = cur_secondary

            cur_state = None
            if has_gt_states:
                cur_state = {k: v[fr:to] for k, v in gt_states.items()}
                extra_pkl["state"] = cur_state

            cur_op_mask = None
            if has_op_mask:
                cur_op_mask = {k: v[fr:to] for k, v in op_masks.items()}
                extra_pkl["op_mask"] = cur_op_mask

            visualize_episode(
                episode_dir=ep_dir,
                gt_step0=cur_gt,
                pred_chunks=cur_pred,
                secondary_chunks=cur_secondary,
                raw_gt_step0=raw_gt_step0[fr:to] if raw_gt_step0 is not None else None,
                raw_pred_chunks=raw_pd_concat[fr:to] if raw_pd_concat is not None else None,
                gt_state=cur_state,
                op_mask=cur_op_mask,
                extra_pkl_data=extra_pkl,
            )
            logger.info(f"Episode {idx} saved to {ep_dir}")

        logger.info(f"Visualization complete: {episodes_dir}")

    # ==================== CoT video rendering ====================
    if save_cot_video and len(all_pixel_values) > 0:
        from g05.utils.eval.visualize import plot_subtask

        cot_pixels = torch.cat(all_pixel_values, dim=0)  # (N, 1, C, H, W)

        # Derive original camera resolution from processor shape_meta (raw_shape = [C, H, W]).
        # pixel_values are VLM-preprocessed (e.g. 224×224 square); resize back for display.
        _cot_output_size = None
        _image_meta = getattr(processor, "shape_meta", {}).get("images", [])
        if _image_meta:
            _raw = _image_meta[0].get("raw_shape")  # [C, H, W]
            if _raw and len(_raw) == 3:
                _cot_output_size = (_raw[2], _raw[1])  # (W, H) for cv.resize

        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        num_vis = min(eval_episodes_num, len(valid_episode_indices))

        for idx in range(num_vis):
            ep_idx = valid_episode_indices[idx]
            fr = int(all_ep_from[ep_idx]) - frame_offset
            to = int(all_ep_to[ep_idx]) - frame_offset
            if to > cot_pixels.shape[0]:
                break

            ep_dir = episodes_dir / f"{idx:06}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            plot_subtask(
                path=ep_dir / "cot_video.mp4",
                pixel_values=cot_pixels[fr:to],
                pred_subtask=all_pred_cot[fr:to],
                gt_subtask=all_gt_cot[fr:to],
                fps=15,
                output_size=_cot_output_size,
            )

        del cot_pixels, all_pixel_values
        logger.info("CoT videos saved.")

    # ==================== Metric summary (model mode only) ====================
    if has_model and all_metrics:
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        logger.info("\n=== Rollout Metrics ===")
        for k, v in avg_metrics.items():
            logger.info(f"  {k}: {v:.6f}")

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")


def evaluate_client_mode(
    dataset: GalaxeaLerobotDataset,
    processor: BaseProcessor,
    output_dir: Path,
    cfg: DictConfig,
    ws_client,
    _event_loop,
    embodiment_name: str | None = None,
    record: bool = False,
):
    """Client-mode evaluation against a protocol-compliant server (scripts/serve.py).

    Each frame: send raw obs → receive a full action chunk ``(chunk_size, dim)``
    → accumulate. The server is stateless w.r.t. inference (no chunk caching),
    so the data path mirrors local-mode ``evaluate_single_dataset`` — both
    produce ``pd_actions = list[dict{part: (1, chunk, dim)}]``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _to_np = lambda x: x.numpy() if isinstance(x, torch.Tensor) else x

    # ---- shape_meta sanity check (handshake metadata captured during connect) ----
    sm = (getattr(ws_client, "metadata", None) or {}).get("shape_meta") or {}
    if sm:
        proc_meta = processor.shape_meta
        srv_state_keys = {m["key"] for m in sm.get("state", [])}
        proc_state_keys = {m["key"] for m in proc_meta.get("state", [])}
        if srv_state_keys and srv_state_keys != proc_state_keys:
            logger.warning(
                "Server shape_meta state keys %s differ from local processor %s — "
                "obs key set may be misaligned",
                srv_state_keys,
                proc_state_keys,
            )

    action_collate_fn = partial(collate_fn_pad_sequences, padding_input_id=processor.pad_token_id)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=action_collate_fn,
    )

    frame_offset, frame_end, valid_episode_indices = compute_valid_episode_indices(dataset)
    all_ep_from = dataset.episode_data_index["from"]
    all_ep_to = dataset.episode_data_index["to"]
    num_episodes = len(valid_episode_indices)
    eval_episodes_num = (
        min(cfg.eval_episodes_num, num_episodes) if cfg.get("eval_episodes_num") else num_episodes
    )
    last_eval_ep = valid_episode_indices[eval_episodes_num - 1]
    eval_end_frame = int(all_ep_to[last_eval_ep]) - frame_offset

    # ==================== Inference loop ====================
    # Every frame: obs → server returns full chunk (chunk_size, dim) → accumulate.
    gt_actions = []  # list[dict{part: (1, chunk, dim)}]
    pd_actions = []  # list[dict{part: (1, chunk, dim)}]
    gt_states = []  # list[dict{part: (1, dim)}]

    recorder = None
    if record:
        from scripts.utils.episode_recorder import EpisodeRecorder

        recorder = EpisodeRecorder(output_dir / "recordings")

    ep_start_to_task = {}
    ep_end_frames = set()
    for ep_idx in valid_episode_indices[:eval_episodes_num]:
        fr = int(all_ep_from[ep_idx]) - frame_offset
        to = int(all_ep_to[ep_idx]) - frame_offset
        ep_start_to_task[fr] = f"episode_{ep_idx}"
        ep_end_frames.add(to - 1)

    for i, batch in tqdm(enumerate(dataloader), desc="client eval", total=eval_end_frame):
        # ---- GT: every frame ----
        action_gt = batch["action"].clone()
        proprio_gt = batch["proprio"].clone()
        gt_post_data = {
            "action": action_gt,
            "proprio": proprio_gt,
            "action_dim_is_pad": batch.get("action_dim_is_pad"),
            "state_dim_is_pad": batch.get("proprio_dim_is_pad"),
        }
        gt_post_data = processor.postprocess(gt_post_data)
        gt_actions.append(dict_apply(gt_post_data["action"], _to_np))
        gt_st = dict_apply(gt_post_data["state"], lambda x: x[:, -1, :] if x.ndim == 3 else x)
        gt_states.append(dict_apply(gt_st, _to_np))

        # ---- Pred: send raw obs, receive full chunk ----
        abs_idx = (
            int(batch["idx"].item())
            if isinstance(batch["idx"], torch.Tensor)
            else int(batch["idx"])
        )
        sample_idx = abs_idx - getattr(dataset, "_start_idx", 0)
        saved_processor = dataset.processor
        dataset.processor = None
        try:
            raw_sample = dataset[sample_idx]
        finally:
            dataset.processor = saved_processor
        raw_obs = _extract_raw_obs(raw_sample, embodiment=embodiment_name)

        response = _event_loop.run_until_complete(ws_client.infer(raw_obs))
        # Server-side errors arrive as a text frame followed by close → surfaces
        # as ConnectionClosed below; we don't need a payload-level error check.

        # Server returns full chunk: {part: ndarray[chunk, dim]}
        chunk_action = {
            part: np.asarray(arr)[np.newaxis, ...]  # (1, chunk, dim)
            for part, arr in response["action"].items()
        }
        pd_actions.append(chunk_action)

        # ---- Recorder ----
        if recorder is not None:
            if i in ep_start_to_task:
                if recorder.active:
                    recorder.stop_and_save()
                recorder.start(ep_start_to_task[i])

            gt_step0_i = {k: v[0, 0, :] for k, v in gt_post_data["action"].items()}
            gt_step0_i = {
                k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in gt_step0_i.items()
            }
            # use chunk step 0 as the "action this frame would issue"
            pred_i = {k: v[0, 0, :] for k, v in chunk_action.items()}

            recorder.add_step(
                images=raw_obs.get("images"),
                gt_action=gt_step0_i,
                pred_action=pred_i,
                extra={k: v for k, v in response.items() if k not in ("action", "state")},
            )

            if i in ep_end_frames:
                recorder.stop_and_save()

        if i + 1 >= eval_end_frame:
            break

    if not gt_actions:
        logger.warning("No frames collected, skipping visualization")
        return

    # ==================== Per-episode visualization ====================
    # Same structure as evaluate_single_dataset: GT step0 (N, dim), Pred full
    # chunks (N, chunk, dim).
    gt_actions = concat_dict_list(gt_actions)  # {part: (N, chunk, dim)}
    gt_step0 = {k: v[:, 0, :] for k, v in gt_actions.items()}  # {part: (N, dim)}
    gt_states = concat_dict_list(gt_states)  # {part: (N, dim)}
    pd_chunks = concat_dict_list(pd_actions)  # {part: (N, chunk, dim)}

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    N = next(iter(gt_step0.values())).shape[0]
    num_vis = min(eval_episodes_num, len(valid_episode_indices))

    for idx in range(num_vis):
        ep_idx = valid_episode_indices[idx]
        fr = int(all_ep_from[ep_idx]) - frame_offset
        to = int(all_ep_to[ep_idx]) - frame_offset
        if to > N:
            break

        cur_gt = {k: v[fr:to] for k, v in gt_step0.items()}
        cur_pred = {k: v[fr:to] for k, v in pd_chunks.items()}
        cur_state = {k: v[fr:to] for k, v in gt_states.items()}

        ep_dir = episodes_dir / f"{idx:06}"
        visualize_episode(
            episode_dir=ep_dir,
            gt_step0=cur_gt,
            pred_chunks=cur_pred,
            gt_state=cur_state,
            secondary_chunks=None,
            raw_gt_step0=None,
            raw_pred_chunks=None,
            extra_pkl_data={"state": cur_state},
        )
        ep_len = to - fr
        logger.info(f"Episode {idx} ({ep_len} frames) saved to {ep_dir}")

    logger.info(f"Client-mode visualization complete: {episodes_dir}")


def run_eval(
    cfg: DictConfig,
    no_model: bool = False,
    client_mode: bool = False,
    server_url: str = "ws://localhost:8765",
    record: bool = False,
) -> None:
    """Core evaluation logic, callable from both Hydra and standalone mode."""
    setup_logging(log_level=logging.INFO, is_main_process=True)

    if cfg.get("seed"):
        set_global_seed(cfg.seed, get_worker_init_fn=False)

    # Apply max_datasets truncation (debug mode). Default 1 preserves the
    # historical behavior of the standalone eval path; pass `max_datasets=0`
    # or `max_datasets=null` on the CLI to disable and use every dataset_dir.
    max_datasets = cfg.get("max_datasets", 1)
    if max_datasets:
        truncate_datasets(cfg, int(max_datasets))

    output_dir = Path(os.path.abspath(os.path.expanduser(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # ---- Model / WebSocket client setup ----
    policy = None
    ws_client = None
    _event_loop = None

    if client_mode:
        import time as _time
        from scripts.utils.policy_ws_client import PolicyWebSocketClient

        _event_loop = asyncio.new_event_loop()
        ws_client = PolicyWebSocketClient(uri=server_url)
        max_retries, retry_interval = 60, 5  # wait up to 5 min
        for attempt in range(1, max_retries + 1):
            try:
                metadata = _event_loop.run_until_complete(ws_client.connect())
                logger.info(f"Connected to policy server at {server_url}, metadata={metadata}")
                break
            except (OSError, ConnectionRefusedError) as e:
                if attempt == max_retries:
                    _event_loop.close()
                    raise ConnectionError(
                        f"Failed to connect to policy server at {server_url} "
                        f"after {max_retries * retry_interval}s: {e}"
                    ) from e
                logger.info(
                    f"Waiting for server at {server_url} (attempt {attempt}/{max_retries})..."
                )
                _time.sleep(retry_interval)
    elif not no_model:
        from accelerate import PartialState

        partial_state = PartialState()
        partial_state.config = cfg

        from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint

        policy = load_model_from_checkpoint(
            cfg.model.model_arch, cfg.ckpt_path, extra_prefixes=["normalizer."]
        )
        # InputPreprocessor is not nn.Module, so model.cuda() doesn't reach the action tokenizer
        if hasattr(policy, "action_tokenizer"):
            policy.action_tokenizer.to("cuda")

    # NOTE: use pretrained norm stats
    # Prefer run_dir (set by load_config_from_run_dir in standalone mode);
    # fallback to ckpt parent.parent for Hydra mode or legacy paths.
    if cfg.get("run_dir"):
        stats_path = Path(cfg.run_dir) / "dataset_stats.json"
    else:
        stats_path = Path(cfg.ckpt_path).parent.parent / "dataset_stats.json"
    dataset_stats = load_dataset_stats_from_json(stats_path)

    # eval_split: "val" (default) or "train"
    eval_split = cfg.get("eval_split", "val")
    is_training_set = eval_split == "train"
    eval_embodiment = cfg.get("eval_embodiment", None)
    logger.info(f"Eval split: {eval_split}")

    # Detect config layout: mixture (embodiment_datasets) vs flat (dataset_dirs)
    is_mixture = "embodiment_datasets" in cfg.data

    if is_mixture:
        logger.info("Dataset layout: mixture (embodiment_datasets)")
        if eval_embodiment:
            filter_embodiment(cfg, eval_embodiment)
    else:
        num_dirs = len(cfg.data.get("dataset_dirs") or [])
        logger.info(f"Dataset layout: flat ({num_dirs} dataset_dirs)")
        if eval_embodiment:
            logger.warning(
                f"eval_embodiment={eval_embodiment!r} is ignored for flat dataset config"
            )

    # Instantiate dataset (strips 'processors' key that belongs to build_processors)
    dataset_eval = instantiate_dataset(cfg, is_training_set=is_training_set)

    try:
        if isinstance(dataset_eval, MixtureLerobotDataset):
            # --- Mixture dataset: build per-embodiment processors and evaluate per-embodiment ---
            mixture_processor = build_processors(cfg)
            mixture_processor.set_normalizer_from_stats(dataset_stats)
            mixture_processor.eval()

            dataset_eval.set_processor(mixture_processor)

            for emb, ds in zip(dataset_eval.embodiments, dataset_eval.datasets):
                emb_type = dataset_eval.embodiments2types[emb]
                emb_processor = mixture_processor[emb_type]
                emb_output_dir = output_dir / emb
                mode_str = (
                    "Client eval"
                    if client_mode
                    else ("Visualizing GT" if no_model else "Evaluating")
                )
                logger.info(f"\n=== {mode_str}: {emb} ({eval_split} split) ===")
                if client_mode:
                    evaluate_client_mode(
                        ds,
                        emb_processor,
                        emb_output_dir,
                        cfg,
                        ws_client=ws_client,
                        _event_loop=_event_loop,
                        embodiment_name=emb,
                        record=record,
                    )
                else:
                    evaluate_single_dataset(policy, ds, emb_processor, emb_output_dir, cfg)
                print(f"Results saved to: {emb_output_dir}")
        else:
            # --- Single dataset: original flow ---
            processor: BaseProcessor = build_processors(cfg)
            processor.set_normalizer_from_stats(dataset_stats)
            processor.eval()
            dataset_eval.set_processor(processor)
            if client_mode:
                evaluate_client_mode(
                    dataset_eval,
                    processor,
                    output_dir,
                    cfg,
                    ws_client=ws_client,
                    _event_loop=_event_loop,
                    record=record,
                )
            else:
                evaluate_single_dataset(policy, dataset_eval, processor, output_dir, cfg)
    finally:
        # ---- Cleanup WebSocket client ----
        if ws_client is not None:
            _event_loop.run_until_complete(ws_client.close())
            logger.info("WebSocket client closed")
        if _event_loop is not None:
            _event_loop.close()

    print(f"\nOutput dir: {output_dir}")


if __name__ == "__main__":
    # Non-Hydra entry (we need to locate a checkpoint's run_dir before composing a
    # config), but all eval behavior is driven by cfg. Use Hydra-style `key=value`
    # CLI overrides to tweak anything in the loaded config, e.g.:
    #   max_datasets=1 eval_episodes_num=2 save_cot_video=true
    #   eval_embodiment=galaxea_r1lite
    parser = argparse.ArgumentParser(description="Open-loop eval / GT-only visualization")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to checkpoint (.pt file)"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="Path to task yaml (e.g. configs/task/pretrain/bench/foo.yaml). "
        "When set, load config via Hydra compose instead of from the "
        "checkpoint's run directory.",
    )
    parser.add_argument(
        "--no_model",
        action="store_true",
        help="Skip model loading — only visualize GT actions and states",
    )
    parser.add_argument(
        "--client_mode",
        action="store_true",
        help="Use WebSocket client to query serve_policy instead of local model",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket server URL for --client_mode (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episodes as video in --client_mode (saved to output_dir/recordings/)",
    )
    args, remaining = parser.parse_known_args()

    if args.client_mode and args.no_model:
        parser.error("--client_mode and --no_model are mutually exclusive")

    # remaining are key=value overrides (same syntax as Hydra overrides)
    overrides = [r for r in remaining if "=" in r]

    if args.task_config:
        cfg = load_config_from_task_yaml(args.task_config, args.ckpt_path, overrides)
    else:
        run_dir = find_run_dir(args.ckpt_path)
        print(f"Found run dir: {run_dir}")
        cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    run_eval(
        cfg,
        no_model=args.no_model,
        client_mode=args.client_mode,
        server_url=args.server_url,
        record=args.record,
    )

# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
G05Policy: policy wrapper for G05Model.

Simplified refactor from GalaxeaJointPolicy:
- no longer inherits GalaxeaARMixin; AR decode lives in G05Model.ar_helper
- model is G05Model, not GalaxeaJoint/PiAR
- FM/AR algorithm details are encapsulated inside G05Model via fm_helper/ar_helper;
  Policy does not need to know them

Based on GalaxeaJointPolicy refactoring.
"""

from __future__ import annotations

import datetime
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from g05.models.base_policy import BasePolicy

from g05.utils.training.action_accuracy import ActionAccuracyEvaluator
from g05.utils.common.import_utils import get_obj_from_str
from g05.utils.logging.log_box import log_box
from g05.utils.logging.logging_config import get_logger
from g05.utils.training.accuracy_accumulator import TrainAccuracyAccumulator

from .io.input_preprocessor import InputPreprocessor
from .helpers.proprio_helper import build_proprio_batch
from .g05_model import G05Model

logger = get_logger(__name__)


@dataclass
class InferenceState:
    """Unified state for three-stage inference.

    prefill() creates the initial state, generate_text() updates it and appends
    generated results, and generate_action() consumes it to generate actions. One
    dataclass is carried through the entire flow.
    """

    # Core KV cache state: these three must have the same length and aligned content.
    kv_cache: list  # List[(K,V)] per layer
    attention_mask: torch.Tensor  # [B, S] TOKEN_INDEX values
    position_ids: torch.Tensor  # [B, S] 0-indexed, synced by mask_helper.compute_position_ids
    last_hidden: torch.Tensor  # [B, d_vlm] context-ending hidden for sampling the first AR token
    pixel_values: Dict[str, torch.Tensor]
    input_ids: torch.Tensor  # [B, S] raw input_ids
    device: torch.device = None
    generated_texts: Optional[List[str]] = None
    generated_ids: Optional[torch.Tensor] = None

    def check_invariants(self, where: str = "") -> None:
        from .model.utils import kv_cache_seq_len

        kv_len = kv_cache_seq_len(self.kv_cache)
        am_len = self.attention_mask.size(-1)
        pid_len = self.position_ids.size(-1)
        bsz = self.attention_mask.size(0)
        tag = f" [{where}]" if where else ""
        if not (kv_len == am_len == pid_len):
            raise AssertionError(
                f"InferenceState length mismatch{tag}: "
                f"kv_cache={kv_len}, attention_mask={am_len}, position_ids={pid_len}. "
                f"All three must grow in sync."
            )
        if self.last_hidden.size(0) != bsz:
            raise AssertionError(
                f"InferenceState batch mismatch{tag}: "
                f"last_hidden batch={self.last_hidden.size(0)}, "
                f"attention_mask batch={bsz}"
            )
        # position_ids: [B, S] for G05, [3, B, S] for Qwen3.5 MRoPE — batch dim differs
        pid_bsz = (
            self.position_ids.size(1) if self.position_ids.ndim == 3 else self.position_ids.size(0)
        )
        if pid_bsz != bsz:
            raise AssertionError(
                f"InferenceState batch mismatch{tag}: "
                f"position_ids batch={pid_bsz}, "
                f"attention_mask batch={bsz}"
            )


@torch.compiler.disable
def _model_log(fh, msg_fn):
    if fh is None:
        return
    msg = msg_fn() if callable(msg_fn) else msg_fn
    fh.write(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} {msg}\n")
    fh.flush()


def _sync_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class G05Policy(BasePolicy):
    """G05 Policy wrapper.

    Assembles G05Model + InputPreprocessor + ActionTokenizer.
    FM/AR algorithm details are encapsulated inside G05Model, while Policy owns
    preprocessing and routing.

    Subclasses adapt to different VLM backbones by overriding these class variables
    without rewriting __init__:
      model_cls                — internal model class
      default_base_vocab_size  — pretrained vocab size
      default_padded_vocab_size— padded/aligned vocab size
      default_hf_processor_class — HF Processor class path
      default_model_type       — model_type string passed to InputPreprocessor
    Post-processing is extended via the _post_processor_init() hook.
    """

    # Subclasses override these 5 class variables to adapt different backbones.
    # The base class does not bind a concrete backbone: processor class and model_type
    # must be provided by subclasses such as G05PolicyQwen35, or by config
    # hf_processor_class.
    model_cls: type = G05Model
    default_base_vocab_size: int = 257153
    default_padded_vocab_size: int = 257216
    default_hf_processor_class: Optional[str] = None
    default_model_type: Optional[str] = None

    @property
    def fp32_param_patterns(self) -> List[str]:
        return [
            # vision embeddings
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            # per-layer norms (substring matches vlm.layers[i] & action_expert.layers[i])
            "input_layernorm",
            "post_attention_layernorm",
            # final norms (G05: model.vlm.norm / model.action_expert.norm)
            "vlm.norm",
            "action_expert.norm",
            # action expert I/O projections + time conditioning
            "action_expert.input_proj",
            "action_expert.output_proj",
            "action_expert.time_embedding",
            "action_expert.time_mlp_in",
            "action_expert.time_mlp_out",
            # proprio encoder & norms
            "proprio_embedder",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ]

    def __init__(self, **model_cfg: DictConfig) -> None:
        super().__init__()
        model_cfg = OmegaConf.create(model_cfg)
        self.config = model_cfg

        # NOTE: model_family is for external code to identify the model family.
        # g05 v2 does not use it internally; PurePromptBuilder/get_prompt_builder is deprecated.
        self.model_family = "g05"
        self.default_image_resolution = (3, 224, 224)
        self.norm_stats = {}

        # --- Build model (subclass may override model_cls) ---
        if model_cfg.pretrained_model_path:
            self.model = self.model_cls.from_pretrained(model_cfg)
        else:
            self.model = self.model_cls(model_cfg)

        self.model_config = model_cfg

        # --- Vocab constants (from config, fallback to class-level defaults) ---
        self._base_vocab_size = model_cfg.get("base_vocab_size", self.default_base_vocab_size)
        self._padded_vocab_size = model_cfg.get("padded_vocab_size", self.default_padded_vocab_size)

        # --- Processor (centralized token registration + embedding resize) ---
        pp_cfg = model_cfg.input_preprocessor
        # hf_processor_path: ckpt_utils nulls pretrained_model_path during eval/serve,
        # but preserves the original path in hf_processor_path so this code can load
        # the tokenizer.
        hf_path = model_cfg.get("hf_processor_path", None) or model_cfg.pretrained_model_path
        hf_processor_class = model_cfg.get("hf_processor_class", self.default_hf_processor_class)
        if hf_processor_class is None or self.default_model_type is None:
            raise ValueError(
                f"{type(self).__name__} is not bound to a VLM backbone: use a concrete subclass "
                "such as G05PolicyQwen35, or provide hf_processor_class in the model config."
            )
        self.processor = InputPreprocessor(
            hf_processor_path=hf_path,
            hf_processor_class=hf_processor_class,
            action_tokenizer_class=model_cfg.action_tokenizer,
            at_config=model_cfg.AT_CONFIG,
            model=self.model,
            base_vocab_size=self._base_vocab_size,
            padded_vocab_size=self._padded_vocab_size,
            pad_token_id=model_cfg.pad_token_id,
            image_token_index=model_cfg.image_token_index,
            num_image_tokens=model_cfg.vision.num_image_tokens,
            input_action_corruption=pp_cfg.input_action_corruption,
            pred_eov=pp_cfg.pred_eov,
            batchify_action=pp_cfg.batchify_action,
            pi05_ft_mode=pp_cfg.pi05_ft_mode,
            proprio_encoder=model_cfg.proprio_encoder,
            model_type=self.default_model_type,
            model_cfg=model_cfg,
            add_loc_tokens=bool(model_cfg.get("add_loc_tokens", False)),
        )
        self.action_tokenizer = self.processor.action_tokenizer
        self._post_processor_init()

        # --- Flags ---
        self.discrete_action = model_cfg.discrete_action
        self.continuous_action = model_cfg.continuous_action
        self.return_continuous_action = model_cfg.return_continuous_action
        self.predict_cot = model_cfg.predict_cot
        self.language_loss_weight = float(model_cfg.get("language_loss_weight", 1.0))
        self.num_input_images = model_cfg.num_input_images
        self.max_chunk_token_length = model_cfg.max_chunk_token_length
        self.max_pad_token_length = model_cfg.max_pad_token_length

        # --- Sync token config to AR helper. The owner exposes it; policy only reads. ---
        tc = self._get_runtime_action_token_config()
        if tc["action_token_begin_idx"] != tc["action_token_end_idx"]:
            self.model.ar_helper.set_token_index_ranges(
                tc["action_token_begin_idx"], tc["action_token_end_idx"]
            )
        if tc.get("bos_blk_id") is not None:
            ar = self.model.ar_helper
            ar.bos_blk_id = tc["bos_blk_id"]
            ar.eos_blk_id = tc["eos_blk_id"]
            ar.block_size = tc["block_size"]
            logger.info(
                f"[G05Policy] BAR synced: bos_blk={ar.bos_blk_id}, "
                f"eos_blk={ar.eos_blk_id}, block_size={ar.block_size}"
            )
        self._sync_bar_runtime(tc)

        # --- Sync EOV token ID, used as AR generate_text stop token. ---
        self.model.ar_helper.eov_token_id = self.processor.eov_token_id
        logger.info(f"[G05Policy] EOV token synced: eov_token_id={self.processor.eov_token_id}")

        # --- Action Accuracy Evaluator (decoupled from policy) ---
        bb = self.processor.bar_builder
        self.action_evaluator = ActionAccuracyEvaluator(
            self.action_tokenizer,
            tokenizer=self.processor.tokenizer,
            block_wise=bb.block_wise,
            bos_blk_id=bb.bos_blk_id,
            eos_blk_id=bb.eos_blk_id,
        )

        # --- Debug log ---
        self._model_log_fh = None
        self._fwd_step = 0

        # --- Train-time accuracy accumulator (grad-accum aware) ---
        # Each micro-batch pushes three accuracies (overall/action_token/cot).
        # finetune.py calls flush_train_accuracy() at log steps to get micro-batch
        # means. This avoids self.train_* being overwritten by later micro-batches
        # and losing all but the last batch when grad_accum > 1.
        self._train_acc = TrainAccuracyAccumulator()

        # ── Box: Policy Config ──────────────────────────────────────
        _ar_status = "✅ enabled" if (self.discrete_action or self.predict_cot) else "❌ disabled"
        _fm_status = "✅ enabled" if self.continuous_action else "❌ disabled"
        _bar_rows = []
        if tc.get("bos_blk_id"):
            ar = self.model.ar_helper
            _bar_rows = [
                ("BAR", f"bos={ar.bos_blk_id}  eos={ar.eos_blk_id}  block_size={ar.block_size}")
            ]

        _cond_steps = model_cfg.get("cond_steps", model_cfg.get("num_obs_steps", 1))
        _horizon_steps = model_cfg.get("horizon_steps", "?")
        _num_input_images = model_cfg.get("num_input_images", "?")
        _num_cameras = "?"
        if isinstance(_num_input_images, int) and isinstance(_cond_steps, int) and _cond_steps > 0:
            _num_cameras = _num_input_images // _cond_steps

        log_box(
            logger,
            "🤖  G05 Policy Config",
            [
                ("AR Path", f"{_ar_status}  (discrete_action={self.discrete_action})"),
                ("FM Path", f"{_fm_status}  (continuous_action={self.continuous_action})"),
                ("predict_cot", str(self.predict_cot)),
                ("return_continuous", str(self.return_continuous_action)),
            ],
        )
        log_box(
            logger,
            "📐  Sequence Config",
            [
                ("obs_steps", str(_cond_steps)),
                ("action_steps", str(_horizon_steps)),
                ("cameras", str(_num_cameras)),
                ("input_images", str(_num_input_images)),
            ],
        )
        if _bar_rows:
            log_box(logger, "🔧  BAR Config", _bar_rows)

    def _get_runtime_action_token_config(self) -> Dict[str, Any]:
        token_configs = getattr(self.processor, "token_configs", None)
        if isinstance(token_configs, dict):
            action_cfg = token_configs.get("action")
            if isinstance(action_cfg, dict):
                return dict(action_cfg)

        token_config = getattr(self.action_tokenizer, "token_config", {})
        if callable(token_config):
            token_config = token_config()
        return dict(token_config)

    def _sync_bar_runtime(self, tc: Dict[str, Any]) -> None:
        ar = self.model.ar_helper
        if tc.get("bos_blk_id") is not None:
            ar.bos_blk_id = tc["bos_blk_id"]
            ar.eos_blk_id = tc.get("eos_blk_id")
            ar.block_size = tc.get("block_size")

        runtime_bar = bool(
            self.action_tokenizer.block_wise_autoregressive and tc.get("bos_blk_id") is not None
        )
        if ar.block_wise_autoregressive != runtime_bar:
            logger.warning(
                f"BAR config mismatch: model_cfg={ar.block_wise_autoregressive} "
                f"tokenizer={self.action_tokenizer.block_wise_autoregressive} "
                f"bos_blk_id={tc.get('bos_blk_id')} "
                f"→ forcing runtime_bar={runtime_bar}"
            )
            ar.block_wise_autoregressive = runtime_bar

        logger.info(
            f"BAR runtime state: tokenizer={self.action_tokenizer.block_wise_autoregressive} "
            f"model_cfg={ar.block_wise_autoregressive} "
            f"(after sync) bos_blk_id={tc.get('bos_blk_id')} "
            f"eos_blk_id={tc.get('eos_blk_id')} block_size={tc.get('block_size')}"
        )

    def _post_processor_init(self) -> None:
        """Post-processing hook after processor and action_tokenizer are ready.

        Subclasses may override for backbone-specific initialization, such as Qwen3.5
        action token offset fixes. The base class syncs the <state> token id to
        proprio_embedder for PaliGemma mlp mode.
        """
        if self.model.proprio_embedder is not None:
            self.model.proprio_embedder.state_token_id = self.processor.state_token_id
            logger.info(
                f"[{self.__class__.__name__}] state token synced: "
                f"state_token_id={self.processor.state_token_id}"
            )

    # ------------------------------------------------------------------
    # Train-time accuracy flush (grad-accum aware)
    # ------------------------------------------------------------------

    def flush_train_accuracy(self) -> Dict[str, float]:
        """Delegate to ``TrainAccuracyAccumulator.flush()``.

        finetune.py calls this once per log step. It returns means over all
        micro-batches in the current log window, clears the accumulator, and starts
        the next window fresh. The returned dict contains only fields pushed during
        this window (overall / action_token / cot), so callers should use
        ``"field" in out`` to decide whether to log instead of overwriting with a
        default 0.0.
        """
        return self._train_acc.flush()

    # ------------------------------------------------------------------
    # Optimizer param groups
    # ------------------------------------------------------------------

    def get_optim_param_groups(
        self,
        lr,
        weight_decay,
        apply_decay_on_norm_and_bias=False,
        backbone_lr_multiplier=1.0,
        vision_lr_multiplier=1.0,
    ):
        """Group params into vision tower / VLM backbone / action expert decay groups.

        - vision tower uses `lr * backbone_lr_multiplier * vision_lr_multiplier`,
          a further discount relative to VLM because pretrained SigLIP is easy to
          damage during further training
        - other backbone parts (VLM + multi_modal_projector, etc.) use
          `lr * backbone_lr_multiplier`
        - action expert uses `lr`
        """
        ae_param_ids = set(id(p) for p in self.model.action_expert.parameters())
        vision_param_ids = set(id(p) for p in self.model.vision_tower.parameters())
        modules_by_name = dict(self.model.named_modules())

        vision_decay, vision_no_decay = [], []
        backbone_decay, backbone_no_decay = [], []
        action_decay, action_no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            owner_name, _, leaf_name = name.rpartition(".")
            owner_module = modules_by_name.get(owner_name)
            apply_wd = self._should_apply_weight_decay(owner_module, leaf_name, param)
            pid = id(param)
            if pid in ae_param_ids:
                (action_decay if apply_wd else action_no_decay).append(param)
            elif pid in vision_param_ids:
                (vision_decay if apply_wd else vision_no_decay).append(param)
            else:
                (backbone_decay if apply_wd else backbone_no_decay).append(param)

        vision_lr = lr * backbone_lr_multiplier * vision_lr_multiplier
        backbone_lr = lr * backbone_lr_multiplier
        param_groups = [
            {
                "params": backbone_decay,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone_decay",
            },
            {
                "params": action_decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "action_decay",
            },
            {
                "params": vision_decay,
                "lr": vision_lr,
                "weight_decay": weight_decay,
                "name": "vision_decay",
            },
            {
                "params": backbone_no_decay,
                "lr": backbone_lr,
                "weight_decay": 0.0,
                "name": "backbone_no_decay",
            },
            {
                "params": action_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "name": "action_no_decay",
            },
            {
                "params": vision_no_decay,
                "lr": vision_lr,
                "weight_decay": 0.0,
                "name": "vision_no_decay",
            },
        ]

        total = sum(len(g["params"]) for g in param_groups)
        expected = len([p for p in self.parameters() if p.requires_grad])
        assert total == expected, f"Param count mismatch: {total} vs {expected}"
        return param_groups

    @staticmethod
    def _should_apply_weight_decay(
        owner_module: Optional[nn.Module],
        leaf_name: str,
        param: nn.Parameter,
    ) -> bool:
        if leaf_name == "bias" or param.ndim <= 1:
            return False
        if isinstance(owner_module, nn.Embedding):
            return False
        return True

    # ------------------------------------------------------------------
    # Forward routing
    # ------------------------------------------------------------------

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(batch, inference_mode=True)

    @staticmethod
    def _build_vqa_template(num_images: int) -> str:
        """Build the VQA template for infer_vqa."""
        from g05.models.g05.io.templates import PALIGEMMA_VQA_TEMPLATE

        if num_images <= 0:
            raise ValueError(f"num_images must be positive, got {num_images}")
        image_template = "".join(f"<image{i}_image_!>" for i in range(num_images))
        return PALIGEMMA_VQA_TEMPLATE.replace("<image_image_!>", image_template)

    def _get_trim_token_ids(self, include_eov: bool = False) -> Optional[List[int]]:
        """Get the common list of trim token IDs."""
        ids = []
        if include_eov:
            eov = self.model.ar_helper.eov_token_id
            if eov is not None:
                ids.append(eov)
        eos = getattr(self.model_config, "eos_token_id", None)
        pad = getattr(self.model_config, "pad_token_id", None)
        if eos is not None:
            ids.append(eos)
        if pad is not None:
            ids.append(pad)
        return ids or None

    def _get_action_generation_max_new_tokens(self) -> int:
        """Return a safe token budget for the discrete action AR stage."""
        ar_default = int(getattr(self.model.ar_helper, "max_new_tokens", 300) or 300)
        action_tokenizer = getattr(self.processor, "action_tokenizer", None)
        if action_tokenizer is None:
            return ar_default

        lengths = []
        for attr in ("action_token_len", "action_token_len_real"):
            value = getattr(action_tokenizer, attr, None)
            if value is not None:
                try:
                    lengths.append(int(value))
                except (TypeError, ValueError):
                    continue
        if not lengths:
            return ar_default

        # The action tokens are followed by static wrapper tokens such as "|"
        # and EOS, and BAR mode may also emit block boundary tokens.
        return max(ar_default, max(lengths) + 16)

    def forward(
        self,
        batch,
        inference_mode=False,
    ):
        """Routing entry point.

        Args:
            batch: collate output dict. See batch_schema.py for the field contract:
                G05TrainBatch for training and G05InferenceBatch for inference.
            inference_mode: True → forward_inference, False → forward_train

        Returns:
            Training: (loss: scalar, loss_dict: Dict[str, Tensor])
            Inference: batch, updated in place with predicted batch["action"]
        """
        samples = batch["samples"]

        if inference_mode:
            was_training = self.training
            self.model.eval()
            generated = self.forward_inference(
                samples=samples,
                pixel_values=batch["pixel_values"],
                actions=batch.get("action"),
                action_dim_is_pad=batch.get("action_dim_is_pad"),
            )
            batch.update(generated)
            self.model.train(was_training)
            return batch
        else:
            return self.forward_train(
                samples=samples,
                pixel_values=batch["pixel_values"],
                actions=batch.get("action"),
                action_pad_masks=batch.get("action_is_pad"),
                action_dim_is_pad=batch.get("action_dim_is_pad"),
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: Optional[torch.FloatTensor] = None,
        action_pad_masks: Optional[torch.BoolTensor] = None,
        action_dim_is_pad: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """Training forward.

        Args:
            samples:           List[Dict] of length B in RoboVQA format for InputPreprocessor
            pixel_values:      Tensor[B, n, C, H, W] or Dict[str, Tensor[B, n_k, C, H_k, W_k]]
            actions:           [B, H, D] normalized GT action
            action_pad_masks:  [B, H] bool, True means this step is padding
            action_dim_is_pad: [B, D] bool, True means this dim is padding (PaddingActionMerger)

        Returns:
            (loss, loss_dict): scalar loss and loss_dict containing fm_loss / ce_loss, etc.
        """
        fh = self._model_log_fh
        step = self._fwd_step
        if isinstance(pixel_values, Dict):
            first_image = next(iter(pixel_values.values()))
            device, dtype = first_image.device, first_image.dtype
        else:
            device, dtype = pixel_values.device, pixel_values.dtype
        batch_size = len(samples)

        # IO checks: actions must be [B, H, D].
        assert actions is not None, "actions required for training"
        assert actions.ndim == 3, f"actions should be [B, H, D], got {actions.shape}"
        assert action_pad_masks is not None, "action_pad_masks is required for training"
        assert action_pad_masks.shape == actions.shape[:2], (
            f"action_pad_masks shape mismatch: {action_pad_masks.shape} vs actions {actions.shape[:2]}"
        )

        # Preprocess input sequences
        input_ids, labels, attention_mask, split_index = self.processor.encode_train(
            samples,
            device=device,
            training=self.training,
            max_chunk_token_length=self.max_chunk_token_length,
            max_pad_token_length=self.max_pad_token_length,
        )

        _model_log(
            fh,
            lambda: (
                f"[FWD] step={step} bs={batch_size} "
                f"seq_len={input_ids.shape[1]} split_index={split_index}"
            ),
        )

        # Temporarily store the full attention_mask (image+text+action) in
        # GlobalMonitor for MetricsCollector.on_forward_end to compute Phase 2
        # sequence stats and padding_ratio. Without this line, the collector always
        # sees None and Phase 2 metrics silently disappear.
        from g05.utils.training.train_utils import get_global_monitor

        _gm = get_global_monitor()
        if _gm is not None:
            _gm.full_attention_mask = attention_mask.detach()

        pixel_values_processed = self.process_pixel_values(pixel_values)

        # Build proprio batch if proprio_encoder is enabled
        proprio_batch = (
            build_proprio_batch(
                samples,
                device=device,
                dtype=torch.float32,
                zero_values=self.model.proprio_encoder == "zeros",
            )
            if self.model.proprio_embedder is not None
            else None
        )

        # Forward through G05Model; time sampling is done inside the model.
        embodiment_types = [s.get("embodiment") for s in samples]
        loss_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_processed,
            split_index=split_index,
            actions=actions,
            action_pad_masks=action_pad_masks,
            action_dim_is_pad=action_dim_is_pad,
            labels=labels,
            continuous_action=self.continuous_action,
            skip_ce_loss=(
                self.continuous_action and not self.discrete_action and not self.predict_cot
            ),
            proprio=proprio_batch,
            embodiment_types=embodiment_types,
        )

        overall_accuracy = loss_dict.pop("overall_accuracy", 0.0)
        action_accuracy = loss_dict.pop("action_accuracy", 0.0)
        cot_accuracy = loss_dict.pop("cot_accuracy", 0.0)

        # Accumulate into TrainAccuracyAccumulator; finetune flushes at log steps.
        # - When grad_accum > 1, keep all micro-batches instead of only the last one.
        # - Log true action_token_accuracy to WandB. The old train_action_accuracy name
        #   actually held overall accuracy; they are now separated.
        self._train_acc.push("overall", overall_accuracy)
        self._train_acc.push("action_token", action_accuracy)
        if self.predict_cot:
            # Record cot_accuracy only when COT prediction is actually enabled; otherwise
            # the always-zero field is just noise.
            self._train_acc.push("cot", cot_accuracy)

        # Instantaneous values from this single forward (current batch only), read
        # directly by metric.py teacher-forcing eval. Training log-step means use
        # flush_train_accuracy().
        self.train_action_accuracy = overall_accuracy
        self.train_cot_accuracy = cot_accuracy if self.predict_cot else 0.0

        if "ce_loss" in loss_dict and self.language_loss_weight != 1.0:
            loss_dict["ce_loss"] = loss_dict["ce_loss"] * self.language_loss_weight
        loss = sum(loss_dict.values())

        loss_value_dict = {k: v.detach() for k, v in loss_dict.items()}

        _model_log(
            fh,
            lambda: (
                f"[LOSS] step={step} loss={float(loss.item()):.6f} "
                + " ".join(f"{k}={float(v.item()):.6f}" for k, v in loss_value_dict.items())
                + f" acc={action_accuracy}"
            ),
        )
        self._fwd_step = step + 1
        return loss, loss_value_dict

    # ------------------------------------------------------------------
    # Inference: three-stage API, prefill -> generate_text -> generate_action
    # ------------------------------------------------------------------

    def process_pixel_values(self, pixel_values):
        if isinstance(pixel_values, dict):
            return pixel_values
        n_img = pixel_values.shape[1]
        if n_img == 1:
            return {"image": pixel_values}
        # Multi-image VLM batch: each image is an independent picture, not a temporal frame.
        # Split into n_img separate n_k=1 cameras so MEM temporal token drop is never triggered.
        return {f"image_{i}": pixel_values[:, i : i + 1] for i in range(n_img)}

    @torch.no_grad()
    def prefill(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> InferenceState:
        """Stage 1: encode text/images + VLM prefill -> KV cache.

        Args:
            samples: List[Dict] of length B in RoboVQA format
            pixel_values: [B, n_img, 3, 224, 224] raw pixel values
        Returns:
            InferenceState containing KV cache, attention mask, position IDs, and seed token
        """
        if isinstance(pixel_values, Dict):
            first_image = next(iter(pixel_values.values()))
            device, dtype = first_image.device, first_image.dtype
        else:
            device, dtype = pixel_values.device, pixel_values.dtype
        pixel_values_processed = self.process_pixel_values(pixel_values)
        input_ids, attention_mask = self.processor.encode_inference(
            samples,
            device=device,
            mode="ar",
            training=self.training,
        )

        # Build proprio batch if proprio_encoder is enabled
        proprio_batch = (
            build_proprio_batch(
                samples,
                device=device,
                dtype=torch.float32,
                zero_values=self.model.proprio_encoder == "zeros",
            )
            if self.model.proprio_embedder is not None
            else None
        )

        vlm_hidden, vlm_kv, position_ids = self.model.vlm_prefill(
            input_ids,
            attention_mask,
            pixel_values_processed,
            dtype=dtype,
            proprio=proprio_batch,
        )
        state = InferenceState(
            kv_cache=vlm_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            last_hidden=vlm_hidden[:, -1, :],
            pixel_values=pixel_values_processed,
            input_ids=input_ids,
            device=device,
        )
        state.check_invariants(where="prefill")
        return state

    @torch.no_grad()
    def generate_text(
        self,
        state: InferenceState,
        *,
        max_new_tokens: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        trim_token_ids: Optional[List[int]] = None,
        return_updated_state: bool = True,
        **ar_kwargs,
    ) -> InferenceState:
        """Stage 2: AR text generation for CoT inference or VQA answers.

        Updates KV cache / attention_mask / seed_ids in place on state, and fills
        generated_texts / generated_ids.

        Args:
            state: InferenceState output by prefill()
            max_new_tokens: maximum generated token count
            stop_token_ids: stop when any of these tokens is generated, e.g. [eov_id] or [eos_id]
            trim_token_ids: token IDs used to truncate before decoding, e.g. [eos_id, pad_id]
            return_updated_state: whether to return updated KV cache, needed by later generate_action
            **ar_kwargs: sampling parameters such as top_k, top_p, temperature
        Returns:
            Updated InferenceState
        """
        state.check_invariants(where="generate_text entry")
        ar_output = self.model.inference_ar(
            state.last_hidden,
            state.attention_mask,
            state.pixel_values,
            past_key_values=state.kv_cache,
            return_kv_cache=return_updated_state,
            stop_token_ids=stop_token_ids,
            max_new_tokens=max_new_tokens,
            **ar_kwargs,
        )
        generated_ids = ar_output["generated_ids"]

        # Grow attention_mask by the newly generated tokens
        new_attn_mask = ar_output.get("attention_mask")
        if new_attn_mask is not None:
            state.attention_mask = new_attn_mask
            state.position_ids = self.model.mask_helper.compute_position_ids(state.attention_mask)

        # Update state using .get(..., default), because ar_helper does not write
        # past_key_values / last_hidden / attention_mask when return_updated_state=False.
        state.kv_cache = ar_output.get("past_key_values", state.kv_cache)
        state.last_hidden = ar_output.get("last_hidden", state.last_hidden)
        state.generated_ids = generated_ids
        state.generated_texts = self.processor.decode_text(generated_ids, trim_token_ids)
        if state.generated_texts:
            logger.info("[CoT] %s", state.generated_texts)
        # Validate only when state was actually updated.
        if new_attn_mask is not None:
            state.check_invariants(where="generate_text exit")
        return state

    @torch.no_grad()
    def generate_action(
        self,
        state: InferenceState,
        samples: List[Dict[str, Any]],
        *,
        action_dim_is_pad: Optional[torch.BoolTensor] = None,
        action_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Stage 3: generate actions, using FM continuous and/or AR discrete paths.

        self.continuous_action / self.discrete_action decide whether to run FM and/or AR.

        Args:
            state: InferenceState, possibly updated by generate_text
            samples: original samples; AR decode needs frequency/embodiment
            action_dim_is_pad: [B, D] bool mask
            action_gt: [B, H, D] GT action for action_evaluator
        """
        state.check_invariants("generate_action entry")
        results: Dict[str, Any] = {}
        timing: Dict[str, float] = {}

        # FM action (continuous)
        if self.continuous_action:
            _sync_if_cuda_available()
            t_fm0 = time.monotonic()
            results["action"] = self.model.inference_fm(
                attention_mask=state.attention_mask,
                pixel_values=state.pixel_values,
                past_key_values=state.kv_cache,
                action_dim_is_pad=action_dim_is_pad,
                position_ids_override=state.position_ids,
                embodiment_types=[s.get("embodiment") for s in samples],
            )
            _sync_if_cuda_available()
            timing["fm_action_ms"] = (time.monotonic() - t_fm0) * 1000.0

        # AR action (discrete)
        if self.discrete_action:
            _sync_if_cuda_available()
            t_ar0 = time.monotonic()
            ar_out = self.model.inference_ar(
                state.last_hidden,
                state.attention_mask,
                state.pixel_values,
                past_key_values=state.kv_cache,
                return_kv_cache=True,
                max_new_tokens=self._get_action_generation_max_new_tokens(),
            )
            _sync_if_cuda_available()
            timing["ar_decode_ms"] = (time.monotonic() - t_ar0) * 1000.0
            # Sync state: in the non-BAR path, past_key_values is appended in place.
            # attention_mask / position_ids must grow together, otherwise the final
            # invariant assertion fails.
            new_am = ar_out.get("attention_mask")
            if new_am is not None:
                state.attention_mask = new_am
                state.position_ids = self.model.mask_helper.compute_position_ids(new_am)
            state.kv_cache = ar_out.get("past_key_values", state.kv_cache)
            state.last_hidden = ar_out.get("last_hidden", state.last_hidden)
            gen_ids = ar_out["generated_ids"]
            # Build full ids (input + CoT + action) so decode_ar can find "Action: "
            prefix_parts = [state.input_ids]
            if state.generated_ids is not None:
                prefix_parts.append(state.generated_ids)
            prefix = torch.cat(prefix_parts, dim=1)
            full_ids = torch.cat([prefix, gen_ids], dim=1)
            gen_list = [full_ids[i] for i in range(full_ids.size(0))]
            # input_parts_meta for WBC codec semantic unpad (cross-embodiment)
            input_parts_meta_list = [
                s.get("action", {}).get("parts_meta", None)
                if isinstance(s.get("action"), dict)
                else None
                for s in samples
            ]
            if all(m is None for m in input_parts_meta_list):
                merger = getattr(getattr(self, "processor", None), "action_state_merger", None)
                fallback_meta = getattr(merger, "max_action_shape_meta", None)
                if fallback_meta is not None:
                    input_parts_meta_list = [fallback_meta] * len(samples)
                else:
                    input_parts_meta_list = None
            _sync_if_cuda_available()
            t_decode0 = time.monotonic()
            decoded_actions, decoded_tokens, _, absent_keys_per_sample = self.processor.decode_ar(
                gen_list,
                horizon_steps=self.model_config.horizon_steps,
                action_dim=self.model_config.action_dim,
                device=state.device,
                action_dim_is_pad=action_dim_is_pad,
                frequencies=[s.get("frequency") for s in samples],
                embodiments=[s.get("embodiment") for s in samples],
                input_parts_meta_list=input_parts_meta_list,
            )
            _sync_if_cuda_available()
            timing["decode_ar_total_ms"] = (time.monotonic() - t_decode0) * 1000.0
            decode_timing = getattr(self.processor, "_last_decode_ar_timing", None)
            if isinstance(decode_timing, dict):
                timing.update(decode_timing)
            results["ar_action"] = torch.stack(decoded_actions, dim=0)
            results["ar_absent_keys"] = absent_keys_per_sample
            results["decoded_action_tokens"] = decoded_tokens
            if "action" not in results:
                results["action"] = results["ar_action"]

            if action_gt is not None:
                results.update(
                    self.action_evaluator(
                        generated_tokens=decoded_tokens,
                        decoded_actions=decoded_actions,
                        samples=samples,
                        action_gt=action_gt,
                        device=state.device,
                        absent_keys_per_sample=absent_keys_per_sample,
                    )
                )

        state.check_invariants("generate_action exit")
        results["_timing"] = timing
        return results

    # ------------------------------------------------------------------
    # High-level inference entry point (composes the three stages)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        """Unified inference forward: prefill -> optional generate_text -> optional generate_action.

        Args:
            samples:      List[Dict] of length B
            pixel_values: [B, n_img, 3, 224, 224]
            optional kwargs:
                actions:           [B, H, D] GT for eval comparison
                action_dim_is_pad: [B, D]
        Returns:
            dict: {"action": [B, H, D], ...}
        """
        action_dim_is_pad = kwargs.pop("action_dim_is_pad", None)
        action_gt = kwargs.pop("actions", None)
        results = {}
        timing: Dict[str, float] = {}
        _sync_if_cuda_available()
        t_total0 = time.monotonic()

        # Stage 1: Prefill
        t0 = time.monotonic()
        state = self.prefill(samples, pixel_values)
        _sync_if_cuda_available()
        timing["prefill_ms"] = (time.monotonic() - t0) * 1000.0

        # Stage 2: optional CoT generation (AR -> EOV stop).
        if self.predict_cot:
            eov_id = self.model.ar_helper.eov_token_id
            t0 = time.monotonic()
            state = self.generate_text(
                state,
                stop_token_ids=[eov_id] if eov_id is not None else None,
                trim_token_ids=self._get_trim_token_ids(include_eov=True),
            )
            _sync_if_cuda_available()
            timing["cot_generate_text_ms"] = (time.monotonic() - t0) * 1000.0
            results["cot_text"] = state.generated_texts
            results["generated_ids"] = state.generated_ids

        # Stage 3: action generation.
        need_action = self.discrete_action or self.continuous_action
        if need_action:
            t0 = time.monotonic()
            action_results = self.generate_action(
                state,
                samples,
                action_dim_is_pad=action_dim_is_pad,
                action_gt=action_gt,
            )
            _sync_if_cuda_available()
            timing["generate_action_total_ms"] = (time.monotonic() - t0) * 1000.0
            action_timing = action_results.pop("_timing", {})
            if isinstance(action_timing, dict):
                timing.update(action_timing)
            results.update(action_results)

        if not results:
            raise ValueError(
                "No action prediction method: check discrete_action/continuous_action config flags"
            )

        _sync_if_cuda_available()
        timing["forward_inference_total_ms"] = (time.monotonic() - t_total0) * 1000.0
        results["_timing"] = timing
        return results

    @torch.no_grad()
    def infer_vqa(
        self,
        prompt: str,
        pixel_values: torch.FloatTensor,
        num_tokens_to_generate: Optional[int] = 512,
        top_k_filtering: Optional[bool] = False,
        top_k: Optional[int] = 128,
        top_p: Optional[float] = 0.95,
        temperature: Optional[float] = 0.7,
        verbose: Optional[bool] = False,
        template: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """VQA inference entry point.

        Args:
            prompt: user question
            pixel_values: [B, n_img, 3, 224, 224] or [n_img, 3, 224, 224]
            num_tokens_to_generate: maximum generated token count
            top_k, top_p, temperature: sampling parameters
            template: optional custom template
        Returns:
            List[str]: generated text answers
        """
        del top_k_filtering, verbose

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 5:
            raise ValueError(
                f"pixel_values must be 4D or 5D, got shape={tuple(pixel_values.shape)}"
            )

        batch_size, num_images = pixel_values.shape[:2]
        template = template or self._build_vqa_template(num_images)
        image_size = tuple(int(x) for x in pixel_values.shape[-2:])
        samples = [
            {
                "template": template,
                "question": prompt,
                "answer": "",
                **{f"image{i}": image_size for i in range(num_images)},
            }
            for _ in range(batch_size)
        ]

        # Stage 1: Prefill
        state = self.prefill(samples, pixel_values)

        # Stage 2: Generate text (stop @ EOS)
        eos_id = getattr(self.model_config, "eos_token_id", None)
        state = self.generate_text(
            state,
            max_new_tokens=num_tokens_to_generate,
            stop_token_ids=[eos_id] if eos_id is not None else None,
            trim_token_ids=self._get_trim_token_ids(),
            return_updated_state=False,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )

        return state.generated_texts

    # ------------------------------------------------------------------
    # Freeze / load
    # ------------------------------------------------------------------

    def freeze_backbones(self, stage: str) -> None:
        if stage in {"full-finetune", "vla-full-train"}:
            pass  # all trainable
        elif stage in {"finetune", "vla-train"}:
            self.model.vision_tower.requires_grad_(False)
        else:
            logger.warning(f"Unknown stage `{stage}`, not freezing")

    def load_from_checkpoint(self, stage, run_dir, pretrained_checkpoint=None):
        raise NotImplementedError("Use from_checkpoint instead.")

    # ------------------------------------------------------------------
    # MFU: FLOPs estimation
    # ------------------------------------------------------------------

    def _per_image_tokens(self, cfg) -> list:
        return [256] * self.num_input_images

    def estimate_training_flops_per_sample(self) -> int:
        """Estimate FLOPs per sample per training step for G05 (fwd + bwd).

        Components: Vision (SiGLIP) + VLM Mixture + Action Expert Mixture.
        Mixture FLOPs delegated to Mixture.estimate_flops().
        """
        cfg = self.model.cfg

        # ---- Vision (SiGLIP, not a Mixture) ----
        num_vision_params = sum(p.numel() for p in self.model.vision_tower.parameters())
        num_proj_params = sum(p.numel() for p in self.model.multi_modal_projector.parameters())
        vis_L = cfg.vision.num_hidden_layers
        vis_H = cfg.vision.num_attention_heads
        vis_d_head = cfg.vision.hidden_size // vis_H
        vision_flops = 0
        for P in self._per_image_tokens(cfg):
            vis_attn = 12 * vis_L * vis_H * vis_d_head * P
            vision_flops += (6 * num_vision_params + vis_attn) * P
            vision_flops += 6 * num_proj_params * P  # projector

        # ---- Mixtures ----
        S_vlm = sum(self._per_image_tokens(cfg)) + cfg.max_text_tokens
        S_action = cfg.horizon_steps
        vlm_flops = self.model.vlm.estimate_flops(query_len=S_vlm)
        ae_flops = self.model.action_expert.estimate_flops(
            query_len=S_action,
            kv_len=S_vlm + S_action,
        )

        return vision_flops + vlm_flops + ae_flops

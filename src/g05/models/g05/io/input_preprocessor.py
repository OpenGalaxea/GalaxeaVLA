# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import logging
import os
import re
import time
import warnings
from copy import deepcopy
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
from g05.utils.data.data_utils import convert_text_to_paligemma
from g05.utils.logging.logging_config import get_logger
from g05.utils.common.special_tokens import SpecialTokenManager

try:
    # Only depends on constants; keep ImportError compatibility so lightweight
    # environments can import this file without tokenizer/scipy dependency chains.
    from g05.data_processor.processor.base_processor import IGNORE_INDEX
except Exception:  # pragma: no cover
    IGNORE_INDEX = -100
from g05.utils.common.pytorch_utils import dict_apply
from g05.tokenizer.utils.misc import left_to_right_align
from g05.tokenizer.token_registry import TokenRegistry
from g05.tokenizer.interface.base_action_tokenizer import DecodeResult
from g05.tokenizer.interface.proprio_encoder import ProprioEncoder
from g05.tokenizer.interface.bar_builder import BARBuilder, make_bar_builder
from g05.utils.hf import resolve_hf_model_path

logger = get_logger(__name__)

PROPRIO_ENCODER_MODES = {"bin", "mlp", "mlp_dropout", "null", "zeros"}


def _is_mapping_like(value: Any) -> bool:
    return hasattr(value, "get") and not isinstance(value, (str, bytes))


def normalize_proprio_encoder_mode(proprio_encoder: Any) -> str:
    if proprio_encoder is None:
        return "null"
    if _is_mapping_like(proprio_encoder):
        mode = str(proprio_encoder.get("mode", "null")).lower()
    else:
        mode = str(proprio_encoder).lower()
    if mode not in PROPRIO_ENCODER_MODES:
        raise ValueError(
            f"Unsupported proprio_encoder={proprio_encoder!r}; "
            f"expected one of {sorted(PROPRIO_ENCODER_MODES)}."
        )
    return mode


def get_proprio_dropout_p(proprio_encoder: Any, default: float = 0.2) -> float:
    if _is_mapping_like(proprio_encoder):
        dropout_p = float(proprio_encoder.get("dropout_p", default))
    else:
        dropout_p = float(default)
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
    return dropout_p


# Defines Action Encoding Schemes
class TOKEN_INDEX(IntEnum):
    # fmt: off
    PADDING_TOKEN_INDEX = 0
    IMAGE_TOKEN_INDEX = 1
    PROPRIO_TOKEN_INDEX = 2
    ACTION_TOKEN_INDEX = 3
    TEXT_TOKEN_INDEX = 4
    COT_TOKEN_INDEX = 5
    PRED_TEXT_TOKEN_INDEX=6
    # fmt: on


@dataclass
class TemplateSegment:
    """Template segment descriptor supporting static text and dynamic placeholders."""

    type: str  # "static" | "dynamic"
    content: str
    sample_key: str = ""
    processor_key: str = ""
    control_key: str = ""  # control tags such as EOC / EOV
    masked: bool = False
    max_tokens: Optional[int] = None  # max token limit; None means unlimited


class BaseModalityProcessor:
    """Base class for modality processors, standardizing process calls."""

    def __call__(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        """
        Unified entry point:
        - data: raw data for the current modality, such as text, tensor, or structured object
        - is_masked: whether it is conditioning input; True should set labels to IGNORE_INDEX
        - training: train/inference flag, useful for processor-side augmentation
        - tokenizer: text tokenizer used to convert strings into token ids
        - sample_dict: full current sample dict for processors needing extra fields
        """
        return self.process(
            data=data,
            is_masked=is_masked,
            training=training,
            tokenizer=tokenizer,
            sample_dict=sample_dict,
        )

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        """
        Core processing logic implemented by subclasses:
        - data: data for the current modality
        - is_masked: whether to mask labels for this segment
        - training: whether this is training mode
        - tokenizer: shared tokenizer
        - sample_dict: full current sample dict for processors needing extra fields
        Returns the input_ids, labels, attention_mask triplet.
        """
        raise NotImplementedError("Subclasses must implement the process method")


class InputPreprocessor:
    """
    Data input preprocessor for tokenizing text, images, proprio, and actions.

    Template placeholder syntax:
        <sampleKey_processorKey>          — unmasked, unlimited length
        <sampleKey_processorKey_!>        — masked with no loss, unlimited length
        <sampleKey_processorKey_!_N>      — masked, up to N tokens with truncation
        <sampleKey_processorKey_N>        — unmasked, up to N tokens

    Examples:
        <command_text_!_200>   — command field handled by text processor, masked, max 200 tokens
        <cot_text_300>         — cot field handled by text processor, unmasked, max 300 tokens
        <action_action>        — action field handled by action processor, unlimited length
    """

    def __init__(
        self,
        hf_processor_path: str,
        hf_processor_class: str,
        action_tokenizer_class: str,
        at_config,
        model=None,
        base_vocab_size: int = 257153,
        padded_vocab_size: int = 257216,
        pad_token_id: int = None,
        model_cfg=None,
        image_token_index: int = None,
        num_image_tokens: int = None,
        input_action_corruption: bool = False,
        control_tokens: Optional[Dict[str, str]] = None,
        batchify_action: bool = True,
        modality_processors: Optional[Dict[str, Any]] = None,
        pred_eov: bool = False,
        pi05_ft_mode: bool = False,
        proprio_encoder: str = "bin",
        model_type: str = "paligemma",
        add_loc_tokens: bool = False,
    ):
        """
        Initialize the input preprocessor.

        Centralized token registration: creates the HF processor + action_tokenizer,
        registers all tokens (action/BAR/EOV/<state>), and triggers model embedding
        resizing.

        Args:
            hf_processor_path: pretrained path for the HF processor
            hf_processor_class: HF processor class name string
            action_tokenizer_class: action tokenizer class name string
            at_config: action tokenizer config (vq_config)
            model: optional G05Model reference for resize_embedding; skipped when absent
            base_vocab_size: original vocab size before EXTRA_TOKENS
            padded_vocab_size: pretrained padded vocab size
            pad_token_id: padding token id
            image_token_index: image token index
            num_image_tokens: token count per image
            input_action_corruption: whether to enable action input corruption for augmentation
            control_tokens: control tag mapping, default {"eoc": "EOC", "eov": "EOV"}
            proprio_encoder: "bin" | "mlp" | "mlp_dropout" | "zeros" | "null";
                mlp_dropout also supports {"mode": "mlp_dropout", "dropout_p": 0.2}
            model_type: "paligemma" | "qwen35", used by SpecialTokenManager
        """
        self.registry = TokenRegistry(
            hf_processor_path, hf_processor_class, add_loc_tokens=add_loc_tokens
        )
        self.hf_processor_path = hf_processor_path
        self.hf_processor = self.registry.hf_processor
        self.tokenizer = self.registry.tokenizer

        from g05.utils.common.import_utils import get_obj_from_str

        tok_cls = get_obj_from_str(action_tokenizer_class)
        pre_register_vocab_size = self.registry.vocab_size
        self.action_tokenizer = tok_cls(
            vq_config=at_config,
            vocab_offset=pre_register_vocab_size,
            hf_vocab_size=pre_register_vocab_size,
        )

        self.registry.register(self.action_tokenizer.new_action_tokens)

        if self.action_tokenizer.block_wise_autoregressive:
            self.action_tokenizer.set_bar_ids(
                bos_blk_id=self.registry.get_id("<bos_blk>"),
                eos_blk_id=self.registry.get_id("<eos_blk>"),
                pad_action_id=self.registry.get_id("<pad_action_token>"),
            )

        eov_tok = "<EOV>"
        self.registry.register([eov_tok])
        self.eov_token_id = self.registry.get_id(eov_tok)

        self.token_manager = SpecialTokenManager.for_model(
            model_type=model_type,
            tokenizer=self.tokenizer,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
        )
        self.proprio_encoder_mode = normalize_proprio_encoder_mode(proprio_encoder)
        self.proprio_dropout_p = get_proprio_dropout_p(proprio_encoder)
        self.state_token_id: Optional[int] = None
        if self.proprio_encoder_mode in {"mlp", "mlp_dropout", "zeros"}:
            state_tok = "<state>"
            self.registry.register([state_tok])
            self.state_token_id = self.registry.get_id(state_tok)

        need_resize = self.action_tokenizer.use_extra_tokens or self.proprio_encoder_mode in {
            "mlp",
            "mlp_dropout",
            "zeros",
        }
        if need_resize:
            self.registry.resize_model(
                model,
                base_vocab_size=base_vocab_size,
                pad_token_id=pad_token_id,
                new_token_names=self.registry.new_tokens or None,
            )

        n_bins = int(at_config.get("n_bins", 256)) if hasattr(at_config, "get") else 256
        min_action = float(at_config.get("min_action", -1)) if hasattr(at_config, "get") else -1.0
        max_action = float(at_config.get("max_action", 1)) if hasattr(at_config, "get") else 1.0
        self.proprio_encoder = ProprioEncoder(
            n_bins=n_bins,
            min_action=min_action,
            max_action=max_action,
        )

        self.bar_builder = make_bar_builder(self.action_tokenizer, self.tokenizer)

        self.pad_token_id = pad_token_id
        self.model_cfg = model_cfg
        self.input_action_corruption = input_action_corruption
        self.batchify_action = batchify_action
        self.control_tokens = control_tokens or {"eoc": "EOC", "eov": "EOV"}
        self.pred_eov = pred_eov
        if model_cfg is not None:
            self.image_token_index = (
                model_cfg.image_token_index if image_token_index is None else image_token_index
            )
            self.num_image_tokens = (
                model_cfg.vision.num_image_tokens if num_image_tokens is None else num_image_tokens
            )
        else:
            self.image_token_index = image_token_index
            self.num_image_tokens = num_image_tokens
        self.model_type = self.token_manager.model_type
        self.modality_processors = {}
        self._register_builtin_processors()
        if modality_processors:
            self.modality_processors.update(modality_processors)
        self.default_template = "<image_image_!> Task:<command_text_!>;STATE:<proprio_proprio_!>:\nAction: <action_action>"
        logger.info(
            f"[InputPreprocessor] Initialized with modality processors: {list(self.modality_processors.keys())}"
        )

        self.pi05_ft_mode = pi05_ft_mode

    def get_added_token_id_map(self) -> Dict[str, int]:
        """Return token string -> current HF token id for tokens registered by this processor."""
        token_to_id: Dict[str, int] = {}
        for token in self.registry.new_tokens:
            token_id = self.registry.get_id(token)
            if token_id is not None:
                token_to_id[token] = int(token_id)
        return token_to_id

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

    def preprocess(
        self,
        samples: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        loss_on_static_text: bool = False,
        control_flag: ["return_prefix", "return_suffix", "context_only"] = None,
        right_align: bool = False,
        training: bool = False,
        device=None,
        template: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor], torch.Tensor]:
        """
        Preprocess input data.

        Args:
            training: whether this is training mode; affects action corruption and alignment.
            template: default template string or list aligned with samples, containing
                placeholders such as <image> or <action_!>. Sample-level `template`
                takes priority; missing templates fall back to the default with warning.
            samples: single sample dict or list of sample dicts, required in template
                mode. If the default template is a list, its length must match samples.
            loss_on_static_text: whether static text contributes to loss.
            control_flag: control tag split mode (None/"return_prefix"/"return_suffix"/"context_only"):
                - None: only remove control tag segments; EOC/EOV do not enter tokenizer
                - "context_only": keep only segments before <EOC>, excluding <EOC>
                - "return_suffix": keep only segments after <EOV>, excluding <EOV>
                - "return_prefix": keep only segments before <EOV>, and replace <EOV>
                  with a prediction segment handled by the text processor, with literal
                  content "<EOV>"
            right_align: whether to right-align after batch padding (left padding),
                usually enabled when returning prefix/context.
            **kwargs: accepted for compatibility with extra arguments.

        Returns:
            tuple: processed (input_ids, labels, attention_mask)
        """
        if samples is None:
            raise ValueError("Template mode is the only supported entry point; provide samples.")

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Template mode requires `self.tokenizer` for text encoding.")

        valid_control_flags = (None, "return_prefix", "return_suffix", "context_only")
        if control_flag not in valid_control_flags:
            raise ValueError(f"control_flag must be one of {valid_control_flags}, got {control_flag!r}")

        # Shallow-copy each sample dict to avoid mutating caller's data
        # (batchify_action path replaces sample["action"] with tokenized string)
        sample_list = [dict(s) for s in (samples if isinstance(samples, list) else [samples])]
        segments_per_sample = self._prepare_segments(
            sample_list, template, control_flag=control_flag
        )

        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        has_action_flags: List[bool] = []

        # Support batchify: if samples require batched action processing, discretize
        # actions as a whole first.
        pretokenized_actions: Optional[Dict[int, Any]] = None
        all_noop_indices: set = set()
        if self.batchify_action and self.action_tokenizer is not None:
            action_list = []
            action_sample_indices = []
            action_parts_meta_list = []
            for i, s in enumerate(sample_list):
                if "action" in s:
                    action_entry = dict(s["action"])
                    action_parts_meta_list.append(action_entry.pop("parts_meta", None))
                    action_list.append(action_entry)
                    action_sample_indices.append(i)
            if len(action_list):
                action_dict = default_collate(action_list)
                action_dict = dict_apply(
                    action_dict, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
                )
                encode_kwargs = {
                    "action_op_mask": action_dict["action_op_mask"],
                    "action_dim_is_pad": action_dict["action_dim_is_pad"],
                }

                frequencies = [sample_list[i].get("frequency") for i in action_sample_indices]
                if frequencies and all(freq is not None for freq in frequencies):
                    encode_kwargs["frequency"] = torch.tensor(
                        frequencies,
                        dtype=torch.float32,
                        device=device,
                    )

                embodiments = [sample_list[i].get("embodiment") for i in action_sample_indices]
                if embodiments and all(emb is not None for emb in embodiments):
                    encode_kwargs["embodiment"] = embodiments

                if action_parts_meta_list and all(pm is not None for pm in action_parts_meta_list):
                    first_parts_meta = action_parts_meta_list[0]
                    if all(pm == first_parts_meta for pm in action_parts_meta_list[1:]):
                        encode_kwargs["input_parts_meta"] = first_parts_meta
                    else:
                        encode_kwargs["input_parts_meta"] = action_parts_meta_list

                pretokenized_actions = self.action_tokenizer(
                    action_dict["value"],
                    encode_kwargs=encode_kwargs,
                )
                # Always pass token ids to BuiltinActionProcessor to avoid
                # text round-trip issues (BPE re-segmentation, block boundary
                # splits) in both use_extra_tokens modes.
                # batch_decode is only needed for non-BAR text-only logging.
            for i, s in enumerate(sample_list):
                if "action" in s:
                    s["action"] = pretokenized_actions[i]
                    if (
                        pretokenized_actions is not None
                        and isinstance(pretokenized_actions[i], list)
                        and len(pretokenized_actions[i]) == 0
                    ):
                        all_noop_indices.add(i)

        # ========================
        # Main loop over samples, processing each sample independently.
        # ========================
        for idx, (sample, segs) in enumerate(zip(sample_list, segments_per_sample)):
            _input_ids, _labels, _attn, _has_action = self._preprocess_single_sample(
                segments=segs,
                sample_dict=sample,
                tokenizer=tokenizer,
                loss_on_static_text=loss_on_static_text,
                training=training,
            )
            tensor_input_ids = torch.tensor(_input_ids, dtype=torch.long)
            tensor_labels = torch.tensor(_labels, dtype=torch.long)
            tensor_attn = torch.tensor(_attn, dtype=torch.float32)

            if device is None:
                device = tensor_input_ids.device

            batch_input_ids.append(tensor_input_ids.to(device))
            batch_labels.append(tensor_labels.to(device))
            batch_attention_mask.append(tensor_attn.to(device))
            has_action_flags.append(_has_action)

        # Mask all labels for all-noop samples (all parts dropped by dropout_noop_parts)
        for idx in all_noop_indices:
            if idx < len(batch_labels):
                batch_labels[idx] = torch.full_like(batch_labels[idx], IGNORE_INDEX)

        # Padding.
        padded_input_ids = pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        ).to(torch.long)
        padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX).to(
            torch.long
        )
        padded_attention_mask = pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=TOKEN_INDEX.PADDING_TOKEN_INDEX
        )

        # Note: context_only / return_prefix / return_suffix are mutually exclusive.
        if control_flag in ("return_prefix", "context_only"):
            right_align = True
        elif control_flag == "return_suffix":
            right_align = False

        if (
            control_flag == "return_prefix" or control_flag == "context_only"
        ) and self.pi05_ft_mode:
            right_align = False
            # Compute prefix target length on-the-fly: num_input_images × tokens_per_img + max_text_tokens.
            # PaliGemma only (pi05_ft_mode is always False for Qwen35).
            cfg = self.model_cfg
            tokens_per_img = (cfg.vision.image_size // cfg.vision.patch_size) ** 2
            target_len = cfg.num_input_images * tokens_per_img + cfg.max_text_tokens
            current_len = padded_input_ids.shape[1]
            if current_len > target_len:
                padded_input_ids = padded_input_ids[:, :target_len]
                padded_attention_mask = padded_attention_mask[:, :target_len]
                padded_labels = padded_labels[:, :target_len]
            elif current_len < target_len:
                pad_len = target_len - current_len
                batch_size = padded_input_ids.shape[0]
                padded_input_ids = torch.cat(
                    [
                        padded_input_ids,
                        torch.full(
                            (batch_size, pad_len),
                            self.pad_token_id,
                            dtype=padded_input_ids.dtype,
                            device=padded_input_ids.device,
                        ),
                    ],
                    dim=1,
                )
                padded_attention_mask = torch.cat(
                    [
                        padded_attention_mask,
                        torch.full(
                            (batch_size, pad_len),
                            TOKEN_INDEX.PADDING_TOKEN_INDEX,
                            dtype=padded_attention_mask.dtype,
                            device=padded_attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                padded_labels = torch.cat(
                    [
                        padded_labels,
                        torch.full(
                            (batch_size, pad_len),
                            IGNORE_INDEX,
                            dtype=padded_labels.dtype,
                            device=padded_labels.device,
                        ),
                    ],
                    dim=1,
                )

        if right_align:
            padded_input_ids, padded_attention_mask, padded_labels = left_to_right_align(
                padded_input_ids, padded_attention_mask, padded_labels
            )
        return padded_input_ids, padded_labels, padded_attention_mask

    def _preprocess_single_sample(
        self,
        segments: List[TemplateSegment],
        sample_dict: Dict[str, Any],
        tokenizer,
        loss_on_static_text: bool,
        training: bool,
    ) -> Tuple[List[int], List[int], List[int], bool]:
        result_input_ids: List[int] = []
        result_labels: List[int] = []
        result_attention_mask: List[int] = []
        has_action = False

        for seg in segments:
            if seg.type == "static":
                text_ids = tokenizer(seg.content, add_special_tokens=False)["input_ids"]
                result_input_ids.extend(text_ids)
                result_attention_mask.extend([TOKEN_INDEX.TEXT_TOKEN_INDEX] * len(text_ids))
                result_labels.extend(
                    text_ids if loss_on_static_text else [IGNORE_INDEX] * len(text_ids)
                )
                continue

            # Allow pure-text dynamic segments without sample_key for control tag
            # replacement and similar cases:
            # TemplateSegment(type="dynamic", processor_key="text", sample_key="", content="...")
            processor_key = seg.processor_key
            if (
                seg.type == "dynamic"
                and processor_key == "text"
                and (seg.sample_key == "" or seg.sample_key is None)
            ):
                value = seg.content
            else:
                sample_key = seg.sample_key
                if sample_key not in sample_dict:
                    raise KeyError(
                        f"Template placeholder <{sample_key}_{processor_key}> is missing from the sample."
                    )
                value = sample_dict[sample_key]

            processor: BaseModalityProcessor = self._get_processor(processor_key)

            ids, lbs, attn = processor.process(
                value,
                is_masked=seg.masked,
                training=training,
                tokenizer=tokenizer,
                sample_dict=sample_dict,
            )
            ids, lbs, attn = self._normalize_processor_output(ids, lbs, attn, seg.masked)
            # Truncate overlong output.
            if seg.max_tokens is not None and len(ids) > seg.max_tokens:
                logger.warning(
                    f"[InputPreprocessor] <{seg.sample_key}_{seg.processor_key}> emitted {len(ids)} tokens, "
                    f"exceeding max_tokens={seg.max_tokens}; truncating to {seg.max_tokens} tokens."
                )
                ids = ids[: seg.max_tokens]
                lbs = lbs[: seg.max_tokens]
                attn = attn[: seg.max_tokens]
            result_input_ids.extend(ids)
            result_labels.extend(lbs)
            result_attention_mask.extend(attn)
            if processor_key == "action":
                has_action = True

        # Ensure there is at least one token.
        if len(result_input_ids) == 0:
            return [self.pad_token_id], [IGNORE_INDEX], [0], has_action

        assert len(result_input_ids) == len(result_attention_mask), (
            f"result_input_ids.shape ({len(result_input_ids)}) should be equal to result_attention_mask.shape ({len(result_attention_mask)})"
        )
        return result_input_ids, result_labels, result_attention_mask, has_action

    def _normalize_processor_output(
        self,
        input_ids: Union[List[int], torch.Tensor],
        labels: Optional[Union[List[int], torch.Tensor]],
        attention_mask: Union[List[int], torch.Tensor],
        is_masked: bool,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Normalize custom processor output to lists and fill missing labels by mask."""

        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.tolist()
            return list(x)

        ids = to_list(input_ids)
        attn = to_list(attention_mask)

        if labels is None:
            labels = ids if not is_masked else [IGNORE_INDEX] * len(ids)
        labels = to_list(labels)

        if not (len(ids) == len(labels) == len(attn)):
            raise ValueError("processor output input_ids/labels/attention_mask lengths differ")

        return ids, labels, attn

    def register_modality(self, key: str, processor: Any):
        """Dynamically register a modality processor implementing process(...)->(ids, labels, attn)."""
        self.modality_processors[key] = processor
        logger.info(
            f"[InputPreprocessor] registered modality processor <{key}>; "
            f"current processors: {list(self.modality_processors.keys())}"
        )

    def _register_builtin_processors(self):
        """Register built-in default processors for image/proprio/state/action and text."""
        processors = {}
        # Register image processor by model_type. The key is always "image", so
        # templates do not need to know the implementation.
        if self.model_type == "qwen35":
            processors["image"] = BuiltinQwen35ImageProcessor(
                patch_size=self.model_cfg.vision.patch_size,
                spatial_merge_size=self.model_cfg.vision.spatial_merge_size,
                pretrained_model_path=self.hf_processor_path,
            )
        else:
            processors["image"] = BuiltinImageProcessor(
                patch_size=self.model_cfg.vision.patch_size,
                image_size=self.model_cfg.vision.image_size,
                image_token_index=self.image_token_index,
            )
        if self.proprio_encoder_mode in {"mlp", "mlp_dropout", "zeros"}:
            if self.state_token_id is None:
                raise ValueError(
                    f"proprio_encoder={self.proprio_encoder_mode!r} but state_token_id is not set. "
                    "Verify that the __init__ <state> token registration path has run."
                )
            if self.proprio_encoder_mode == "mlp_dropout":
                proprio_proc: BaseModalityProcessor = BuiltinProprioMLPDropoutProcessor(
                    state_token_id=self.state_token_id,
                    dropout_p=self.proprio_dropout_p,
                )
            else:
                proprio_proc = BuiltinProprioMLPProcessor(
                    state_token_id=self.state_token_id,
                )
        elif self.proprio_encoder_mode == "null":
            proprio_proc = BuiltinProprioNullProcessor()
        else:
            proprio_proc = BuiltinProprioProcessor(self.proprio_encoder, self.tokenizer)
        processors.update(
            {
                "proprio": proprio_proc,
                "action": BuiltinActionProcessor(
                    bar_builder=self.bar_builder,
                    input_action_corruption=self.input_action_corruption,
                ),
                "text": BuiltinTextProcessor(),
                # Affordance CoT: attention_mask is marked as COT_TOKEN_INDEX (5).
                "affordance": BuiltinAffordanceProcessor(),
            }
        )
        self.modality_processors.update(processors)
        logger.info(
            f"[InputPreprocessor] Built-in modality processors registered: {list(self.modality_processors.keys())}"
        )

    def _get_processor(self, key: str) -> BaseModalityProcessor:
        """Get the processor; missing keys raise instead of falling back to text."""
        if key in self.modality_processors:
            return self.modality_processors[key]
        raise KeyError(f"No processor found for <{key}>; register one or use default text.")

    def _parse_template(self, template: str) -> List[TemplateSegment]:
        """Split a template into static text and dynamic placeholder segments.

        Dynamic placeholder format is <sampleKey_processorKey>[_!]; control tags
        support <EOC>/<EOV>.

        Note: <...> tags containing |, such as Qwen3.5 <|im_start|>, are not parsed
        as placeholders and remain static text.
        """
        template = self.token_manager.resolve_template(template)
        pattern = re.compile(r"<([^<>|]+)>")
        segments: List[TemplateSegment] = []
        last_idx = 0

        for match in pattern.finditer(template):
            if match.start() > last_idx:
                static_text = template[last_idx : match.start()]
                if static_text:
                    segments.append(TemplateSegment(type="static", content=static_text))

            raw_key = match.group(1).strip()

            # Step 1: handle control tags.
            if raw_key in {self.control_tokens.get("eoc"), self.control_tokens.get("eov")}:
                segments.append(
                    TemplateSegment(type="control", content=raw_key, control_key=raw_key)
                )
                last_idx = match.end()
                continue

            # Step 2: handle placeholders.
            # Extract trailing max_tokens limit first, formatted as _digits, e.g. _200.
            max_tokens = None
            _max_tokens_match = re.match(r"^(.+)_(\d+)$", raw_key)
            if _max_tokens_match:
                raw_key = _max_tokens_match.group(1)
                max_tokens = int(_max_tokens_match.group(2))

            masked = raw_key.endswith("_!")
            key = raw_key[:-2] if masked else raw_key
            if "_" not in key:
                # Step 2.1: if placeholder format does not match, try special tokens.
                # Priority: SpecialTokenManager → tokenizer vocab lookup
                special_tok = self.token_manager.resolve(raw_key)
                if special_tok is None:
                    special_tok = _resolve_special_token(
                        self.tokenizer,
                        raw_key,
                    )
                if special_tok is not None:
                    # If `<raw_key>` is a registered special token or a tokenizer-known
                    # single token, keep it as static text.
                    segments.append(TemplateSegment(type="static", content=special_tok))
                    last_idx = match.end()
                    continue

                eoc = self.control_tokens.get("eoc")
                eov = self.control_tokens.get("eov")
                raise ValueError(
                    f"Template segment `<{raw_key}>` cannot be parsed: it is neither a control tag "
                    f"`<{eoc}>`/`<{eov}>`, nor a tokenizer-learnable single token such as `<eos>`, "
                    f"and it does not match the placeholder format. Use `<sampleKey_processorKey>`, "
                    f"`<sampleKey_processorKey_!>`, or `<sampleKey_processorKey_!_maxTokens>`.\n"
                    f"Examples: `<command_text_!>`, `<command_text_!_200>`, `<cot_text>`, "
                    f"`<action_action>`, `<image_image_!>`.\n"
                    f"Current raw_key={raw_key!r}. Placeholders must contain an underscore "
                    f"separating sampleKey and processorKey."
                )
            sample_key, processor_key = key.rsplit("_", 1)
            segments.append(
                TemplateSegment(
                    type="dynamic",
                    content="",
                    sample_key=sample_key,
                    processor_key=processor_key,
                    masked=masked,
                    max_tokens=max_tokens,
                )
            )
            last_idx = match.end()

        if last_idx < len(template):
            tail_text = template[last_idx:]
            if tail_text:
                segments.append(TemplateSegment(type="static", content=tail_text))

        return segments

    def _parse_control_tokens(
        self,
        segments: List[TemplateSegment],
        control_flag: ["return_prefix", "return_suffix", "context_only"] = None,
    ) -> List[TemplateSegment]:
        """
        Control tag slicing/replacement rules:
        - return_prefix: return segments before <EOV>, replacing the <EOV> position
          with a prediction segment handled by the text processor whose literal
          content is "<EOV>", allowing the model to predict that token directly.
        - return_suffix: return only segments after <EOV>, excluding <EOV>.
        - context_only: return only segments before <EOC>, excluding <EOC>. If that
          prefix contains <EOV>, replace it with a learnable literal "<EOV>" segment
          to avoid stripping it.
        Control tags themselves do not produce tokens; they are removed regardless of
        slicing, or replaced under return_prefix/context_only.
        """
        eoc = self.control_tokens.get("eoc")
        eov = self.control_tokens.get("eov")

        def _make_eov_pred_seg() -> TemplateSegment:
            # Let the model directly predict the literal "<EOV>" token. It is
            # learnable and has no sample_key.
            return TemplateSegment(
                type="dynamic",
                content=f"<{eov}>",
                sample_key="",
                processor_key="text",
                masked=not self.pred_eov,
            )

        def _strip_controls(
            segs: List[TemplateSegment], *, keep_eov: bool = False
        ) -> List[TemplateSegment]:
            out: List[TemplateSegment] = []
            for s in segs:
                if s.type != "control":
                    out.append(s)
                    continue
                if keep_eov and s.control_key == eov:
                    out.append(_make_eov_pred_seg())
            return out

        # Find the first EOC/EOV positions.
        # Whether static segments contribute to loss is controlled by loss_on_static_text.
        # To make only post-EOC static text learnable, convert static_text after EOC into
        # dynamic(text, masked=False) segments so labels are always produced.
        # Control segments are not otherwise handled here, including EOV replacement;
        # that is delegated to the later control_flag branches.
        eoc_pos = None
        eov_pos = None
        converted_segments: List[TemplateSegment] = []
        after_eoc = False
        for i, seg in enumerate(segments):
            if seg.type == "control":
                if seg.control_key == eoc:
                    if eoc_pos is not None:
                        raise ValueError(
                            f"Template contains multiple <{eoc}> tags; only one EOC is supported. "
                            f"Please check the template: {''.join(s.content for s in segments)}"
                        )
                    eoc_pos = i
                    after_eoc = True
                elif seg.control_key == eov:
                    eov_pos = i
                converted_segments.append(seg)
                continue
            if after_eoc and seg.type == "static" and seg.content:
                converted_segments.append(
                    TemplateSegment(
                        type="dynamic",
                        content=seg.content,
                        sample_key="",
                        processor_key="text",
                        masked=False,
                    )
                )
            else:
                converted_segments.append(seg)

        # Later logic uniformly uses converted segments; control segment positions stay unchanged.
        segments = converted_segments

        if control_flag is None:
            # Default: remove control tags to avoid downstream processing errors.
            return _strip_controls(segments, keep_eov=True)

        if control_flag == "context_only":
            if eoc_pos is None:
                return _strip_controls(segments, keep_eov=True)
            # context_only returns segments before EOC; if they contain EOV, keep it as a token.
            return _strip_controls(segments[:eoc_pos], keep_eov=True)

        # eov_pos + 1 keeps eov in the prefix.
        if control_flag == "return_suffix":
            if eov_pos is None:
                warnings.warn(
                    "No EOV found in the segments, returning nothing, if is in VLM training you can ignore this warning.",
                    UserWarning,
                )
                return []  # return nothing
            return _strip_controls(segments[eov_pos + 1 :])

        if control_flag == "return_prefix":
            if eov_pos is None:
                return _strip_controls(segments)
            prefix = _strip_controls(segments[: eov_pos + 1], keep_eov=True)
            return prefix

        raise ValueError(
            f"Unknown control_flag={control_flag!r}; only context_only/return_prefix/return_suffix are supported."
        )

    def _prepare_segments(
        self,
        sample_list: List[Dict[str, Any]],
        template: Optional[str] = None,
        control_flag: ["return_prefix", "return_suffix", "context_only"] = None,
    ) -> List[List[TemplateSegment]]:
        """Use the provided template first, falling back to defaults with warnings."""
        per_sample_segments: List[List[TemplateSegment]] = []
        for idx, sample in enumerate(sample_list):
            tpl = template or sample.get("template", None)
            if tpl is None:
                tpl = self.default_template
                warnings.warn(f"Sample {idx} did not provide template; falling back to the default template.", UserWarning)
            _segments = self._parse_template(tpl)
            _segments = self._parse_control_tokens(_segments, control_flag)
            per_sample_segments.append(_segments)
        return per_sample_segments

    # ==================================================================
    # High-level API: encode_train / encode_inference / decode_ar
    # ==================================================================

    def encode_train(
        self,
        samples: List[Dict[str, Any]],
        device: torch.device,
        training: bool = True,
        max_chunk_token_length: int = 2048,
        max_pad_token_length: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, int]:
        """Training encode: prefix + suffix -> concat -> truncate -> pad.

        Args:
            samples: List[Dict], length B
            device: target device
            training: whether this is training mode
            max_chunk_token_length: maximum sequence length, truncating overflow
            max_pad_token_length: fixed FSDP length, or None for no padding

        Returns:
            (input_ids [B,S], labels [B,S], attention_mask [B,S], split_index)
        """
        prefix_ids, prefix_labels, prefix_mask = self.preprocess(
            samples=deepcopy(samples),
            control_flag="return_prefix",
            device=device,
            training=training,
            **kwargs,
        )
        suffix_ids, suffix_labels, suffix_mask = self.preprocess(
            samples=deepcopy(samples),
            control_flag="return_suffix",
            device=device,
            training=training,
            **kwargs,
        )
        split_index = prefix_ids.shape[1]

        input_ids = torch.cat([prefix_ids, suffix_ids], dim=-1)
        labels = torch.cat([prefix_labels, suffix_labels], dim=-1)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=-1)

        # Truncate
        input_ids, labels, attention_mask, split_index = self._truncate(
            input_ids,
            labels,
            attention_mask,
            split_index,
            max_chunk_token_length,
        )
        # FSDP pad
        if max_pad_token_length is not None:
            input_ids, labels, attention_mask, split_index = self._fsdp_pad(
                input_ids,
                labels,
                attention_mask,
                split_index,
                max_pad_token_length,
            )
        return input_ids, labels, attention_mask, split_index

    def encode_inference(
        self,
        samples: List[Dict[str, Any]],
        device: torch.device,
        mode: str = "fm",
        training: bool = False,
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Inference encode.

        Args:
            samples: List[Dict], length B
            device: target device
            mode: "fm" → return_prefix, "ar" → context_only
            training: whether this is training mode

        Returns:
            (input_ids [B, S], attention_mask [B, S])
        """
        if mode == "fm":
            control_flag = "return_prefix"
        elif mode == "ar":
            control_flag = "context_only"
        else:
            raise ValueError(f"encode_inference mode must be 'fm' or 'ar', got {mode!r}")

        input_ids, _, attention_mask = self.preprocess(
            samples=deepcopy(samples),
            control_flag=control_flag,
            device=device,
            training=training,
            **kwargs,
        )
        return input_ids, attention_mask

    def decode_text(
        self,
        generated_ids: torch.Tensor,
        trim_token_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """Decode generated token IDs to text.

        Truncate at the first token in trim_token_ids, then tokenizer.decode.
        Used for VQA answer decoding and CoT text extraction.

        Args:
            generated_ids: [B, gen_len] raw token IDs
            trim_token_ids: truncation token list, e.g. [eos_id, pad_id, eov_id]
        Returns:
            List[str] of length B
        """
        tokenizer = self.tokenizer
        outputs = []
        for i in range(generated_ids.size(0)):
            ids = generated_ids[i]
            if trim_token_ids:
                for tid in trim_token_ids:
                    if tid is None:
                        continue
                    pos = (ids == tid).nonzero(as_tuple=False)
                    if pos.numel() > 0:
                        ids = ids[: int(pos[0].item())]
            outputs.append(tokenizer.decode(ids) if ids.numel() > 0 else "")
        return outputs

    def decode_ar(
        self,
        generated_ids: List[Optional[torch.Tensor]],
        horizon_steps: int,
        action_dim: int,
        device: torch.device = None,
        action_dim_is_pad: Optional[torch.BoolTensor] = None,
        frequencies: Optional[List[Any]] = None,
        embodiments: Optional[List[Any]] = None,
        input_parts_meta_list: Optional[List[Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[set[str]]]:
        """AR backward: generated token ids -> continuous actions.

        Args:
            generated_ids: List of [gen_len] per sample (None = decode failed)
            horizon_steps: action chunk length H
            action_dim: action dimension D
            device: target device
            action_dim_is_pad: [B, D] optional
            input_parts_meta_list: Per-sample input_parts_meta for semantic unpad

        Returns:
            (decoded_actions, decoded_action_tokens, decoded_cot_texts, absent_keys_per_sample)
            decoded_actions: List of [H, D] tensors
            decoded_action_tokens: List of [N] tensors (raw action token ids)
            decoded_cot_texts: List of str, extracted CoT text per sample
            absent_keys_per_sample: List of set[str], keys absent from each sample's AR decode
        """
        tokenizer = self.tokenizer
        timing = {
            "decode_ar_string_decode_ms": 0.0,
            "action_text_tokenizer_encode_ms": 0.0,
            "action_tokenizer_encode_ms": 0.0,
            "action_tokenizer_decode_ms": 0.0,
        }
        t_decode_ar0 = time.monotonic()

        # Phase 1: parse — split each sample into (cot_text, action_str | None)
        cot_texts: List[str] = []
        action_strs: List[Optional[str]] = []
        for i, ids in enumerate(generated_ids):
            if ids is None:
                cot_texts.append("")
                action_strs.append(None)
                continue
            t0 = time.monotonic()
            decoded_str = tokenizer.decode(ids)
            timing["decode_ar_string_decode_ms"] += (time.monotonic() - t0) * 1000.0
            parts = decoded_str.split("Action: ", 1)
            if len(parts) < 2:
                logger.warning(
                    f"[decode_ar] sample {i}: no 'Action: ' found in generated string, "
                    f"returning zeros. Preview: {decoded_str[:200]}"
                )
                cot_texts.append(self._clean_special_tokens(decoded_str).strip())
                action_strs.append(None)
                continue
            cot_texts.append(self._clean_special_tokens(parts[0]).strip())
            action_str = self._extract_action_str(decoded_str)
            if not hasattr(self, "_action_log_count"):
                self._action_log_count = 0
            if self._action_log_count < 5:
                logger.info(f"Decoded action [{self._action_log_count}]: {action_str}")
                self._action_log_count += 1
            action_strs.append(action_str)

        # Phase 2: decode action tokens → continuous actions
        decoded_actions: List[torch.Tensor] = []
        decoded_tokens: List[torch.Tensor] = []
        absent_keys_per_sample: List[set[str]] = []
        for i, action_str in enumerate(action_strs):
            if action_str is None:
                decoded_actions.append(torch.zeros(horizon_steps, action_dim))
                decoded_tokens.append(torch.zeros(1, dtype=torch.long))
                absent_keys_per_sample.append(set())
                continue
            if len(action_str.strip()) == 0:
                logger.warning(f"[decode_ar] sample {i}: empty action string, returning zeros")
                decoded_actions.append(torch.zeros(horizon_steps, action_dim))
                decoded_tokens.append(torch.zeros(1, dtype=torch.long))
                absent_keys_per_sample.append(set())
                continue

            _device = device or generated_ids[i].device
            t0 = time.monotonic()
            raw_tokens = torch.tensor(
                tokenizer.encode(action_str, add_special_tokens=False),
                device=_device,
            )
            timing["action_text_tokenizer_encode_ms"] += (time.monotonic() - t0) * 1000.0
            decode_kwargs = {}
            if action_dim_is_pad is not None:
                decode_kwargs["action_dim_is_pad"] = action_dim_is_pad[i : i + 1]
            if frequencies is not None and i < len(frequencies) and frequencies[i] is not None:
                decode_kwargs["frequency"] = torch.tensor(
                    [frequencies[i]],
                    dtype=torch.float32,
                    device=_device,
                )
            if embodiments is not None and i < len(embodiments) and embodiments[i] is not None:
                decode_kwargs["embodiment"] = [embodiments[i]]
            if (
                input_parts_meta_list is not None
                and i < len(input_parts_meta_list)
                and input_parts_meta_list[i] is not None
            ):
                decode_kwargs["input_parts_meta"] = input_parts_meta_list[i]

            t0 = time.monotonic()
            result = self.action_tokenizer.decode_token_ids_to_actions(
                raw_tokens,
                time_horizon=horizon_steps,
                action_dim=action_dim,
                decode_kwargs=decode_kwargs or None,
            )
            timing["action_tokenizer_decode_ms"] += (time.monotonic() - t0) * 1000.0
            if isinstance(result, DecodeResult):
                action, absent = result.action, result.absent_keys
            else:
                action, absent = result, set()
            decoded_actions.append(action.cpu())
            decoded_tokens.append(raw_tokens)
            absent_keys_per_sample.append(absent)

        timing["decode_ar_processor_total_ms"] = (time.monotonic() - t_decode_ar0) * 1000.0
        self._last_decode_ar_timing = timing
        return decoded_actions, decoded_tokens, cot_texts, absent_keys_per_sample

    # ------------------------------------------------------------------
    # Private helpers for encode_train
    # ------------------------------------------------------------------

    def _clean_special_tokens(self, decoded_str: str) -> str:
        """Remove block-wise AR markers and control tokens from decoded string."""
        if self.bar_builder.block_wise:
            BOB = self.tokenizer.decode([self.bar_builder.bos_blk_id])
            EOB = self.tokenizer.decode([self.bar_builder.eos_blk_id])
            decoded_str = decoded_str.replace(BOB, "").replace(EOB, "")
            if self.bar_builder.pad_action_id is not None:
                PAD = self.tokenizer.decode([self.bar_builder.pad_action_id])
                decoded_str = decoded_str.replace(PAD, "")
        # Delegate to SpecialTokenManager — strips bos/eos/pad/image.
        decoded_str = self.token_manager.clean(decoded_str)
        # Structured control tokens are not in the SpecialTokenManager registry
        # because from_ids only records tokenizer bos/eos/pad/image. Scan them
        # explicitly here. Without this, AR decoded strings may contain <EOV>/<EOC>,
        # polluting the "Action: " slice in _extract_action_str and action re-encode.
        for tok in ("<EOV>", "<EOC>"):
            decoded_str = decoded_str.replace(tok, "")
        return decoded_str

    def _known_action_token_strings(self) -> set[str]:
        action_tokenizer = getattr(self, "action_tokenizer", None)
        if action_tokenizer is None:
            return set()

        known: set[str] = set()
        for attr in ("action_tokens", "new_action_tokens"):
            value = getattr(action_tokenizer, attr, None)
            if value is None:
                continue
            if callable(value):
                value = value()
            known.update(str(tok) for tok in value)
        return known

    def _extract_delimiterless_action_prefix(self, after_action: str) -> str:
        """Return a leading run of action/group tokens when the static suffix is absent."""
        known_tokens = self._known_action_token_strings()
        fallback_re = re.compile(r"<(?:action\d{4}|[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)>")

        pos = 0
        tokens: List[str] = []
        while pos < len(after_action):
            if after_action[pos].isspace():
                pos += 1
                continue
            match = re.match(r"<[^<>|]+>", after_action[pos:])
            if match is None:
                break
            tok = match.group(0)
            if known_tokens:
                if tok not in known_tokens:
                    break
            elif fallback_re.fullmatch(tok) is None:
                break
            tokens.append(tok)
            pos += len(tok)
        return "".join(tokens).strip()

    def _extract_action_str(self, decoded_str: str) -> str:
        """Extract the action string from AR generated text, between Action: and |."""
        decoded_str = self._clean_special_tokens(decoded_str)
        after_action = decoded_str.split("Action: ")[1]
        if "|" in after_action:
            return after_action.split("|")[0].strip()

        action_prefix = self._extract_delimiterless_action_prefix(after_action)
        if action_prefix:
            logger.debug(
                "[_extract_action_str] '|' delimiter not found after 'Action: '; "
                "using delimiterless action-token prefix."
            )
            return action_prefix

        logger.warning(
            f"[_extract_action_str] '|' delimiter not found after 'Action: ', "
            f"output may be truncated: {after_action[:200]}"
        )
        return after_action.strip()

    @staticmethod
    def _truncate(
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.Tensor,
        split_index: int,
        max_len: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, int]:
        """Truncate to max_len from the prefix tail to preserve the full suffix."""
        total_len = input_ids.shape[1]
        if total_len <= max_len:
            return input_ids, labels, attention_mask, split_index

        suffix_len = total_len - split_index
        new_prefix_len = max(0, max_len - suffix_len)
        # Keep prefix[:new_prefix_len] + full suffix.
        keep_idx = torch.cat(
            [
                torch.arange(0, new_prefix_len, device=input_ids.device),
                torch.arange(split_index, total_len, device=input_ids.device),
            ]
        )
        return (
            input_ids[:, keep_idx],
            labels[:, keep_idx],
            attention_mask[:, keep_idx],
            new_prefix_len,
        )

    def _fsdp_pad(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.Tensor,
        split_index: int,
        target_len: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, int]:
        """Pad or truncate to target_len as required by FSDP fixed length."""
        cur = input_ids.shape[1]
        if cur > target_len:
            return (
                input_ids[:, :target_len],
                labels[:, :target_len],
                attention_mask[:, :target_len],
                min(split_index, target_len),
            )
        if cur == target_len:
            return input_ids, labels, attention_mask, split_index
        # cur < target_len: pad
        pad_len = target_len - cur
        bs = input_ids.shape[0]
        dev = input_ids.device
        input_ids = torch.cat(
            [
                input_ids,
                torch.full((bs, pad_len), self.pad_token_id, dtype=input_ids.dtype, device=dev),
            ],
            dim=1,
        )
        labels = torch.cat(
            [
                labels,
                torch.full((bs, pad_len), IGNORE_INDEX, dtype=labels.dtype, device=dev),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.full(
                    (bs, pad_len),
                    TOKEN_INDEX.PADDING_TOKEN_INDEX,
                    dtype=attention_mask.dtype,
                    device=dev,
                ),
            ],
            dim=1,
        )
        return input_ids, labels, attention_mask, split_index


class BuiltinTextProcessor(BaseModalityProcessor):
    """Default text processor: encode strings into text tokens."""

    def __init__(self, convert_loc_to_paligemma: bool = False, bbox_seq: str = "xyxy"):
        self.convert_loc_to_paligemma = convert_loc_to_paligemma
        self.bbox_seq = bbox_seq

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        """
        Use PRED_TEXT_TOKEN_INDEX for tokens to predict, otherwise TEXT_TOKEN_INDEX.
        """
        text_value = data if isinstance(data, str) else str(data)
        if self.convert_loc_to_paligemma:
            text_value = convert_text_to_paligemma(text_value, seq=self.bbox_seq)

        ids = tokenizer(text_value, add_special_tokens=False)["input_ids"]
        if is_masked:
            attn = [TOKEN_INDEX.TEXT_TOKEN_INDEX] * len(ids)
        else:
            attn = [TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX] * len(ids)
        labels = None if is_masked else ids
        return ids, labels, attn


class BuiltinImageProcessor(BaseModalityProcessor):
    def __init__(self, patch_size: int, image_size: int, image_token_index: int):
        self.patch_size = patch_size
        self.image_size = image_size
        self.image_token_index = image_token_index
        self.num_image_tokens = (image_size // patch_size) ** 2

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        num_image_tokens_total = self.num_image_tokens
        ids = [self.image_token_index] * num_image_tokens_total
        attn = [TOKEN_INDEX.IMAGE_TOKEN_INDEX] * num_image_tokens_total
        labels = [IGNORE_INDEX] * num_image_tokens_total
        return ids, labels, attn


class BuiltinQwen35ImageProcessor(BaseModalityProcessor):
    """Qwen3.5 image processor: <|vision_start|> + <|image_pad|>*N + <|vision_end|>."""

    def __init__(
        self,
        patch_size: int,
        spatial_merge_size: int,
        pretrained_model_path: str,
    ):
        import json as _json

        pretrained_model_path = resolve_hf_model_path(
            pretrained_model_path,
            allow_patterns=["config.json"],
        )
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path) as f:
            config = _json.load(f)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token_index = config["image_token_id"]
        self.boi_token_index = config["vision_start_token_id"]
        self.eoi_token_index = config["vision_end_token_id"]

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        # data must be (H, W), set by samples_builder from shape_meta image sizes.
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            raise ValueError(
                f"BuiltinQwen35ImageProcessor expects data=(H, W) from samples_builder, "
                f"got {data!r}. Ensure GalaxeaCoTProcessor has shape_meta with resolved "
                f"image shapes (either explicit 'shape' or 'camera_type' + camera_size_config)."
            )
        H, W = int(data[0]), int(data[1])
        n = (H // self.patch_size // self.spatial_merge_size) * (
            W // self.patch_size // self.spatial_merge_size
        )
        ids = [self.boi_token_index] + [self.image_token_index] * n + [self.eoi_token_index]
        attn = (
            [TOKEN_INDEX.TEXT_TOKEN_INDEX]
            + [TOKEN_INDEX.IMAGE_TOKEN_INDEX] * n
            + [TOKEN_INDEX.TEXT_TOKEN_INDEX]
        )
        labels = [IGNORE_INDEX] * (n + 2)
        return ids, labels, attn


class BuiltinProprioMLPProcessor(BaseModalityProcessor):
    """Placeholder processor for the MLP proprio encoder.

    Emits only N special `<state>` tokens as placeholders, where N=num_obs_steps is
    inferred dynamically from the leading dimension of proprio. The raw proprio
    tensor is not handled here; `G05Model._forward_embed()` scatters
    `proprio_embedder(proprio)` into `<state>` positions, mirroring the image token
    replacement mechanism.

    Labels are always IGNORE_INDEX because state does not participate in CE loss and
    is only conditioning input.
    """

    def __init__(self, state_token_id: int):
        if state_token_id is None:
            raise ValueError("state_token_id cannot be None.")
        self.state_token_id = int(state_token_id)

    def _num_state_tokens(self, data: Any) -> int:
        """Infer num_obs_steps from proprio data, i.e. the number of state tokens."""
        value = data["value"] if isinstance(data, dict) else data
        value_tensor = torch.as_tensor(value)
        return 1 if value_tensor.ndim <= 1 else int(value_tensor.shape[0])

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        n = self._num_state_tokens(data)
        ids = [self.state_token_id] * n
        attn = [TOKEN_INDEX.PROPRIO_TOKEN_INDEX] * n
        labels = [IGNORE_INDEX] * n
        return ids, labels, attn


class BuiltinProprioMLPDropoutProcessor(BuiltinProprioMLPProcessor):
    """MLP proprio processor with sample-level null dropout."""

    def __init__(self, state_token_id: int, dropout_p: float = 0.2):
        super().__init__(state_token_id)
        self.dropout_p = float(dropout_p)
        if not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        if self.dropout_p > 0.0 and torch.rand(()) < self.dropout_p:
            return [], [], []
        return super().process(data, is_masked, training, tokenizer, sample_dict)


class BuiltinProprioNullProcessor(BaseModalityProcessor):
    """Drop proprio placeholders from the token stream."""

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        return [], [], []


class BuiltinProprioProcessor(BaseModalityProcessor):
    def __init__(self, proprio_encoder, tokenizer):
        self.proprio_encoder = proprio_encoder
        self.tokenizer = tokenizer

    @torch._dynamo.disable
    def _tokenize_proprio(self, proprio: Any) -> List[int]:
        if self.proprio_encoder is None:
            raise ValueError("proprio_encoder is required to process proprio.")

        tokenized_proprio = self.proprio_encoder.encode(proprio)
        proprio_tokens = self.tokenizer(tokenized_proprio, add_special_tokens=False)["input_ids"]
        return proprio_tokens

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        proprio_tokens = self._tokenize_proprio(data)
        attn = [TOKEN_INDEX.PROPRIO_TOKEN_INDEX] * len(proprio_tokens)
        labels = [IGNORE_INDEX] * len(proprio_tokens) if is_masked else proprio_tokens
        return proprio_tokens, labels, attn


class BuiltinActionProcessor(BaseModalityProcessor):
    def __init__(self, bar_builder, input_action_corruption: bool):
        self.bar_builder = bar_builder
        self.input_action_corruption = input_action_corruption
        if self.bar_builder is None:
            raise ValueError("bar_builder is required to process action.")
        if self.input_action_corruption:
            import warnings

            warnings.warn(
                "input_action_corruption=True is set but the corruption logic has been "
                "removed in the TokenRegistry refactor. This parameter is now a no-op. "
                "Please remove it from your config.",
                DeprecationWarning,
                stacklevel=2,
            )

    def process(
        self,
        data,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        if isinstance(data, list):
            action_tokens, attn, labels = self.bar_builder.build_io_from_ids(data)
        elif isinstance(data, str):
            action_tokens, attn, labels = self.bar_builder.build_io(data)
        else:
            raise AssertionError(
                f"[BuiltinActionProcessor] expected data to be a pre-tokenized action "
                f"string or token id list, got type={type(data).__name__}. \n"
                f"Common cause: the template contains an <action_action> placeholder but "
                f"batchify_action=False, so sample['action'] was not replaced with a "
                f"tokenized string. \n"
                f"Fix: (a) set batchify_action=True, or (b) use a samples_builder without "
                f"an <action_action> placeholder, such as SubtaskCoTBuilderFMOnly."
            )

        return action_tokens, labels, attn


class BuiltinAffordanceProcessor(BaseModalityProcessor):
    """Affordance CoT processor: format pixel coordinates as text with COT_TOKEN_INDEX."""

    def _build_affordance_cot(self, data: Dict[str, Any]) -> str:
        def _parse_paligemma_loc_token(uv: torch.Tensor) -> str:
            if torch.any(uv < 0):
                return "Not visible"
            uv_mapped = torch.round(uv * 1024).to(torch.int32)
            uv_mapped = torch.clamp(uv_mapped, min=0, max=1023)
            return f"(<loc{uv_mapped[0]:04d}>, <loc{uv_mapped[1]:04d}>)"

        assert data["operating_hand"] in ["left", "right", "none", "left,right"], (
            f"operating_hand {data['operating_hand']} is not correct"
        )
        operating_hand = data["operating_hand"]
        if operating_hand == "none":
            return ""

        left_arm_keys = ["left_wrist_uv_left_projection", "head_uv_left_projection"]
        right_arm_keys = ["right_wrist_uv_right_projection", "head_uv_right_projection"]
        both_arm_keys = left_arm_keys + right_arm_keys
        keys = (
            both_arm_keys
            if operating_hand == "left,right"
            else (left_arm_keys if operating_hand == "left" else right_arm_keys)
        )

        cot = "Affordance: "
        for key in keys:
            camera_view = key.split("_")[0]
            loc_token = _parse_paligemma_loc_token(data[key])
            cot += f"{camera_view} camera: {loc_token}; "

        return cot

    def process(
        self,
        data: Any,
        is_masked: bool,
        training: bool,
        tokenizer,
        sample_dict: Dict[str, Any],
    ):
        # Support two input forms:
        # - dict: build Affordance CoT text from structured fields
        # - str/others: treat directly as text
        if isinstance(data, dict):
            text_value = self._build_affordance_cot(data)
        else:
            text_value = data if isinstance(data, str) else str(data)

        ids = tokenizer(text_value, add_special_tokens=False)["input_ids"]
        attn = [TOKEN_INDEX.COT_TOKEN_INDEX] * len(ids)
        labels = None if is_masked else ids
        return ids, labels, attn


def _resolve_special_token(tokenizer, raw_key: str) -> Optional[str]:
    tok_str = f"<{raw_key}>"

    # Option A: FastTokenizer-style tokenizers, such as HuggingFace.
    token_id = tokenizer.convert_tokens_to_ids(tok_str)

    # Check whether this hits a valid ID that is not UNK.
    unk_id = tokenizer.unk_token_id
    if token_id is not None and token_id != unk_id:
        # Some tokenizers return unk_id or a fixed value such as 0 or 1 when missing,
        # so ensure this ID truly corresponds to the desired string.
        return tok_str

    return None

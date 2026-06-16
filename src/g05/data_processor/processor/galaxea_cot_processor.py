# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from typing import Dict, Any, Optional, List

from .base_processor import FullProcessor
from .samples_builder import BaseSamplesBuilder
from copy import deepcopy
from g05.utils.data.normalizer import NormMode
from g05.data_processor.transforms.action_filter import BaseActionFilter

import logging

logger = logging.getLogger()


class GalaxeaCoTProcessor(FullProcessor):
    """
    Processor for VLM training with Chain-of-Thought (CoT) support.

    Responsibility split:
    - GalaxeaCoTProcessor: tensor processing (images, action/state normalization)
    - SamplesBuilder: RoboVQA samples dict + composable template construction
    - Model-side InputPreprocessor: tokenization

    Extends FullProcessor with:
    - Two-phase preprocess: _process_tensors() → SamplesBuilder.build()
    - VLM input action normalization
    - Hardcode mode for instruction/proprio/action override
    """

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_state_transforms: Optional[List[Any]],
        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        action_state_merger,
        action_filter: BaseActionFilter,
        # image transform
        train_transforms: Optional[Dict[str, List[Any]]],
        val_transforms: Optional[Dict[str, List[Any]]],
        # instruction transform
        drop_high_level_prob: float,
        use_zh_instruction: bool,
        pad_token_id: int,
        image_token_index: int,
        max_text_tokens: int,
        num_input_cameras: int,
        # camera size config: {camera_type: [H, W]}; used to auto-resolve shape from camera_type
        camera_size_config: Optional[Dict[str, List[int]]] = None,
        # accepted for config compat, not used (tokenization is done by the model's InputPreprocessor)
        tokenizer_params: Optional[Dict[str, Any]] = None,
        # use_cot (for backward compatibility with config, not used)
        use_cot: bool = False,
        # cot_steps (deprecated, ignored — use samples_builder._target_ instead)
        cot_steps: Optional[List[str]] = None,
        # optional: per-key norm override (None = use norm_default_mode for all keys)
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # use discrete action
        discrete_action: bool = True,
        # vlm input action normalization (independent from training action norm)
        vlm_input_action_norm_default_mode: Optional[NormMode] = None,
        vlm_input_action_norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        # hardcode mode (for backward compatibility with GalaxeaCoTProcessorHardcode)
        hardcode_instruction: Optional[str] = None,
        hardcode_proprio_pad_zeros: bool = False,
        hardcode_action_pad_ones: bool = False,
        # embodiment_type: can be set here via config, or dynamically by MixtureLerobotDataset.set_processor()
        embodiment_type: Optional[str] = None,
        # dummy mode clip range (applied when norm_mode is "dummy")
        dummy_clip_default: tuple = (-5.0, 5.0),
        dummy_clip_exception: Optional[Dict[str, Dict[str, tuple]]] = None,
        # SamplesBuilder: config selects the concrete builder class via _target_.
        # Default is BaseSamplesBuilder (no CoT); can switch to SubtaskCoTBuilder, etc.
        samples_builder: Optional[Any] = None,
        # image resize override: patches Resize in transforms; backward-compatible (None = no-op)
        image_resize: Optional[List[int]] = None,
    ):
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.discrete_action = discrete_action
        self._embodiment_type = embodiment_type

        if self._embodiment_type is not None:
            logger.debug(f"[GalaxeaCoTProcessor] Embodiment type: {self._embodiment_type}")
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps,
            num_output_cameras=num_output_cameras,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            action_filter=action_filter,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
            max_text_tokens=max_text_tokens,
            num_input_cameras=num_input_cameras,
            camera_size_config=camera_size_config,
            vlm_input_action_norm_default_mode=vlm_input_action_norm_default_mode,
            vlm_input_action_norm_exception_mode=vlm_input_action_norm_exception_mode,
            dummy_clip_default=dummy_clip_default,
            dummy_clip_exception=dummy_clip_exception,
        )

        if self.discrete_action:
            from g05.data_processor.transforms.action_state_merger import ConcatLeftAlign

            if isinstance(self.action_state_merger, ConcatLeftAlign):
                raise ValueError(
                    "ConcatLeftAlign is incompatible with discrete_action=True. "
                    "Use PaddingActionMerger instead — discrete tokenizers require "
                    "per-part aligned padding, not flat concatenation."
                )

        # Hardcode mode
        self.hardcode_instruction = hardcode_instruction

        image_sizes = {
            meta["key"]: (meta["shape"][1], meta["shape"][2])
            for meta in (self.shape_meta.get("images") or [])
        }
        builder_cls = samples_builder or BaseSamplesBuilder
        self.samples_builder = builder_cls(
            num_input_images=self.num_input_images,
            image_sizes=image_sizes,
            embodiment_type=embodiment_type,
            hardcode_proprio_pad_zeros=hardcode_proprio_pad_zeros,
            hardcode_action_pad_ones=hardcode_action_pad_ones,
        )

    # ------------------------------------------------------------------ #
    #  train / eval — propagate to samples_builder                       #
    # ------------------------------------------------------------------ #

    def train(self):
        super().train()
        if hasattr(self.samples_builder, "set_training"):
            self.samples_builder.set_training(True)

    def eval(self):
        super().eval()
        if hasattr(self.samples_builder, "set_training"):
            self.samples_builder.set_training(False)

    # ------------------------------------------------------------------ #
    #  embodiment_type property — syncs to samples_builder               #
    # ------------------------------------------------------------------ #

    @property
    def embodiment_type(self) -> Optional[str]:
        return self._embodiment_type

    @embodiment_type.setter
    def embodiment_type(self, value: Optional[str]):
        self._embodiment_type = value
        if hasattr(self, "samples_builder"):
            self.samples_builder.embodiment_type = value

    # ------------------------------------------------------------------ #
    #  Instruction override                                               #
    # ------------------------------------------------------------------ #

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """Override to use hardcoded instruction if configured."""
        if self.hardcode_instruction is not None:
            return self.hardcode_instruction
        return data["task"]

    # ------------------------------------------------------------------ #
    #  Phase 1: Tensor processing                                         #
    # ------------------------------------------------------------------ #

    def _process_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data into tensors without building RoboVQA samples.

        Returns:
            Sample dict containing processed tensors and two temporary fields:
            - _instructions: instruction text consumed by SamplesBuilder
            - _vlm_action: VLM action consumed by SamplesBuilder, possibly None
        """
        sample = {}

        # Normalize all bilingual text fields in "Chinese@English" format according
        # to use_zh_instruction.
        for key in ("task", "atomic_task", "future_task", "high_level_instruction", "action_hint"):
            if key in data and isinstance(data[key], str) and "@" in data[key]:
                zh, eng = data[key].split("@", 1)
                data[key] = zh if self.use_zh_instruction else eng

        # 1. instruction
        instructions = self.augment_instruction(data)
        sample["_instructions"] = instructions

        # 2. image
        pixel_values = self.process_images(data)
        sample["pixel_values"] = self.build_pixel_values(pixel_values)

        # Copy action before transform for open-loop evaluation
        if "action" in data:
            sample["gt_action"] = self.action_state_merger.forward(deepcopy(data))["action"]

        # 3. action & state
        data = self.preprocess_action_state(data)

        # VLM input action: independently normalized action copy used to fill
        # template text. See the detailed comment in base_processor.py
        # preprocess_action_state. If None, SamplesBuilder falls back to the
        # training action.
        sample["_vlm_action"] = data.pop("vlm_action", None)

        sample = self.build_action_sample(data, sample)
        sample["idx"] = data["idx"]
        for meta_key in ("task", "embodiment", "dataset_locator"):
            if meta_key in data:
                sample[meta_key] = data[meta_key]

        return sample

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Two-phase pipeline:
        1. _process_tensors(): action/state normalization, image transform
        2. SamplesBuilder.build(): RoboVQA samples dict + composable template construction

        Args:
            data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index

        Returns:
            sample: Dict[str, Any], which can be collated:
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action": Optional, deepcopy of input action for open loop eval
                - "action_op_mask": Optional, torch.BoolTensor -> [action_dim]
                - "samples": RoboVQA format dict for multimodal processing
                - "idx": int, sample index
        """
        # Phase 1: tensor processing
        sample = self._process_tensors(data)

        # Phase 2: RoboVQA samples construction (delegated to SamplesBuilder)
        sample["samples"] = self.samples_builder.build(data, sample)

        return sample

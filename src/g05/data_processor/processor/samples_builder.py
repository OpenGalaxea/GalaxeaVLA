# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""RoboVQA Samples Builder hierarchy, with one responsibility per class.

Each template construction path is an independent class selected by _target_ in config.

Existing builders:
- BaseSamplesBuilder:              minimal VLA (no CoT)
- SubtaskCoTBuilder:               subtask CoT
- BBoxCoTBuilder:                  bbox CoT target
- MemorySamplesBuilder:            memory input

Adding a builder takes only 3 steps:

    1. Inherit BaseSamplesBuilder and implement the template property as a complete string:

        class MyBuilder(BaseSamplesBuilder):
            @property
            def template(self):
                return (
                    f"{self._images}<bos>Embodiment: <embodiment_text_!>; "
                    f"Task: <command_text_!_200> MyField: <myfield_text> State: <proprio_proprio_!>;\\n"
                    f"Action: <EOV><EOC><action_action>|<eos>"
                )

    2. Implement _populate_extra_samples(data, samples), extracting data into samples:

            def _populate_extra_samples(self, data, samples):
                samples["myfield"] = data["my_raw_field"]

    3. Select it in config via _target_:

        processor:
          samples_builder:
            _target_: g05.data_processor.processor.samples_builder.MyBuilder

    build() automatically fills command/proprio/action/embodiment/images from template placeholders.
    You only need to handle the fields introduced by your builder.

Smoke test: python -m g05.data_processor.processor.samples_builder
"""

from typing import Dict, Any, List, Optional
import logging

import torch

logger = logging.getLogger(__name__)


class BaseSamplesBuilder:
    """Minimal VLA training template with no CoT and no extra segments.

    Template (embodiment=r1, 2 cameras, discrete):
        PaliGemma:
            <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
            Action: <EOV><EOC><action_action>|<eos>
        Qwen3.5-instruct:
            <|im_start|>user\\n<image0_image_!><image1_image_!>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;<|im_end|>\\n<|im_start|>robot\\n
            Action: <EOV><EOC><action_action>|<|endoftext|>

    <bos>/<eos>/<chat_user_prefix>/<chat_user_suffix>/<chat_assistant_prefix>
    are replaced by InputPreprocessor._parse_template -> SpecialTokenMap.resolve()
    according to the model type.
    """

    # Subclasses declare required data fields; can_handle() checks they exist and are non-empty.
    required_fields: tuple = ()
    # Fields actually required at inference time (model inputs before EOC).
    # Eval uses this field set for matching.
    eval_required_fields: tuple = ()

    def __init__(
        self,
        num_input_images: int,
        image_sizes: Dict[str, Any],
        embodiment_type: Optional[str] = None,
        hardcode_proprio_pad_zeros: bool = False,
        hardcode_action_pad_ones: bool = False,
        hardcode_instruction: Optional[str] = None,
    ):
        assert len(image_sizes) > 0, "image_sizes must be non-empty."
        # {camera_key: (H, W)} — target size per camera after resize, from shape_meta.
        # For historical frames (num_obs_steps > 1): same camera key maps to same size
        # across all obs steps; index wraps via i % num_cameras.
        self._image_sizes: Dict[str, Any] = dict(image_sizes)
        self._image_keys: List[str] = list(image_sizes.keys())
        self.num_input_images = num_input_images
        self._embodiment_type = embodiment_type
        self.hardcode_proprio_pad_zeros = hardcode_proprio_pad_zeros
        self.hardcode_action_pad_ones = hardcode_action_pad_ones
        self.hardcode_instruction = hardcode_instruction

        self._images = "".join(f"<image{i}_image_!>" for i in range(num_input_images))

    @property
    def embodiment_type(self) -> Optional[str]:
        return self._embodiment_type

    @embodiment_type.setter
    def embodiment_type(self, value: Optional[str]) -> None:
        self._embodiment_type = value

    # String-level sentinels: null placeholders from the tasks table, pandas NaN
    # values serialized as strings, etc.
    _INVALID_STRINGS: frozenset = frozenset({"", "null", "none", "nan", "-1"})

    def _check_fields(self, data: Dict[str, Any], fields: tuple) -> bool:
        import math

        for k in fields:
            v = data.get(k)
            if v is None:
                return False
            if isinstance(v, float) and math.isnan(v):
                return False
            if isinstance(v, int) and v == -1:
                return False
            if isinstance(v, str) and v.strip().lower() in self._INVALID_STRINGS:
                return False
        return True

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Whether this sample has all annotation fields required by the builder."""
        return self._check_fields(data, self.required_fields)

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        """Eval path: check only fields truly needed at inference time.

        CoT target fields after EOC are generated by the model during AR inference,
        so they are not checked. This allows datasets without CoT annotations to
        still match the correct eval builder.
        """
        return self._check_fields(data, self.eval_required_fields)

    @property
    def template(self) -> str:
        """Complete template. Subclasses override this property.

        <bos>/<eos>/<chat_user_prefix>/<chat_user_suffix>/<chat_assistant_prefix>
        are replaced by SpecialTokenMap.resolve() before parsing.
        PaliGemma: <bos>=<bos>, <eos>=<eos>, <chat_user_suffix>=\\n, the rest are empty.
        Qwen3.5-instruct: <bos>="", <eos>=<|endoftext|>, chat tokens inject the dialog format.
        """
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        """Subclass override: extract extra data into samples."""
        pass

    def _override_command(self, data: Dict[str, Any]) -> Optional[str]:
        """Subclass override: return non-None to replace the command slot text.

        The command slot is the injected value for <command_text_!_200>.
        Default returns None, preserving sample['_instructions'] (the task).
        """
        return None

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """Common build flow; subclasses do not need to override it."""
        instructions = sample.pop("_instructions")
        if self.hardcode_instruction is not None:
            instructions = self.hardcode_instruction
        cmd_override = self._override_command(data)
        if cmd_override is not None:
            instructions = cmd_override
        # VLM input action: independently normalized copy, preferred for
        # template <action_action>. Falls back to the training action when None.
        vlm_action = sample.pop("_vlm_action")

        samples: Dict[str, Any] = {}
        tpl = self.template

        # Let subclasses populate extra data.
        self._populate_extra_samples(data, samples)

        samples["template"] = tpl
        samples["command"] = instructions

        proprio_pad_mask = (
            torch.zeros_like(sample["proprio_dim_is_pad"])
            if self.hardcode_proprio_pad_zeros
            else sample["proprio_dim_is_pad"]
        )
        samples["proprio"] = {
            "value": sample["proprio"],
            "proprio_dim_is_pad": proprio_pad_mask,
        }

        samples["embodiment"] = (
            self.embodiment_type if self.embodiment_type is not None else "unknown"
        )

        if "<action_action" in tpl and "action" in sample:
            action_pad_mask = (
                torch.ones_like(sample["action_dim_is_pad"])
                if self.hardcode_action_pad_ones
                else sample["action_dim_is_pad"]
            )
            samples["action"] = {
                "value": vlm_action if vlm_action is not None else sample["action"],
                "action_dim_is_pad": action_pad_mask,
                "action_op_mask": sample.get("action_op_mask"),
                "parts_meta": sample.get("action_parts_meta"),
            }

        if "frequency" in data:
            samples["frequency"] = data["frequency"]

        num_cameras = len(self._image_keys)
        for i in range(self.num_input_images):
            if f"<image{i}_image" in tpl:
                cam_key = self._image_keys[i % num_cameras]
                samples[f"image{i}"] = self._image_sizes[cam_key]

        return samples


class BaseActionSamplesBuilderFMOnly(BaseSamplesBuilder):
    """Base Action Samples Builder FM-only: minimal VLA template with no CoT or extras.

    Template:
        <image0_image_!><image1_image_!><bos>Embodiment: ...; Task: ... State: <proprio>;
        Action: <EOV><EOC><eos>
    """

    @property
    def template(self) -> str:
        return (
            f"{self._images}<bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;\n"
            f"Action: <EOV><EOC><eos>"
        )


class MemorySamplesBuilder(BaseSamplesBuilder):
    """VLA + memory input (masked, no loss).

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;
        Action: <EOV><EOC><action_action>|<eos>
    """

    required_fields = ("memory",)
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        samples["memory"] = data.get("memory", "")


class SubtaskCoTBuilder(BaseSamplesBuilder):
    """Output a task text CoT segment (Subtask: ...) before action.

    CoT text source: data["atomic_task"], decoded from atomic_task_index
    (r1lite/r1pro _merged_final_v30). atomic_task is strictly required; datasets
    without this annotation return can_handle=False.

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!><EOC><atomic_task_text>|
        Action: <EOV><action_action>|<eos>

    Warning: this template requires the action segment to be discretized correctly
       (batchify_action=True, or an action_tokenizer that can process raw action
       dicts). If using FM-only continuous actions without tokenizing action, use
       SubtaskCoTBuilderFMOnly instead.
    """

    required_fields = ("atomic_task",)
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        samples["prompt"] = "predict subtask"


class TaskAsSubtaskCoTBuilder(SubtaskCoTBuilder):
    """Use the current frame's task text as the CoT target instead of atomic_task.

    Intended for foldbench-style hardcode_instruction configs where command is fixed
    to a high-level description, while data["task"] preserves per-frame fine-grained
    labels and can be used directly as the CoT target. required_fields=("task",)
    makes any dataset match, but this should only be used on data with fine-grained
    task annotations.
    """

    required_fields = ("task",)
    eval_required_fields = ()

    def _populate_extra_samples(self, data, samples):
        samples["atomic_task"] = f"Subtask: {data.get('task', '')}"
        samples["prompt"] = "predict subtask"


class SubtaskCoTBuilderFMOnly(SubtaskCoTBuilder):
    """Subtask CoT (FM-only variant): AR predicts only subtask text; action uses FM.

    Compared with SubtaskCoTBuilder, this template removes the `<action_action>` placeholder:
        <image0_image_!><image1_image_!><bos>Embodiment: ...; Task: ... State: <proprio>;
        <prompt_text_!><EOC><atomic_task_text>|
        Action: <EOV><eos>

    Why this variant exists:
      fmonly training usually sets `batchify_action=false`, so action bypasses the
      tokenizer. The parent `SubtaskCoTBuilder` template still contains an
      `<action_action>` placeholder, which triggers BuiltinActionProcessor.process(raw_dict).
      When the action template serializes `{action}` directly, Python `str()` turns
      the raw dict into a literal string such as
      `{'value': tensor([[...]]), 'action_dim_is_pad': ...}`; that string is then
      sliced into the AR label. The model is therefore trained to predict Python
      tensor repr text, so eval-time CoT may emit long tensor-like strings such as
      "[ 1.1111, -0.6111, ...". This looks undertrained but is actually the wrong
      target.

      This class removes the action placeholder, so the AR head receives loss only
      on subtask text. The erroneous training signal is removed at the source while
      the FM path continues to train on continuous actions normally.
    """

    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|"
            "Action: <EOV><eos>"
        )


class FutureSubtaskCoTBuilder(SubtaskCoTBuilder):
    """Use the task from a future N-th frame as the CoT target.

    The dataset pre-decodes it as data["future_task"]. Unlike SubtaskCoTBuilder,
    the target is a future-frame subtask description instead of the current frame,
    giving the model look-ahead planning ability.

    When the target would pass the end of an episode, the dataset clamps to the
    last frame (copying it), matching action chunk behavior. This is treated as a
    valid training signal.

    The dataset must register delta_timestamps["task_index"] (already present in
    GalaxeaLerobotDataset) and decode future_task in __getitem__
    (implemented in lerobot_dataset_v3.py).
    """

    required_fields = ("future_task",)
    eval_required_fields = ()

    def _populate_extra_samples(self, data, samples):
        future_task = data.get("future_task", "")
        samples["atomic_task"] = f"Subtask: {future_task}"
        samples["prompt"] = "predict future subtask"


class BBoxCoTBuilder(BaseSamplesBuilder):
    """BBox CoT: the model outputs object bboxes before action.

    Data source (lerobot_dataset_v3.py):
        data["bbox"] <- bbox_index looks up the tasks table and returns a JSON
        string {"obj_name": [x1,y1,x2,y2], ...}. Coordinates are normalized and
        each frame may contain multiple objects.

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>\n<bbox_text>|
        Action: <EOV><EOC><action_action>|<eos>
    """

    required_fields = ("bbox",)
    eval_required_fields = ("bbox",)

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Additionally require non-empty bbox JSON; warning_lite batches may contain '{}'."""
        if not super().can_handle(data):
            return False
        import json

        try:
            bbox_data = json.loads(data["bbox"]) if isinstance(data["bbox"], str) else data["bbox"]
            return bool(bbox_data)
        except (json.JSONDecodeError, TypeError):
            return False

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        """Eval path: preserve non-empty bbox JSON check.

        Empty JSON '{}' is meaningless for inference. BBoxSubtaskCoTBuilder
        (eval_required_fields=()) skips this check because bbox is not in
        eval_required_fields.
        """
        if not super().can_handle_for_eval(data):
            return False
        # BBoxSubtaskCoTBuilder inherits this method with eval_required_fields=().
        # Since bbox is not in its eval_required_fields, skip the non-empty JSON check.
        # Convention: subclasses that do not need bbox as inference input must remove
        # bbox from eval_required_fields.
        if "bbox" not in self.eval_required_fields:
            return True
        import json

        try:
            bbox_data = json.loads(data["bbox"]) if isinstance(data["bbox"], str) else data["bbox"]
            return bool(bbox_data)
        except (json.JSONDecodeError, TypeError):
            return False

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><bbox_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    @staticmethod
    def _format_bbox_json(bbox_json: str) -> str:
        """Format {"obj": [x1,y1,x2,y2], ...} as CoT text."""
        import json

        def _loc(v):
            return f"<loc{max(0, min(1023, round(v * 1024))):04d}>"

        def _fmt(name, bbox):
            x1, y1, x2, y2 = bbox
            locs = "".join(_loc(v) for v in [y1, x1, y2, x2])
            return f"{name} {locs}"

        try:
            bbox_data = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
        except (json.JSONDecodeError, TypeError):
            return ""
        if not bbox_data:
            return ""
        parts = [_fmt(name, coords) for name, coords in bbox_data.items() if coords]
        return "BBox: " + "; ".join(parts) if parts else ""

    def _populate_extra_samples(self, data, samples):
        bbox_text = self._format_bbox_json(data.get("bbox", "{}"))
        samples["bbox"] = bbox_text
        samples["prompt"] = "predict bbox"


class BBoxSubtaskCoTBuilder(BBoxCoTBuilder):
    """Joint bbox + subtask CoT: output bbox, then subtask, then action.

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>\n<EOC><bbox_text>|<atomic_task_text>|
        Action: <EOV><action_action>|<eos>
    """

    required_fields = ("bbox", "atomic_task")
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><bbox_text>|<atomic_task_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(data, samples)  # sets samples["bbox"], samples["prompt"]
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        samples["prompt"] = "predict bbox, subtask and action"


class Trace2DCoTBuilder(BaseSamplesBuilder):
    """2D trace CoT: output the gripper 2D contact point before action.

    Data source (lerobot_dataset_v3.py):
        data["trace_2d"] <- 2d_trace_index looks up the tasks table, JSON string
            {"uv_left": [u,v]|null, "visb_left": bool,
             "uv_right": [u,v]|null, "visb_right": bool, ...}
        Coordinates are normalized to 0-1 and correspond to the current-frame
        projected gripper locations for the left/right arms.

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>
        <EOC><trace_2d_text>|
        Action: <EOV><action_action>|<eos>

    CoT text example:
        Trace: Left <loc0543><loc0436>; Right None
    """

    required_fields = ("trace_2d",)
    eval_required_fields = ()

    def can_handle(self, data: Dict[str, Any]) -> bool:
        if not super().can_handle(data):
            return False
        import json

        try:
            td = (
                json.loads(data["trace_2d"])
                if isinstance(data["trace_2d"], str)
                else data["trace_2d"]
            )
            # Meaningful only if at least one arm is visible.
            return bool(td.get("visb_left") or td.get("visb_right"))
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><trace_2d_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    @staticmethod
    def _format_trace_2d_json(trace_json: str) -> str:
        """Format trace_2d JSON as CoT text, encoding UV coordinates with <loc> tokens."""
        import json

        def _loc(v):
            return f"<loc{max(0, min(1023, round(v * 1024))):04d}>"

        try:
            data = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
        except (json.JSONDecodeError, TypeError):
            return ""

        parts = []
        uv_left = data.get("uv_left")
        if data.get("visb_left") and uv_left is not None:
            parts.append(f"Left {_loc(uv_left[0])}{_loc(uv_left[1])}")
        else:
            parts.append("Left None")

        uv_right = data.get("uv_right")
        if data.get("visb_right") and uv_right is not None:
            parts.append(f"Right {_loc(uv_right[0])}{_loc(uv_right[1])}")
        else:
            parts.append("Right None")

        return "Trace: " + "; ".join(parts)

    def _populate_extra_samples(self, data, samples):
        trace_text = self._format_trace_2d_json(data.get("trace_2d", "{}"))
        samples["trace_2d"] = trace_text
        samples["prompt"] = "predict 2d trace of gripper"


class SubtaskActionHintCoTBuilder(SubtaskCoTBuilder):
    """Joint subtask + action_hint CoT.

    Predict the subtask description first, then the current-frame gripper action hint.

    Data source (lerobot_dataset_v3.py):
        data["atomic_task"]  <- decoded from atomic_task_index
                                (r1lite/r1pro _merged_final_v30)
        data["action_hint"]  <- action_hint_index looks up the tasks table
                                (frame-level gripper action description), e.g.
                                "Right gripper moves forward right down, rotates roll positive pitch positive, closes."

    Template (embodiment=r1, 2 cameras, discrete):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;
        <prompt_text_!>
        <EOC><atomic_task_text>|<action_hint_text>|
        Action: <EOV><action_action>|<eos>
    """

    # Strictly require atomic_task to match the data reality: action_hint only appears
    # in _merged_final_v30, where atomic_task always coexists.
    required_fields = ("atomic_task", "action_hint")
    eval_required_fields = ()

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<prompt_text_!>\n<EOC><atomic_task_text>|<action_hint_text>|"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(
            data, samples
        )  # sets samples["atomic_task"], samples["prompt"]
        action_hint = data.get("action_hint") or ""
        samples["action_hint"] = f"ActionHint: {action_hint}"
        samples["prompt"] = "predict subtask and action hint"


class HighLevelAtomicTaskCoTBuilder(SubtaskCoTBuilder):
    """Use high_level_instruction as command, then CoT outputs atomic_task before action.

    Data source (r1lite/r1pro _merged_final_v30):
        data["high_level_instruction"] <- decoded from high_level_instruction_index
                                          (injected into command slot)
        data["atomic_task"]            <- decoded from atomic_task_index (CoT target)

    Template matches SubtaskCoTBuilder; only the injected command value changes to
    high_level_instruction.
    """

    required_fields = ("high_level_instruction", "atomic_task")
    eval_required_fields = ("high_level_instruction",)

    def _override_command(self, data):
        return data.get("high_level_instruction")

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(
            data, samples
        )  # sets samples["atomic_task"], samples["prompt"]
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"


class AtomicTaskBaseSamplesBuilder(BaseSamplesBuilder):
    """Match BaseSamplesBuilder template but inject atomic_task into the command slot.

    This builder has no CoT and goes directly to action.

    Data source:
        data["atomic_task"] <- decoded from atomic_task_index
                               (r1lite/r1pro _merged_final_v30)
    """

    required_fields = ("atomic_task",)
    eval_required_fields = ("atomic_task",)

    def _override_command(self, data):
        return data.get("atomic_task")


class MemoryCoTBuilder(MemorySamplesBuilder):
    """Memory input + memory_update CoT output.

    Data source (lerobot_dataset_v3.py):
        data["memory"]        <- prev_memory_index looks up the tasks table
                                 (previous-frame memory, input, masked with no loss)
        data["memory_update"] <- memory_index looks up the tasks table
                                 (current-frame memory update, CoT output target)

    Template (embodiment=r1, 2 cameras):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Memory: <memory_text_!> State: <proprio_proprio_!>;
        <cotprefix_text_!><EOC><memory_update_text>|
        Action: <EOV><action_action>|<eos>
    """

    required_fields = ("memory", "memory_update")
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><memory_update_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(data, samples)  # sets samples["memory"] = data["memory"]
        memory_update = data.get("memory_update")
        samples["memory_update"] = f"Updated Memory: {memory_update}" if memory_update else ""
        samples["cotprefix"] = "Please output the updated memory:"


class MemorySubtaskCoTBuilder(MemorySamplesBuilder):
    """Memory input + subtask CoT + memory_update CoT output.

    Data fields:
        data["memory"]        <- prev_memory_index (input, masked)
        data["atomic_task"]   <- atomic_task_index (CoT output)
        data["memory_update"] <- memory_index (CoT output)

    Template:
        <images><bos>Embodiment: ...; Task: ... Memory: <memory_text_!> State: ...;
        <cotprefix_text_!><EOC><atomic_task_text>|<memory_update_text>|
        Action: <EOV><action_action>|<eos>
    """

    required_fields = ("memory", "atomic_task")
    eval_required_fields = ("memory",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Memory: <memory_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><atomic_task_text>|<memory_update_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        super()._populate_extra_samples(data, samples)  # sets samples["memory"]
        samples["atomic_task"] = f"Subtask: {data.get('atomic_task', '')}"
        memory_update = data.get("memory_update")
        samples["memory_update"] = f"Updated Memory: {memory_update}" if memory_update else ""
        samples["cotprefix"] = "Please output the subtask and updated memory:"


class PlanStepCoTBuilder(BaseSamplesBuilder):
    """Plan input (masked) + plan step CoT output.

    Data source (lerobot_dataset_v3.py):
        data["plan"]      <- plan_index looks up the tasks table
                             (full plan text, input, masked with no loss)
        data["plan_step"] <- raw parquet int field (current step number, 0-based)

    Plan format example:
        "1. Pick up the towel\\n2. Move to the table\\n3. Fold the towel"

    Template (embodiment=r1, 2 cameras):
        <image0_image_!><image1_image_!><bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> Plan: <plan_text_!> State: <proprio_proprio_!>;
        <cotprefix_text_!><EOC><plan_step_text>|
        Action: <EOV><action_action>|<eos>
    """

    required_fields = ("plan", "plan_step")
    eval_required_fields = ("plan",)

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "Plan: <plan_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<cotprefix_text_!><EOC><plan_step_text>|\n"
            "Action: <EOV><action_action>|<eos>"
        )

    def _populate_extra_samples(self, data, samples):
        plan = data.get("plan", "")
        plan_step_raw = data.get("plan_step")
        if hasattr(plan_step_raw, "item"):
            step_num = plan_step_raw.item()
        else:
            step_num = int(plan_step_raw) if plan_step_raw is not None else 0
        # Extract the "{step_num}. <text>" line from the plan.
        current_step = ""
        for line in plan.split("\n"):
            line = line.strip()
            if line.startswith(f"{step_num}."):
                current_step = line[len(f"{step_num}.") :].strip()
                break
        samples["plan"] = plan
        samples["plan_step"] = f"Step: {current_step}" if current_step else ""
        samples["cotprefix"] = "Please output the current plan step:"


class MixedSamplesBuilder(BaseSamplesBuilder):
    """During training, randomly choose an annotated candidate builder by weight.

    During inference, use eval_builder. If no candidate builder has annotations
    (or inference has no eval_builder), fall back to BaseSamplesBuilder (no CoT).

    Config example:
        samples_builder:
          _target_: g05.data_processor.processor.samples_builder.MixedSamplesBuilder
          _partial_: true
          _recursive_: false        # prevent Hydra from recursively instantiating candidates
          candidates:
            - _target_: g05.data_processor.processor.samples_builder.SubtaskCoTBuilder
              weight: 1.0
            - _target_: g05.data_processor.processor.samples_builder.BBoxCoTBuilder
              weight: 0.5
          eval_builder:             # inference always uses this builder; null = no CoT
            _target_: g05.data_processor.processor.samples_builder.SubtaskCoTBuilder
    """

    def __init__(
        self,
        num_input_images: int,
        image_sizes: Dict[str, Any],
        embodiment_type: Optional[str] = None,
        candidates: Optional[List[Any]] = None,
        eval_builder: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(num_input_images, image_sizes, embodiment_type, **kwargs)
        self._training: bool = True

        _builder_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in ("hardcode_proprio_pad_zeros", "hardcode_action_pad_ones", "hardcode_instruction")
        }

        self._candidates: List[tuple] = []
        for cand in candidates or []:
            cls, weight = self._resolve_candidate(cand)
            if weight <= 0:
                raise ValueError(
                    f"MixedSamplesBuilder candidate {cls.__name__} has non-positive weight={weight}; "
                    f"weights must be > 0 or random.choices may raise ValueError or degenerate."
                )
            builder = cls(
                num_input_images=num_input_images,
                image_sizes=image_sizes,
                embodiment_type=embodiment_type,
                **_builder_kwargs,
            )
            self._candidates.append((builder, weight))

        self._eval_builder: Optional[BaseSamplesBuilder] = None
        if eval_builder is not None:
            cls, _ = self._resolve_candidate(eval_builder)
            self._eval_builder = cls(
                num_input_images=num_input_images,
                image_sizes=image_sizes,
                embodiment_type=embodiment_type,
                **_builder_kwargs,
            )

    @staticmethod
    def _resolve_candidate(cand: Any):
        """Resolve (cls, weight) from DictConfig, dict, or class."""
        from pydoc import locate

        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(cand, DictConfig):
                cand = OmegaConf.to_container(cand, resolve=True)
        except ImportError:
            pass
        if isinstance(cand, dict):
            cls = locate(cand["_target_"])
            if cls is None:
                raise ImportError(f"Cannot locate class: {cand['_target_']}")
            return cls, float(cand.get("weight", 1.0))
        return cand, 1.0

    @BaseSamplesBuilder.embodiment_type.setter
    def embodiment_type(self, value: Optional[str]) -> None:
        self._embodiment_type = value
        for builder, _ in self._candidates:
            builder.embodiment_type = value
        if self._eval_builder is not None:
            self._eval_builder.embodiment_type = value

    def set_training(self, training: bool) -> None:
        self._training = training
        # Propagate train/eval mode into nested MixedSamplesBuilder instances.
        for builder, _ in self._candidates:
            if hasattr(builder, "set_training"):
                builder.set_training(training)
        if self._eval_builder is not None and hasattr(self._eval_builder, "set_training"):
            self._eval_builder.set_training(training)

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        import random

        if not self._training:
            if self._eval_builder is not None and self._eval_builder.can_handle_for_eval(data):
                logger.debug("[MixedSamplesBuilder] eval → %s", type(self._eval_builder).__name__)
                return self._eval_builder.build(data, sample)
            logger.debug("[MixedSamplesBuilder] eval → BaseSamplesBuilder (fallback, no CoT)")
            return super().build(data, sample)

        applicable = [(b, w) for b, w in self._candidates if b.can_handle(data)]
        if not applicable:
            logger.debug("[MixedSamplesBuilder] train → BaseSamplesBuilder (no annotation matched)")
            return super().build(data, sample)
        builders, weights = zip(*applicable)
        chosen = random.choices(builders, weights=list(weights), k=1)[0]
        logger.debug(
            "[MixedSamplesBuilder] train → %s (from %d applicable)",
            type(chosen).__name__,
            len(applicable),
        )
        return chosen.build(data, sample)


# ====================================================================== #
#  Smoke test: python -m g05.data_processor.processor.samples_builder            #
# ====================================================================== #

if __name__ == "__main__":
    import sys

    import types as _types
    from g05.utils.common.special_tokens import SpecialTokenManager

    _paligemma = SpecialTokenManager.for_model("paligemma")
    _qwen35_base = SpecialTokenManager.for_model("qwen35")
    _mock_inst = _types.SimpleNamespace(
        bos_token="<|im_start|>",
        eos_token="<|endoftext|>",
        pad_token=None,
    )
    _qwen35_inst = SpecialTokenManager.for_model("qwen35", tokenizer=_mock_inst)
    TOKEN_MAPS = [
        ("paligemma", _paligemma),
        ("qwen35-base", _qwen35_base),
        ("qwen35-instruct", _qwen35_inst),
    ]

    CASES = [
        ("BaseSamplesBuilder", BaseSamplesBuilder, {}),
        ("MemorySamplesBuilder", MemorySamplesBuilder, {"embodiment_type": "r1"}),
        ("SubtaskCoTBuilder", SubtaskCoTBuilder, {"embodiment_type": "r1"}),
        ("BBoxCoTBuilder", BBoxCoTBuilder, {"embodiment_type": "r1"}),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder, {"embodiment_type": "r1"}),
        ("Trace2DCoTBuilder", Trace2DCoTBuilder, {"embodiment_type": "r1"}),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder, {"embodiment_type": "r1"}),
        ("MemoryCoTBuilder", MemoryCoTBuilder, {"embodiment_type": "r1"}),
        ("PlanStepCoTBuilder", PlanStepCoTBuilder, {"embodiment_type": "r1"}),
    ]

    DEFAULTS = dict(
        num_input_images=2, image_sizes={"image": (256, 256), "wrist_image": (256, 256)}
    )

    print("=" * 70)
    print("SamplesBuilder Smoke Test")
    print("=" * 70)

    for name, cls, kwargs in CASES:
        builder = cls(**{**DEFAULTS, **kwargs})
        print(f"\n--- {name} ---")
        print(f"  raw template:")
        for line in builder.template.split("\n"):
            print(f"    {repr(line)}")
        for model_name, token_map in TOKEN_MAPS:
            resolved = token_map.resolve_template(builder.template)
            print(f"  [{model_name}]:")
            for line in resolved.split("\n"):
                print(f"    {line}")

    # ---- MixedSamplesBuilder ----
    print("\n" + "=" * 70)
    print("MixedSamplesBuilder Tests")
    print("=" * 70)

    mixed = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            },
            {
                "_target_": "g05.data_processor.processor.samples_builder.BBoxCoTBuilder",
                "weight": 0.5,
            },
        ],
        eval_builder={"_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder"},
    )
    assert len(mixed._candidates) == 2, "Expected 2 candidates"
    assert mixed._eval_builder is not None, "eval_builder should be set"

    # can_handle: data contains atomic_task or bbox.
    data_with_subtask = {"atomic_task": "pick up cup"}
    data_with_bbox = {"bbox": '{"towel": [0.113, 0.479, 0.28, 0.688]}'}
    data_empty = {}

    applicable_subtask = [b for b, _ in mixed._candidates if b.can_handle(data_with_subtask)]
    applicable_bbox = [b for b, _ in mixed._candidates if b.can_handle(data_with_bbox)]
    applicable_empty = [b for b, _ in mixed._candidates if b.can_handle(data_empty)]

    assert (
        len(applicable_subtask) == 1 and type(applicable_subtask[0]).__name__ == "SubtaskCoTBuilder"
    ), f"data with atomic_task should only match SubtaskCoTBuilder, got {applicable_subtask}"
    assert len(applicable_bbox) == 1 and type(applicable_bbox[0]).__name__ == "BBoxCoTBuilder", (
        f"data with bbox should only match BBoxCoTBuilder, got {applicable_bbox}"
    )
    assert len(applicable_empty) == 0, "data without annotations should match no candidates"

    # sentinel / dirty value filtering
    subtask_builder = applicable_subtask[0]
    bbox_builder = applicable_bbox[0]
    import math

    assert not subtask_builder.can_handle({"task": "null"}), "string 'null' should be rejected"
    assert not subtask_builder.can_handle({"task": "none"}), "string 'none' should be rejected"
    assert not subtask_builder.can_handle({"task": "None"}), "string 'None' should be rejected"
    assert not subtask_builder.can_handle({"task": float("nan")}), "float NaN should be rejected"
    assert not subtask_builder.can_handle({"task": -1}), "int -1 should be rejected"
    assert not bbox_builder.can_handle({"bbox": "{}"}), "empty JSON '{}' should be rejected"
    assert not bbox_builder.can_handle({"bbox": "null"}), "bbox 'null' string should be rejected"
    print("  ✓ can_handle() filtering correct (incl. NaN/null/sentinel/-1/empty-JSON)")

    # Inference mode: eval_builder is used.
    mixed.set_training(False)
    assert not mixed._training, "set_training(False) should set _training=False"
    print("  ✓ set_training(False) OK")

    # Guard weight <= 0: should raise during __init__.
    try:
        MixedSamplesBuilder(
            **DEFAULTS,
            embodiment_type="r1",
            candidates=[
                {
                    "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                    "weight": 0,
                }
            ],
        )
        raise AssertionError("weight=0 should raise ValueError")
    except ValueError as e:
        assert "non-positive weight" in str(e), f"unexpected error: {e}"
    print("  ✓ weight <= 0 raises ValueError")

    # set_training propagation for nested MixedSamplesBuilder.
    inner_mixed = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            }
        ],
    )
    # outer uses inner as eval_builder, indirectly verifying MixedSamplesBuilder
    # also supports set_training.
    outer = MixedSamplesBuilder(
        **DEFAULTS,
        embodiment_type="r1",
        candidates=[
            {
                "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilder",
                "weight": 1.0,
            }
        ],
    )
    # Manually inject nesting to simulate a real scenario.
    outer._eval_builder = inner_mixed
    outer.set_training(False)
    assert inner_mixed._training is False, (
        "Nested MixedSamplesBuilder._training should be propagated as False by the outer builder"
    )
    outer.set_training(True)
    assert inner_mixed._training is True, (
        "Nested MixedSamplesBuilder._training should be propagated as True by the outer builder"
    )
    print("  ✓ nested MixedSamplesBuilder set_training propagates correctly")

    # embodiment_type propagation
    mixed.embodiment_type = "r1pro"
    assert mixed._eval_builder.embodiment_type == "r1pro"
    for b, _ in mixed._candidates:
        assert b.embodiment_type == "r1pro"
    print("  ✓ embodiment_type propagation OK")

    # ---- Slot-content semantics (regression guards) ----
    # These assertions prevent subtle bugs like the BBoxSubtaskCoTBuilder case:
    # samples["atomic_task"] must contain atomic_task text instead of task text
    # when atomic_task exists.
    print("\n" + "=" * 70)
    print("Slot-content Semantics")
    print("=" * 70)

    ATOMIC_TXT = "reach for and grasp the blue towel"
    TASK_TXT = "Fetch towel"
    HL_TXT = "Tidy the basket"
    HINT_TXT = "Right gripper closes."
    BBOX_TXT = '{"towel": [0.1, 0.2, 0.3, 0.4]}'
    MEM_TXT = "previous memory state"
    MEM_UPDATE_TXT = "new memory state"

    data_full = {
        "task": TASK_TXT,
        "atomic_task": ATOMIC_TXT,
        "high_level_instruction": HL_TXT,
        "action_hint": HINT_TXT,
        "bbox": BBOX_TXT,
        "memory": MEM_TXT,
        "memory_update": MEM_UPDATE_TXT,
    }
    data_no_atomic = {k: v for k, v in data_full.items() if k != "atomic_task"}

    def _slot(builder, data):
        out = {}
        builder._populate_extra_samples(data, out)
        return out

    # 1) atomic_task slot must use atomic_task text when atomic_task exists.
    atomic_slot_cases = [
        ("SubtaskCoTBuilder", SubtaskCoTBuilder),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder),
        ("MemorySubtaskCoTBuilder", MemorySubtaskCoTBuilder),
        ("HighLevelAtomicTaskCoTBuilder", HighLevelAtomicTaskCoTBuilder),
    ]
    for name, cls in atomic_slot_cases:
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        assert b.can_handle(data_full), f"{name} should can_handle(data_full)"
        out = _slot(b, data_full)
        slot = out.get("atomic_task", "")
        assert ATOMIC_TXT in slot, (
            f"{name}: atomic_task slot should contain atomic_task text, got {slot!r}; "
            f"fallback may have incorrectly used task text"
        )
        assert TASK_TXT not in slot, (
            f"{name}: coarse task text should not appear when data has atomic_task, got {slot!r}"
        )
    print(f"  ✓ atomic_task slot prefers atomic_task across {len(atomic_slot_cases)} builders")

    # 2) TaskAsSubtaskCoTBuilder: task fallback case for foldbench-style
    # hardcode_instruction configs.
    b = TaskAsSubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert b.can_handle(data_no_atomic), (
        "TaskAsSubtaskCoTBuilder should still can_handle when atomic_task is missing via task fallback"
    )
    slot = _slot(b, data_no_atomic).get("atomic_task", "")
    assert TASK_TXT in slot, f"TaskAsSubtaskCoTBuilder should use task as the CoT target, got {slot!r}"
    print("  ✓ TaskAsSubtaskCoTBuilder uses task text as the CoT target")

    # 3) Strict required fields: missing atomic_task must make can_handle=False.
    for name, cls in [
        ("SubtaskCoTBuilder", SubtaskCoTBuilder),
        ("BBoxSubtaskCoTBuilder", BBoxSubtaskCoTBuilder),
        ("SubtaskActionHintCoTBuilder", SubtaskActionHintCoTBuilder),
        ("MemorySubtaskCoTBuilder", MemorySubtaskCoTBuilder),
        ("HighLevelAtomicTaskCoTBuilder", HighLevelAtomicTaskCoTBuilder),
        ("AtomicTaskBaseSamplesBuilder", AtomicTaskBaseSamplesBuilder),
    ]:
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        assert not b.can_handle(data_no_atomic), (
            f"{name} must return can_handle=False when atomic_task is missing because it is required"
        )
    print(
        "  ✓ Subtask/BBoxSubtask/SubtaskActionHint/MemorySubtask/HighLevelAtomicTask/AtomicTaskBase strictly require atomic_task"
    )

    # 4) _override_command: AtomicTaskBase uses atomic_task; HighLevel uses
    # high_level_instruction.
    atb = AtomicTaskBaseSamplesBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert atb._override_command(data_full) == ATOMIC_TXT, (
        f"AtomicTaskBaseSamplesBuilder._override_command should return atomic_task, got {atb._override_command(data_full)!r}"
    )
    hl = HighLevelAtomicTaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert hl._override_command(data_full) == HL_TXT, (
        f"HighLevelAtomicTaskCoTBuilder._override_command should return high_level_instruction, got {hl._override_command(data_full)!r}"
    )
    # Other builders do not override by default; they return None and keep upstream
    # instructions.
    for cls in (BaseSamplesBuilder, SubtaskCoTBuilder, BBoxCoTBuilder):
        b = cls(**{**DEFAULTS, "embodiment_type": "r1"})
        assert b._override_command(data_full) is None, f"{cls.__name__} should not override command"
    print(
        "  ✓ _override_command: AtomicTaskBase=atomic_task, HighLevel=high_level_instruction, others=None"
    )

    # ---- can_handle_for_eval tests ----
    print("\n" + "=" * 70)
    print("can_handle_for_eval Tests")
    print("=" * 70)

    stcb = SubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert stcb.can_handle_for_eval({}), "SubtaskCoTBuilder.can_handle_for_eval({}) should be True"
    assert not stcb.can_handle({}), "SubtaskCoTBuilder.can_handle({}) must be False"
    _out = {}
    stcb._populate_extra_samples({}, _out)
    assert _out["atomic_task"] == "Subtask: ", (
        f"SubtaskCoTBuilder._populate_extra_samples with missing key: {_out['atomic_task']!r}"
    )
    print(
        "  ✓ SubtaskCoTBuilder: can_handle_for_eval(no-annot)=True, can_handle(no-annot)=False, populate safe"
    )

    bcb = BBoxCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert bcb.can_handle_for_eval({"bbox": '{"towel": [0.1, 0.2, 0.3, 0.4]}'}), (
        "BBoxCoTBuilder.can_handle_for_eval with valid bbox should be True"
    )
    assert not bcb.can_handle_for_eval({"bbox": "{}"}), (
        "BBoxCoTBuilder.can_handle_for_eval with empty bbox '{}' should be False"
    )
    assert not bcb.can_handle_for_eval({}), (
        "BBoxCoTBuilder.can_handle_for_eval without bbox should be False"
    )
    print("  ✓ BBoxCoTBuilder: can_handle_for_eval preserves non-empty JSON check")

    bscb = BBoxSubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert bscb.can_handle_for_eval({}), (
        "BBoxSubtaskCoTBuilder.can_handle_for_eval({}) should be True"
    )
    assert not bscb.can_handle({}), "BBoxSubtaskCoTBuilder.can_handle({}) must be False"
    _out2 = {}
    bscb._populate_extra_samples({}, _out2)
    assert _out2["atomic_task"] == "Subtask: ", (
        f"BBoxSubtaskCoTBuilder._populate_extra_samples missing keys: {_out2['atomic_task']!r}"
    )
    print("  ✓ BBoxSubtaskCoTBuilder: can_handle_for_eval(no-annot)=True, populate safe")

    hlb = HighLevelAtomicTaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert hlb.can_handle_for_eval({"high_level_instruction": "Tidy"}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle_for_eval with hl_inst should be True"
    )
    assert not hlb.can_handle_for_eval({}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle_for_eval without hl_inst should be False"
    )
    assert not hlb.can_handle({"high_level_instruction": "Tidy"}), (
        "HighLevelAtomicTaskCoTBuilder.can_handle requires atomic_task too"
    )
    print(
        "  ✓ HighLevelAtomicTaskCoTBuilder: can_handle_for_eval checks only high_level_instruction"
    )

    mcb = MemoryCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert mcb.can_handle_for_eval({"memory": "prev"}), (
        "MemoryCoTBuilder.can_handle_for_eval with memory should be True"
    )
    assert not mcb.can_handle_for_eval({}), (
        "MemoryCoTBuilder.can_handle_for_eval without memory should be False"
    )
    assert not mcb.can_handle({"memory": "prev"}), (
        "MemoryCoTBuilder.can_handle requires memory_update too"
    )
    print("  ✓ MemoryCoTBuilder: can_handle_for_eval checks only memory")

    mscb = MemorySubtaskCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    _out3 = {}
    mscb._populate_extra_samples({"memory": "M"}, _out3)
    assert _out3["atomic_task"] == "Subtask: ", (
        f"MemorySubtaskCoTBuilder._populate_extra_samples missing atomic_task: {_out3['atomic_task']!r}"
    )
    print("  ✓ MemorySubtaskCoTBuilder._populate_extra_samples safe with missing atomic_task")

    psb = PlanStepCoTBuilder(**{**DEFAULTS, "embodiment_type": "r1"})
    assert psb.can_handle_for_eval({"plan": "1. Pick\n2. Fold"}), (
        "PlanStepCoTBuilder.can_handle_for_eval with plan should be True"
    )
    assert not psb.can_handle_for_eval({}), (
        "PlanStepCoTBuilder.can_handle_for_eval without plan should be False"
    )
    assert not psb.can_handle({"plan": "1. Pick"}), (
        "PlanStepCoTBuilder.can_handle requires plan_step too"
    )
    print("  ✓ PlanStepCoTBuilder: can_handle_for_eval checks only plan")

    print("\n" + "=" * 70)
    print("All cases OK.")
    print("=" * 70)

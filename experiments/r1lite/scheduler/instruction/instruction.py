import json
import pathlib
from enum import Enum
from std_msgs.msg import String

INSTRUCTION_PATH = pathlib.Path(__file__).resolve().parent / "instruction.txt"

class InstructionAction(Enum):
    """Instruction handling action."""
    RESET = "reset"  # Reset the environment.
    CONTINUE = "continue"  # Continue execution.
    SKIP = "skip"  # Skip this observation: empty instruction or nothing.


class InstructionManager:
    def __init__(self, config):
        self.latest_bbox_dict = {"bbox": [], "head_img_base64": ""}
        self.source_target_bbox = None
        self.text_instruction_file = INSTRUCTION_PATH
        self.instruction = ""

        self.last_instruction = ""
        self.extra_info = None  # Store extra_info as an instance variable.

        self.use_vlm = config["use_vlm"]
        self.image_as_condition = config["image_as_condition"]
        self.bbox_as_instruction = config["bbox_as_instruction"]
        self.image_condition_lang_prefix = config["image_condition_lang_prefix"]
        self.pp_lower_half = config["pp_lower_half"]
        self._validate_config()

    def get_instruction(self, obs: dict) -> InstructionAction:
        '''
        Process instructions and update observation data.
        
        Args:
            obs: observation data dictionary, mutated in place.
            
        Returns:
            InstructionAction: indicates the next action to execute.
                - RESET: reset the environment.
                - CONTINUE: continue execution.
                - SKIP: skip this observation: empty instruction or nothing.
        '''
        if self.use_vlm:
            instruction, bbox, head_img_base64 = self._get_instruction_from_vlm()
        else:
            instruction = self._get_instruction_from_file()
        
        print(f"instruction: {instruction}")
        # Handle empty instruction or nothing.
        if instruction in ['', 'nothing']:
            self.last_instruction = instruction
            obs["task"] = instruction
            return InstructionAction.SKIP
        
        # Handle reset instruction.
        elif instruction == "reset":
            self.last_instruction = instruction
            obs["task"] = instruction
            return InstructionAction.RESET
        
        # Handle new instruction.
        elif instruction != self.last_instruction:
            if self.use_vlm:
                self.extra_info = self._get_extra_info_from_vlm(instruction, bbox, head_img_base64)
            else:
                self.extra_info = self._get_extra_info(instruction)

            if self.extra_info is not None:
                self.last_instruction = instruction
                # Update observation data.
                if "image" in self.extra_info:
                    obs["images"]["head_condition"] = self.extra_info["image"]
                if "instruction" in self.extra_info:
                    obs["task"] = self.extra_info["instruction"]
            # Pass source_target_bbox through to obs for the serve_policy.py processor.
            if self.source_target_bbox is not None:
                obs["source_target_bbox"] = self.source_target_bbox
            return InstructionAction.CONTINUE
        else:
            if "image" in self.extra_info:
                obs["images"]["head_condition"] = self.extra_info["image"]
            if "instruction" in self.extra_info:
                obs["task"] = self.extra_info["instruction"]
            if self.source_target_bbox is not None:
                obs["source_target_bbox"] = self.source_target_bbox

            return InstructionAction.CONTINUE

    def _get_instruction_from_vlm(self):
        latest_bbox_dict = self.latest_bbox_dict
        latest_bbox = latest_bbox_dict["bbox"]
        latest_head_img_base64 = latest_bbox_dict["head_img_base64"]
        return (self.instruction, latest_bbox, latest_head_img_base64)

    def _get_instruction_from_file(self):
        return self.text_instruction_file.read_text().replace('\n','')

    def _get_extra_info(self, instruction: str):
        return {"instruction": instruction}

    def _get_extra_info_from_vlm(self, instruction: str, bbox: list[int], head_img_base64: str):
        from utils.message.message_convert import decode_img_from_base64
        from utils.message.bbox_utils import get_bbox_image, get_paligemma_box_instruction

        extra_info = {}
        if head_img_base64 == "":
            return None
        latest_head_rgb = decode_img_from_base64(head_img_base64, output_format="rgb")
        if self.pp_lower_half:
            # Take the bottom half (height 360) of the image
            img_height = latest_head_rgb.shape[0]
            latest_head_rgb = latest_head_rgb[img_height//2:, :, :]
        if self.image_as_condition:
            condition_image = get_bbox_image(
                latest_head_rgb, 
                bbox, 
            )
            system_instruction = self.image_condition_lang_prefix
            extra_info["image"] = condition_image
            extra_info["instruction"] = system_instruction
        elif self.bbox_as_instruction:
            paligemma_instrctuion = get_paligemma_box_instruction(latest_head_rgb, bbox)
            extra_info["instruction"] = paligemma_instrctuion
        else:
            extra_info["instruction"] = instruction
        return extra_info

    def _refine_ll_instruction(self, instruction):
        if '[Low]' in instruction or instruction in ['reset', 'stop']:
            return instruction
        elif '[low]' in instruction:
            return instruction.replace("[low]", "[Low]")
        else:
            return f"[Low]:{instruction}"
    
    def _ehi_instruction_callback(self, msg: String):
        vlm_data_dict = json.loads(msg.data)
        bbox_dict = {}
        lower_prompt_list = vlm_data_dict["lower_prompt_list"]
        bbox_dict["bbox"] = vlm_data_dict["bbox"]
        bbox_dict["head_img_base64"] = vlm_data_dict["head_img_base64"]

        if not len(lower_prompt_list):
            return

        low_level_instruction = lower_prompt_list[0]
        self.instruction = self._refine_ll_instruction(low_level_instruction)
        self.latest_bbox_dict = bbox_dict
        # Parse source_target_bbox, the normalized two-box data provided by the new VLM.
        self.source_target_bbox = vlm_data_dict.get("source_target_bbox")

    def _validate_config(self):
        if self.use_vlm:
            return
        if not (self.image_as_condition or self.bbox_as_instruction):
            return
        raise ValueError(
            "image_as_condition and bbox_as_instruction require use_vlm=true. "
            "Remote bbox lookup has been removed; provide bbox data via hs/vlm_out2vla."
        )


if __name__ == "__main__":
    config = {
        "use_vlm": False,
        "image_as_condition": False,
        "bbox_as_instruction": False,
        "image_condition_lang_prefix": "",
        "pp_lower_half": False,
    }
    instruction_manager = InstructionManager(config)
    # Test reading instructions from file.
    instruction = instruction_manager._get_instruction_from_file()
    print(f"instruction from file: '{instruction}'")
    # Test refinement.
    print(instruction_manager._refine_ll_instruction("pick up the cup"))
    print(instruction_manager._refine_ll_instruction("[Low]pick up the cup"))
    print("InstructionManager smoke test passed.")

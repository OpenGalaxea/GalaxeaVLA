from typing import Dict, Any, Optional, Literal, List

import torch
from .base_processor import BaseProcessor, NormMode
from transformers import AutoTokenizer

IGNORE_INDEX = -100


class GalaxeaZeroProcessor(BaseProcessor):
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,

        action_state_transforms: Optional[List[Any]], 

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]], 

        action_state_merger, 

        # image transform
        train_transforms: Optional[Dict[str, List[Any]]],
        val_transforms: Optional[Dict[str, List[Any]]], 
        num_output_cameras: int, 

        # instruction transform
        drop_high_level_prob: float,
        use_zh_instruction: bool, 

        # tokenization
        pad_token_id: int,
        image_token_index: int,
        tokenizer_params: Dict[str, Any],
        max_image_text_tokens: int,
        max_text_tokens: int,
        num_input_cameras: int,
        num_image_tokens_per_camera: int,
    ):
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        
        super().__init__(
            shape_meta=shape_meta,
            num_obs_steps=num_obs_steps, 

            action_state_transforms=action_state_transforms, 

            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,

            action_state_merger=action_state_merger, 

            train_transforms=train_transforms,
            val_transforms=val_transforms,
            num_output_cameras=num_output_cameras,

            drop_high_level_prob=drop_high_level_prob, 
            use_zh_instruction=use_zh_instruction, 
        )

        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_params)

        self.max_text_tokens = max_text_tokens
        self.max_image_text_tokens = max_image_text_tokens
        self.num_image_tokens_per_camera = num_image_tokens_per_camera

        assert max_text_tokens <= max_image_text_tokens, "`max_text_tokens` must be less than or equal to `max_image_text_tokens`"
        self.num_input_images = num_obs_steps * num_input_cameras
        self.total_image_tokens = self.num_input_images * num_image_tokens_per_camera
        assert self.total_image_tokens > 0, "`total_image_tokens` must be greater than 0"
        assert max_image_text_tokens == self.total_image_tokens + max_text_tokens, \
            "`max_image_text_tokens` must be equal to `num_input_images * num_image_tokens + max_text_tokens`"

    def tokenize_instruction(self, instruction: str | List[str]) -> torch.Tensor:
        PROMPT_TEMPLATE = '{bos_token}Task: {instruction}, '
        if isinstance(instruction, str):
            instruction = [instruction]
        
        instruction = [PROMPT_TEMPLATE.format(bos_token=self.tokenizer.bos_token, instruction=instruct) for instruct in instruction]
        
        input_text = self.tokenizer(
            instruction,
            add_special_tokens=False,
            return_tensors="pt",
        )
        # 1. tokenize text instruction
        # [batch_size, text_seq_len]
        text_input_ids = input_text.input_ids
        # [batch_size, text_seq_len]
        attention_mask = input_text.attention_mask
        # [batch_size, text_seq_len] filled with `IGNORE_INDEX`
        labels = torch.full_like(text_input_ids, fill_value=IGNORE_INDEX).to(text_input_ids.device)

        batch_size, current_length = text_input_ids.shape
        # pad text_input_ids to max_text_tokens
        if current_length < self.max_text_tokens:
            padding_length = self.max_text_tokens - current_length
            text_input_ids = torch.nn.functional.pad(text_input_ids, (0, padding_length), value=self.pad_token_id)
            labels = torch.nn.functional.pad(labels, (0, padding_length), value=IGNORE_INDEX)
        else:
            text_input_ids = text_input_ids[:self.max_text_tokens]
            labels = labels[:self.max_text_tokens]
        
        # 2. tokenize image tokens
        image_input_ids = [self.image_token_index] * self.total_image_tokens
        image_input_ids = torch.tensor(image_input_ids).to(text_input_ids.device)
        image_input_ids = image_input_ids.unsqueeze(0).repeat(batch_size, 1)

        # 3. merge text_input_ids and image_input_ids
        # [batch_size, text_seq_len]
        input_ids = torch.cat([text_input_ids[:, :1], image_input_ids, text_input_ids[:, 1:]], dim=1)
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX).to(input_ids.device)
        attention_mask = input_ids.ne(self.pad_token_id)

        assert input_ids.shape[1] == self.max_image_text_tokens, \
            f"input_ids length {input_ids.shape[1]} does not match max_image_text"
        
        return input_ids.squeeze(0), labels.squeeze(0), attention_mask.squeeze(0)

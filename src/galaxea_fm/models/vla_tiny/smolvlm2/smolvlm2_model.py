# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Written by Orr Zohar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import torch
from torch import nn

from transformers.utils.logging import get_logger

from .config import SmolVLMConfig
from .modules import SmolVLMConnector
from .smolvlm2_vision import SmolVLMVisionTransformer
from .smolvlm2_text import SmolVLMTextModel

from galaxea_fm.models.kv_cache import KVCache


logger = get_logger(__name__)


class SmolVLMModel(nn.Module):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def __init__(self, config: SmolVLMConfig):
        super().__init__()
        self.config = config
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = SmolVLMVisionTransformer(config.vision_config)
        self.connector = SmolVLMConnector(config)
        self.text_model = SmolVLMTextModel(config.text_config)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def get_param_dtype(self):
        return next(iter(self.parameters())).dtype

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
            image_mask = input_ids == self.config.image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.get_param_dtype())  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            # no images, leave one empty image.
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()
        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states = image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    def _prepare_attention_mask_and_position_ids(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Prepare attention mask and position ids for the text model.
        This does NOT merge image embeddings - that should be done via inputs_merger first.
        """
        batch_size = inputs_embeds.shape[0]
        q_len = inputs_embeds.shape[1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: create a lower-triangular causal mask
            # Use a large negative value to mask out future tokens (will become 0 after softmax)
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=min_dtype, dtype=dtype, device=device
            )
            # Make it lower triangular (including diagonal) - tokens can attend to themselves and past tokens
            causal_mask = torch.triu(causal_mask, diagonal=1)
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            # cumsum gives [1, 2, 3, ...], so we need to subtract 1 to get [0, 1, 2, ...]
            position_ids = attention_mask.cumsum(-1)[:, -1] - 1
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 0 as position.
            # cumsum gives [1, 2, 3, ...], so we need to subtract 1 to get [0, 1, 2, ...]
            position_ids = (
                (attention_mask.cumsum(-1) - 1)
                .masked_fill_((attention_mask == 0), 0)
                .to(device)
            )

        return causal_mask, position_ids


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        # retrieve input_ids and inputs_embeds
        batch_size, seq_length = input_ids.shape
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # START VISUAL INPUTS INTEGRATION
        # Process images and merge with text embeddings
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask).to(inputs_embeds.device)
            # Merge image embeddings into text embeddings
            # This is safe to call even during generation because:
            # - On first pass: input_ids contains image tokens, merger replaces them with image features
            # - On subsequent passes: input_ids only contains the new token (no image tokens), merger does nothing
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        # Prepare attention mask and position ids
        attention_mask_prepared, position_ids = self._prepare_attention_mask_and_position_ids(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        output = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_prepared,
            position_ids=position_ids,
            kv_cache=kv_cache,
            **kwargs,
        )

        return output


class SmolVLMForConditionalGeneration(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SmolVLMModel(config)
        self.image_token_id = self.config.image_token_id
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.text_model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> dict:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            kv_cache=kv_cache,
            **kwargs,
        )

        hidden_states = output
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        output = {
            "logits": logits,
        }

        if kv_cache is not None:
            output["kv_cache"] = kv_cache

        return output


__all__ = ["SmolVLMForConditionalGeneration", "SmolVLMModel"]

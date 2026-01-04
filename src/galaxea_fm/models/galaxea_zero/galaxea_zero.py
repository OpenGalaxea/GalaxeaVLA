"""
This file is based on work from open-pi-zero (https://github.com/allenzren/open-pi-zero),
licensed under the MIT License.

Modifications:
   Copyright (c) 2025 Galaxea AI.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""
import os
import glob
from typing import Optional, Tuple
from safetensors import safe_open

import torch
from torch import nn

from accelerate.logging import get_logger
from einops import rearrange

from .modules import (
    ActionEncoder,
    ActionDecoder,
    SinusoidalPosEmb,
)
from ...utils.import_utils import get_obj_from_str

logger = get_logger(__name__)


class GalaxeaZero(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.cond_steps = cfg.cond_steps
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )
        self.num_input_images = cfg.num_input_images

        self.image_text_hidden_size = cfg.joint.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.joint.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.joint.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = 0.001

        # loss weights for padding actions
        self.padding_action_weight = cfg.get("padding_action_weight", 1.0) 

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = get_obj_from_str(cfg.vision.name)(cfg.vision)
        self.multi_modal_projector = get_obj_from_str(cfg.vision_projector.name)(cfg.vision_projector)

        # Mixtures
        self.joint_model = get_obj_from_str(cfg.joint.name)(cfg.joint)

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(cfg.time_hidden_size)
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(self.action_hidden_size)
        
        # proprio encoder
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = ActionDecoder(
            self.action_hidden_size,
            self.action_dim,
            num_layers=cfg.action_decoder_layers,
        )

        self.freeze_by_stage(stage=cfg.vla_training_strategy)

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        assert len(safetensors_files) > 0, "No pre-trained weights found"
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        logger.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        logger.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in lm part of the joint model")
    
    def freeze_non_lora_weights_in_action_expert(self):
        for name, param in self.joint_model.mixtures["action"].named_parameters():
            param.requires_grad = True if "lora_" in name else False
        logger.info("Froze non-lora weights in action expert part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
    
    def freeze_by_stage(self, stage: str):
        if stage in {"full-finetune", "vla-full-train"}:
            logger.info(f"[TRAINABLE]        🔥   =>> Vision Backbone `{self.vision_tower}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> VLM expert `{self.joint_model.mixtures['vlm']}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> Action expert `{self.joint_model.mixtures['action']}`")  # noqa: E501
        elif stage in {"action-expert-only"}:
            self.freeze_non_lora_weights_in_vlm()
            logger.info(f"[FROZEN]           🥶   =>> Vision Backbone `{self.vision_tower}`")  # noqa: E501
            logger.info(f"[FROZEN]           🥶   =>> VLM expert `{self.joint_model.mixtures['vlm']}`")  # noqa: E501
            logger.info(f"[TRAINABLE]        🔥   =>> Action expert `{self.joint_model.mixtures['action']}`")  # noqa: E501
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        device = attention_mask.device
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype, device=device,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/action attend to image/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1, device=device).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1, device=device).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
            device=device,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz, seq_len = input_ids.shape
        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        # NOTE: embed_tokens will use autocast dtype automatically
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # pixel_values: [Batch_Size, T, C, H, W], T can be # of history frames or more cameras
        pixel_values = rearrange(pixel_values, "b t c h w -> (b t) c h w")
        
        # [Batch_Size*T, Channels, Height, Width] -> [Batch_Size*T, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values)
        # [Batch_Size*T, Num_Patches, Embed_Dim] -> [Batch_Size, T*Num_Patches, Embed_Dim]
        selected_image_feature = rearrange(selected_image_feature, "(b t) c d -> b (t c) d", b=bsz)
        # [Batch_Size, T*Num_Patches, Embed_Dim] -> [Batch_Size, T*Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # AMP Best Practice: Use scaled_image_features.dtype to ensure consistency
        # In autocast context, all intermediate tensors will have the same dtype
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), 
            self.pad_token_id, 
            dtype=scaled_image_features.dtype,  # Use image features dtype for consistency
            device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        # Ensure dtype consistency when assigning
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        pixel_values: torch.FloatTensor,
        proprios: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz = pixel_values.size(0)

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        
        # AMP Best Practice: Use dtype from model's output for consistency
        dtype = inputs_embeds.dtype

        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = (
            self.split_full_mask_into_submasks(causal_mask))

        kv_caches = self.joint_model.init_mixture_caches()

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for i in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        pixel_values: torch.ByteTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        action_pad_masks: torch.BoolTensor,
        t: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        
        # AMP Best Practice: Use dtype from model's output for consistency
        dtype = inputs_embeds.dtype
        
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        bsz = pixel_values.size(0)
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # proprio
        assert proprios.shape[1] == self.num_proprio_tokens, \
            f"Expected {self.num_proprio_tokens} history steps, got {proprios.shape[1]}"
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)

        output, intermdeiate_outputs = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
            return_intermediate_layers=True,
        )
        action_embeds = output["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)
        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        l2 = (v_psi - d_psi) ** 2 # (bs, horizon, action_dim)
        action_weights = torch.ones_like(l2)
        action_weights[action_pad_masks] = self.padding_action_weight        
        loss_dict = {"fm_loss": torch.mean(action_weights * l2)}

        return loss_dict

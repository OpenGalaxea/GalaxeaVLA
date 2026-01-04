from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from transformers.utils import ModelOutput

from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.utils.pytorch_utils import set_global_seed

from .galaxea_zero import GalaxeaZero
from ...utils.import_utils import get_obj_from_str
logger = get_logger(__name__)


@dataclass
class GalaxeaModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    actions: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None

# TODO: inherit BasePolicy
class GalaxeaZeroPolicy(BasePolicy):
    def __init__(
        self,
        # TODO: make config compatible with hydra
        **model_cfg: DictConfig,
    ) -> None:
        super().__init__()
        model_cfg = OmegaConf.create(model_cfg)
        self.model: GalaxeaZero = get_obj_from_str(model_cfg.get("model_name"))(model_cfg)
        self.backbone_lr_multiplier = model_cfg.backbone_lr_multiplier

        self.pad_token_id = 0
        # Instance Attributes for a generic VLM
        self.all_module_keys, self.trainable_module_keys = None, None
        # Set Weight Initialization Seed for Projector Consistency
        # torch.manual_seed(self.config.hidden_size)
        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector", "action_expert"]
        self.trainable_module_keys = []

        # Action Expert
        # TODO: configure flow_sampling
        self.flow_sampling = "beta"
        flow_alpha = 1.5
        flow_beta = 1
        self.flow_t_max = 1 - 0.001
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        self.model_config = model_cfg  
        self.cached_key_values = None
        # Trackers
        self.vision_backbone_requires_grad = False

        if self.model_config.get("pretrained_model_path", None):
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()

    def get_optim_param_groups(self, lr, weight_decay):
        """
        This function returns a list of parameter groups for the optimizer
        """
        assert len(list(self.parameters())) == len(list(self.model.parameters()))
        action_expert_params_id = set(id(p) for p in self.model.action_expert_parameters)

        action_expert_params = [p for p in self.model.parameters() if id(p) in action_expert_params_id]
        backbone_params = [p for p in self.model.parameters() if id(p) not in action_expert_params_id]

        param_groups = [
            {
                "params": [p for p in backbone_params if p.requires_grad], 
                "lr": lr * self.backbone_lr_multiplier, 
                "weight_decay": weight_decay, 
                "name": "backbone", 
            },
            {
                "params": [p for p in action_expert_params if p.requires_grad], 
                "lr": lr, 
                "weight_decay": weight_decay, 
                "name": "action_expert"
            },
        ]

        all_requires_grad_params = [p for p in self.parameters() if p.requires_grad]
        assert len(all_requires_grad_params) == sum([len(g['params']) for g in param_groups])

        return param_groups

    @property
    def num_patches(self) -> int:
        return self.vision_backbone.vision_model.embeddings.num_patches

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    def forward(
        self,
        batch: Dict[str, torch.Tensor], 
        inference_mode=False,
    ):
        # for loss test
        # return self.test_forward_loss(batch)
        
        if inference_mode:
            was_training = self.training
            self.model.eval()
            normalized_action = self.forward_inference(
                input_ids = batch["input_ids"], 
                attention_mask = batch["attention_mask"], 
                pixel_values = batch["pixel_values"], 
                proprio = batch["proprio"],
            )
            batch["action"] = normalized_action
            self.model.train(was_training)
            return batch
        else:
            return self.forward_train(
                input_ids = batch["input_ids"], 
                attention_mask = batch["attention_mask"], 
                pixel_values = batch["pixel_values"], 
                actions = batch["action"], 
                action_pad_masks = batch["action_is_pad"], 
                proprio = batch["proprio"],
            )

        print_current_state()
        print(f"loss: {loss}")
        print("===" * 10)

        return loss, loss_value_dict

    def forward_train(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_pad_masks: Optional[torch.BoolTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
    ):
        device = input_ids.device
        dtype = pixel_values.dtype
        t = self.sample_fm_time(len(input_ids)).to(dtype)
        
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        # proprio_processd = proprio[:, None] # add cond_steps dimension NOTE: added at batchtransform

        loss_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            proprios=proprio,
            actions=actions,
            action_pad_masks=action_pad_masks,
            t=t.to(dtype=dtype, device=device)
        )
        loss = sum(loss_dict.values())
        loss_value_dict = {key: val.detach() for key, val in loss_dict.items()}
        
        return loss, loss_value_dict

    @torch.no_grad()
    def forward_inference(
        self,                    
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
    ):
        # input_ids, attention_mask = self.pre_process_inputs(input_ids, attention_mask)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)

        sampled_actions = self.model.infer_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            proprios=proprio,
        )

        return sampled_actions

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(batch, inference_mode=True)

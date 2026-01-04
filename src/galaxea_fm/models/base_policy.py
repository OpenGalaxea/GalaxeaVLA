"""
Reference:
- https://github.com/real-stanford/diffusion_policy
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from galaxea_fm.utils.normalizer import LinearNormalizer


class BasePolicy(nn.Module):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    @classmethod
    def from_pretrained(cls, cfg): # type: ignore
        pass  # Load model weights from pretrained checkpoint

    def get_optim_param_groups(self, lr, weight_decay) -> List[Dict]:
        raise NotImplementedError()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training:
            return self.compute_loss(batch)
        else:
            return self.predict_action(batch)

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()
    
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError


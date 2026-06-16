
"""
Reference:
- https://github.com/real-stanford/diffusion_policy
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from g05.utils.data.normalizer import LinearNormalizer


class BasePolicy(nn.Module):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    @property
    def fp32_param_patterns(self) -> List[str]:
        """Subclasses must override and declare fp32 parameter patterns by substring match.

        Models that do not need fp32 parameters should explicitly return an empty list.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement fp32_param_patterns property. "
            "Return [] if no parameters need float32."
        )

    def apply_fp32_params(self) -> None:
        """Cast parameters matching fp32_param_patterns to float32."""
        patterns = self.fp32_param_patterns
        if not patterns:
            return
        for name, param in self.named_parameters():
            if any(p in name for p in patterns):
                param.data = param.data.to(dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, cfg):  # type: ignore
        pass  # Load model weights from pretrained checkpoint

    def get_optim_param_groups(self, lr, weight_decay) -> List[Dict]:
        raise NotImplementedError()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training:
            return self.compute_loss(batch)
        else:
            return self.predict_action(batch)

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def estimate_training_flops_per_sample(self) -> int:
        """Estimate FLOPs per sample per training step (forward + backward).

        Override in subclass to enable MFU tracking.
        Returns 0 by default (MFU tracking disabled).
        """
        return 0

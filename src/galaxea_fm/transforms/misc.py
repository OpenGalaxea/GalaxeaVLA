from typing import List

import torch

from galaxea_fm.transforms import BaseActionStateTransform


class WrapStateAngle(BaseActionStateTransform):
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch
    
    def backward(self, batch):
        return batch
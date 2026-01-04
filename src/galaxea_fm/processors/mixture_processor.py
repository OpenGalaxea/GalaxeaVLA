from typing import Dict, Any
from abc import ABC

from galaxea_fm.processors.base_processor import BaseProcessor


class MixtureProcessor(ABC):
    def __init__(
        self, 
        embodiment_processors: Dict[str, BaseProcessor], 
    ):
        self.processors = embodiment_processors

    def train(self):
        for processor in self.processors.values():
            processor.train()

    def eval(self):
        for processor in self.processors.values():
            processor.eval()
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data["embodiment"] in self.processors:
            processor = self.processors[data["embodiment"]]
            return processor.preprocess(data)
        else:
            raise ValueError(f"No processor found for embodiment: {data['embodiment']}")
        
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data["embodiment"] in self.processors:
            processor = self.processors[data["embodiment"]]
            return processor.postprocess(data)
        else:
            raise ValueError(f"No processor found for embodiment: {data['embodiment']}")
    
    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any]): 
        pe = set(self.processors.keys())
        de = set(dataset_stats.keys())
        assert pe == de, f"Embodiment of processors {pe} and dataset stats {de} mismatch."

        for e, p in self.processors.items():
            p.set_normalizer_from_stats(dataset_stats[e])

    def __getitem__(self, emb: str):
        assert emb in self.processors
        return self.processors[emb]
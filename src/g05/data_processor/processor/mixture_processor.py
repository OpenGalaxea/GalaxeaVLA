# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

from abc import ABC
from typing import Any, Dict

from g05.data_processor.processor.base_processor import BaseProcessor


class MixtureProcessor(ABC):
    def __init__(
        self,
        embodiment_processors: Dict[str, BaseProcessor],
    ):
        # Keyed by logical embodiment_type. Outer dataset names are only data
        # source names and must not be used for processor routing.
        self.processors = embodiment_processors
        self.pad_token_id = embodiment_processors[next(iter(embodiment_processors))].pad_token_id
        for emb_type, processor in self.processors.items():
            assert processor.pad_token_id == self.pad_token_id, (
                f"Pad token id mismatch for embodiment_type {emb_type}."
            )

    def train(self):
        for processor in self.processors.values():
            processor.train()

    def eval(self):
        for processor in self.processors.values():
            processor.eval()

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "MixtureProcessor is a registry/manager only. "
            "Bind or call a concrete per-embodiment processor instead."
        )

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "MixtureProcessor is a registry/manager only. "
            "Bind or call a concrete per-embodiment processor instead."
        )

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any]):
        for emb_type, processor in self.processors.items():
            if emb_type not in dataset_stats:
                raise KeyError(
                    f"No stats found for embodiment_type '{emb_type}'. "
                    f"Available stats keys: {list(dataset_stats.keys())}"
                )
            stats = dataset_stats[emb_type]
            try:
                processor.set_normalizer_from_stats(stats)
            except AssertionError as ex:
                raise AssertionError(f"Embodiment type '{emb_type}': {ex}") from ex

    def __getitem__(self, emb: str):
        if emb in self.processors:
            return self.processors[emb]
        raise KeyError(
            f"embodiment_type '{emb}' not found in processors. "
            f"Available: {sorted(self.processors.keys())}"
        )

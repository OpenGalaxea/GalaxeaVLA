# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Base classes for data transforms.

Only depends on abc/torch, keeping it lightweight enough for upstream modules
such as ``g05.utils.data.normalizer`` to import safely without processor cycles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch


class BaseActionStateTransform(ABC):
    """
    Unified base class for all data transforms.

    Attributes:
        invertible: whether the transform is invertible. If True, backward must
            recover the input to forward.
        rtol/atol: error tolerances for roundtrip tests.
    """

    invertible: bool = True  # Subclasses may override.
    rtol: float = 1e-5
    atol: float = 1e-5

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward transform."""
        raise NotImplementedError

    def backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Backward transform. Non-invertible transforms return the original batch by default."""
        if not self.invertible:
            return batch
        raise NotImplementedError(
            f"{self.__class__.__name__} is marked as invertible but backward() is not implemented"
        )

    def test_roundtrip(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Roundtrip test: forward -> backward, verifying that the original data is recovered.
        Only valid when invertible=True.

        Returns:
            Dictionary containing test results:
            - passed: bool
            - max_error: float
            - error_keys: List[str]  # keys exceeding the threshold
        """
        if not self.invertible:
            return {"passed": True, "reason": "not_invertible"}

        # Deep copy to avoid in-place mutation.
        original = self._deep_copy_batch(batch)

        # Forward → Backward
        transformed = self.forward(batch)
        recovered = self.backward(transformed)

        # Compare errors.
        errors = {}
        error_keys = []

        for k in original:
            if k not in recovered:
                errors[k] = float("inf")
                error_keys.append(k)
                continue

            orig_v, rec_v = original[k], recovered[k]

            if isinstance(orig_v, torch.Tensor) and isinstance(rec_v, torch.Tensor):
                if orig_v.shape != rec_v.shape:
                    errors[k] = float("inf")
                    error_keys.append(k)
                else:
                    max_err = (orig_v - rec_v).abs().max().item()
                    errors[k] = max_err
                    if max_err > self.atol:
                        error_keys.append(k)
            elif isinstance(orig_v, dict) and isinstance(rec_v, dict):
                # Recursively process nested dictionaries.
                nested_result = self._compare_nested_dicts(orig_v, rec_v, k)
                errors.update(nested_result["errors"])
                error_keys.extend(nested_result["error_keys"])

        max_error = max(errors.values()) if errors else 0.0

        return {
            "passed": len(error_keys) == 0,
            "max_error": max_error,
            "error_keys": error_keys,
        }

    def _deep_copy_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-copy batch to avoid in-place mutation."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.clone()
            elif isinstance(v, dict):
                result[k] = self._deep_copy_batch(v)
            elif isinstance(v, list):
                result[k] = [item.clone() if isinstance(item, torch.Tensor) else item for item in v]
            else:
                result[k] = v
        return result

    def _compare_nested_dicts(
        self, orig: Dict[str, Any], rec: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Recursively compare nested dictionaries."""
        errors = {}
        error_keys = []

        for k in orig:
            full_key = f"{prefix}.{k}" if prefix else k

            if k not in rec:
                errors[full_key] = float("inf")
                error_keys.append(full_key)
                continue

            orig_v, rec_v = orig[k], rec[k]

            if isinstance(orig_v, torch.Tensor) and isinstance(rec_v, torch.Tensor):
                if orig_v.shape != rec_v.shape:
                    errors[full_key] = float("inf")
                    error_keys.append(full_key)
                else:
                    max_err = (orig_v - rec_v).abs().max().item()
                    errors[full_key] = max_err
                    if max_err > self.atol:
                        error_keys.append(full_key)
            elif isinstance(orig_v, dict) and isinstance(rec_v, dict):
                nested = self._compare_nested_dicts(orig_v, rec_v, full_key)
                errors.update(nested["errors"])
                error_keys.extend(nested["error_keys"])

        return {"errors": errors, "error_keys": error_keys}

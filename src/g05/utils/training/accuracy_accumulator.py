"""
TrainAccuracyAccumulator: grad-accum-aware train-time accuracy accumulator.

Background: Policy.forward computes one accuracy value per micro-batch. If written
directly to ``self.train_xxx`` attributes, grad_accumulation > 1 keeps only the last
micro-batch, so logged accuracy does not represent all data seen by the optimizer step.

Solution: use this class in Policy to maintain a {field: [per_microbatch_values]}
dict. Push every micro-batch, then flush micro-batch means at log steps and clear.

Typical usage:
    class MyPolicy:
        def __init__(self):
            self.train_acc = TrainAccuracyAccumulator()

        def forward(self, batch):
            ...
            self.train_acc.push("overall", overall_accuracy)
            self.train_acc.push("action_token", action_accuracy)
            if self.predict_cot:
                self.train_acc.push("cot", cot_accuracy)

    # In the finetune.py log step:
    acc_dict = model.train_acc.flush()
    # {"overall": ..., "action_token": ..., "cot": ...}
"""

from __future__ import annotations

from typing import Dict, List


class TrainAccuracyAccumulator:
    """Grad-accum-aware per-field accuracy accumulator.

    Field semantics, guaranteed by the Policy caller:
      - overall       — all-token accuracy (text + action)
      - action_token  — action-token-only accuracy, the true action accuracy
      - cot           — non-action token accuracy, recorded only for predict_cot

    Allowed fields are restricted by ``VALID_FIELDS`` to avoid silently swallowing
    typos. Fields not listed in ``VALID_FIELDS`` raise ``ValueError``.
    """

    VALID_FIELDS = ("overall", "action_token", "cot")

    def __init__(self):
        self._values: Dict[str, List[float]] = {}

    def push(self, field: str, value) -> None:
        """Register one micro-batch accuracy value.

        Args:
            field: field name, must be in ``VALID_FIELDS``
            value: float or 0-d tensor; tensors are converted with ``.item()``
        """
        if field not in self.VALID_FIELDS:
            raise ValueError(
                f"TrainAccuracyAccumulator.push: unknown field {field!r}; "
                f"valid fields are {self.VALID_FIELDS}"
            )
        val = float(value.item()) if hasattr(value, "item") else float(value)
        self._values.setdefault(field, []).append(val)

    def flush(self) -> Dict[str, float]:
        """Return micro-batch means for each field, then clear.

        Fields with no data are omitted from the returned dict. Callers should use
        checks like ``"overall" in acc_dict`` to decide whether to log, instead of
        defaulting to 0.0.
        """
        out: Dict[str, float] = {}
        for field, values in self._values.items():
            if values:
                out[field] = sum(values) / len(values)
        self._values = {}
        return out

    def clear(self) -> None:
        """Clear without returning, for exception paths or manual reset."""
        self._values = {}

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._values.items()}
        return f"TrainAccuracyAccumulator({counts})"

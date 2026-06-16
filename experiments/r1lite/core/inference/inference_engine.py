from abc import ABC, abstractmethod
from typing import Dict, Any


class InferenceEngine(ABC):
    """Base inference engine with a unified interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the inference service."""
        pass

    @abstractmethod
    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference and predict actions.

        Args:
            batch: preprocessed batch data.

        Returns:
            Dictionary containing predicted actions.
        """
        pass

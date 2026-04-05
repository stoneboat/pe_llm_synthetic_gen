"""Abstract base class for downstream utility evaluation tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class UtilityTask(ABC):
    """Base class for downstream evaluation tasks.

    Each task knows how to evaluate the quality of synthetic data
    against the original private data distribution.
    """

    def __init__(self, config: dict):
        self.config = config
        self.name: str = config.get("name", "unknown")

    @abstractmethod
    def evaluate(
        self,
        synthetic_data_path: str,
        dataset_config: dict,
        result_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the evaluation and return metrics.

        Args:
            synthetic_data_path: path to synthetic samples CSV.
            dataset_config: dataset configuration for finding test/dev data.
            result_dir: directory to save evaluation outputs.

        Returns:
            Dict of metric_name -> value.
        """
        ...

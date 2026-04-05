"""Abstract base class for dataset adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import collections
import numpy as np


@dataclass
class DatasetResult:
    """Container for loaded dataset."""
    samples: List[str]
    labels: List[str]
    label_counter: collections.Counter
    label_indexer: Dict[str, List[int]]


class DatasetAdapter(ABC):
    """Base class for dataset adapters.

    Each adapter knows how to:
    - load private training data and labels
    - load/compute private embeddings
    - provide dataset-specific metadata (label columns, prompt templates, etc.)
    - format output CSVs
    """

    def __init__(self, config: dict):
        self.config = config
        self.name: str = config.get("name", "unknown")
        self.train_data_file: str = config.get("train_data_file", "")
        self.dev_data_file: str = config.get("dev_data_file", "")
        self.test_data_file: str = config.get("test_data_file", "")
        self.num_private_samples: int = config.get("num_private_samples", -1)
        self.embeddings_file: str = config.get("embeddings_file", "")

    @abstractmethod
    def load_data(
        self,
        data_file: Optional[str] = None,
        num_samples: int = -1,
        subsample_one_class: bool = False,
        gen: bool = False,
    ) -> DatasetResult:
        """Load dataset and return structured result."""
        ...

    @abstractmethod
    def get_label_columns(self) -> List[str]:
        """Return the label column names for this dataset."""
        ...

    @abstractmethod
    def get_csv_header(self) -> List[str]:
        """Return CSV header for saving synthetic samples."""
        ...

    @abstractmethod
    def format_sample_row(self, text: str, label: str) -> List[str]:
        """Format a single sample + label into a CSV row."""
        ...

    def get_variation_type(self) -> str:
        """Return the default variation type for this dataset."""
        return self.config.get("variation_type", "yelp_rephrase_tone")

    def get_subcategory_source(self) -> Optional[str]:
        """Return path to subcategory/writers file, or None."""
        return self.config.get("subcategory_source", None)

    def get_default_embedding_model(self) -> str:
        """Return the default embedding model for this dataset."""
        return self.config.get("embedding_model", "stsb-roberta-base-v2")

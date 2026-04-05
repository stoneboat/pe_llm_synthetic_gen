"""Abstract base class for DP synthetic text generation mechanisms.

The mechanism interface is designed so that:
1. The outer loop (generate initial -> iterate rounds) is common structure.
2. Each mechanism controls how DP counting, weight computation, and
   survivor selection work.
3. A future AnnealedTopKMechanism can override weight computation
   without changing dataset or downstream task code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import collections
import numpy as np


@dataclass
class RoundResult:
    """Output of a single PE round."""
    selected_samples: List[str]
    selected_labels: List[str]
    next_round_samples: List[str]
    next_round_labels: List[str]
    updated_label_counter: collections.Counter
    count_per_class: Dict[str, Tuple[np.ndarray, np.ndarray]]  # class -> (noisy, clean)


class Mechanism(ABC):
    """Base class for DP synthetic text generation mechanisms.

    Subclasses must implement:
    - generate_initial: create the initial synthetic pool
    - run_round: execute one PE iteration (embed, count, select, vary)
    - compute_weights: convert DP histogram counts to selection weights

    The outer experiment runner calls these methods in sequence.
    """

    def __init__(self, config: dict, api: Any):
        """
        Args:
            config: mechanism-specific configuration dict.
            api: the text generation API (e.g., HFAPI instance).
        """
        self.config = config
        self.api = api

        # Common PE parameters
        self.noise_multiplier: float = config.get("noise_multiplier", 0.0)
        self.num_nearest_neighbor: int = config.get("num_nearest_neighbor", 1)
        self.nn_mode: str = config.get("nn_mode", "L2")
        self.count_threshold: float = config.get("count_threshold", 0.0)
        self.select_syn_mode: str = config.get("select_syn_mode", "rank")
        self.save_syn_mode: str = config.get("save_syn_mode", "selected")
        self.combine_divide_L: int = config.get("L", 1)
        self.init_combine_divide_L: int = config.get("init_L", 1)
        self.lookahead_degree: int = config.get("lookahead_degree", 0)
        self.lookahead_self: bool = config.get("lookahead_self", False)
        self.donnot_keep_last_iter: bool = config.get("donnot_keep_last_iter", False)
        self.compute_fid: bool = config.get("compute_fid", True)

        # Schedules (parsed from config)
        self.num_samples_schedule: List[int] = config.get("num_samples_schedule", [1000])
        self.variation_degree_schedule: List[float] = config.get("variation_degree_schedule", [0.5])

        # Feature extractor settings
        self.feature_extractor: str = config.get("feature_extractor", "stsb-roberta-base-v2")
        self.feature_extractor_batch_size: int = config.get("feature_extractor_batch_size", 1024)

    @abstractmethod
    def generate_initial(
        self,
        num_samples: int,
        label_counter: collections.Counter,
    ) -> Tuple[List[str], List[str], collections.Counter]:
        """Generate the initial synthetic sample pool.

        Returns:
            (samples, labels, sync_label_counter)
        """
        ...

    @abstractmethod
    def run_round(
        self,
        t: int,
        syn_samples: List[str],
        additional_info: List[str],
        sync_labels_counter: collections.Counter,
        private_classes: List[str],
        all_private_features: np.ndarray,
        private_labels_indexer: Dict[str, List[int]],
        result_folder: str,
    ) -> RoundResult:
        """Execute one PE round.

        Args:
            t: current round index
            syn_samples: current synthetic samples
            additional_info: current labels
            sync_labels_counter: per-class sample counts
            private_classes: ordered class list
            all_private_features: private embeddings
            private_labels_indexer: class -> index mapping
            result_folder: path for saving artifacts

        Returns:
            RoundResult with selected and next-round samples.
        """
        ...

    @abstractmethod
    def compute_weights(
        self,
        noisy_counts: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """Convert DP histogram counts into selection weights.

        This is the key method that differs between mechanisms:
        - Original: uses counts directly (rank or prob mode)
        - AnnealedTopK: applies thresholding + top-k truncation + normalization

        Args:
            noisy_counts: noisy DP histogram counts for one class.
            t: current round index (for schedule-dependent mechanisms).

        Returns:
            Weight vector (same length as noisy_counts).
        """
        ...

    def expand_initial_pool(
        self,
        seed_samples: List[str],
        seed_labels: List[str],
        sync_labels_counter: collections.Counter,
        private_classes: List[str],
    ) -> Tuple[List[str], List[str], collections.Counter]:
        """Expand initial pool when L > 1.

        Default implementation uses variation API.
        Can be overridden by subclasses.
        """
        if self.init_combine_divide_L <= 1:
            return seed_samples, seed_labels, sync_labels_counter

        syn_samples = []
        additional_info = []
        current_idx = 0

        for class_ in private_classes:
            num_samples_per_class = sync_labels_counter[class_]
            if num_samples_per_class == 0:
                continue

            seed_per_class = seed_samples[current_idx:current_idx + num_samples_per_class]
            labels_per_class = seed_labels[current_idx:current_idx + num_samples_per_class]

            new_variants, _, _, _, _ = self.api.text_variation(
                sequences=seed_per_class,
                additional_info=labels_per_class,
                num_variations_per_sequence=self.init_combine_divide_L - 1,
                variation_degree=self.variation_degree_schedule[0],
            )

            syn_samples.extend(seed_per_class)
            for x in new_variants:
                syn_samples.extend(x.tolist())
            additional_info.extend(labels_per_class * self.init_combine_divide_L)

            current_idx += num_samples_per_class
            sync_labels_counter[class_] = num_samples_per_class * self.init_combine_divide_L

        return syn_samples, additional_info, sync_labels_counter

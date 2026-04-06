"""PE with annealed top-k post-processing (Algorithm 2 from the proposal).

Extends the original AUG-PE mechanism by replacing the weight computation
with thresholded, top-k-truncated, normalized weights.  Everything else
(embedding, DP histogram, variation, saving) is inherited unchanged.

Algorithm 2 pseudocode (lines 22-41):
    c_tilde  = DP_NN_Histogram counts (noisy)
    w        = max(c_tilde - H, 0)            # elementwise threshold
    J        = TopKIndices(w, k_top^(t))       # keep top-k
    w_hat    = w  restricted to J, zeros elsewhere
    if sum(w_hat) > 0:
        return w_hat / sum(w_hat)              # normalize
    else:
        return uniform over J                  # fallback
"""

import logging
from typing import Any, List

import numpy as np

from ..registry import register
from .original_aug_pe import OriginalAugPEMechanism


@register("mechanism", "pe_top_k")
class PETopKMechanism(OriginalAugPEMechanism):
    """AUG-PE with annealed top-k weight post-processing.

    New config keys consumed (beyond those in OriginalAugPEMechanism):
        threshold_H       (float):  elementwise threshold subtracted from
                                    noisy counts before top-k.  Default 0.0.
        k_top_schedule    (list[int]):  per-round k_top values.  Length must
                                    equal ``len(num_samples_schedule)``.
                                    If a single int is given it is broadcast.
    """

    def __init__(self, config: dict, api: Any):
        super().__init__(config, api)
        self.threshold_H: float = float(config.get("threshold_H", 0.0))

        raw_schedule = config.get("k_top_schedule", [])
        if not raw_schedule:
            raise ValueError(
                "pe_top_k mechanism requires 'k_top_schedule' in config"
            )
        # Accept a single value and broadcast to schedule length
        if isinstance(raw_schedule, (int, float)):
            raw_schedule = [int(raw_schedule)] * len(self.num_samples_schedule)
        elif isinstance(raw_schedule, str):
            raw_schedule = [int(x.strip()) for x in raw_schedule.split(",") if x.strip()]
            if len(raw_schedule) == 1:
                raw_schedule = raw_schedule * len(self.num_samples_schedule)
        else:
            raw_schedule = [int(x) for x in raw_schedule]
            if len(raw_schedule) == 1:
                raw_schedule = raw_schedule * len(self.num_samples_schedule)

        self.k_top_schedule: List[int] = raw_schedule

    # ------------------------------------------------------------------
    # Override: relax the assertion that requires sum(counts) > 0,
    # because the top-k fallback explicitly handles the all-zero case.
    # ------------------------------------------------------------------
    def _validate_counts(self, counts: np.ndarray, class_name: str) -> None:
        """Allow all-zero counts — the fallback handles this."""
        pass

    # ------------------------------------------------------------------
    # Core override: Algorithm 2 weight computation
    # ------------------------------------------------------------------
    def compute_weights(self, noisy_counts: np.ndarray, t: int) -> np.ndarray:
        """Annealed top-k weight computation (Algorithm 2, lines 28-41).

        Args:
            noisy_counts: output of dp_nn_histogram (already max(c_tilde, 0)
                          when count_threshold=0 in the base DP counter).
            t: round index, used to look up k_top_schedule[t].

        Returns:
            Normalized weight vector of the same shape as noisy_counts.
        """
        n = len(noisy_counts)
        if n == 0:
            return noisy_counts

        # --- line 28: elementwise thresholding ---
        w = np.maximum(noisy_counts - self.threshold_H, 0.0)

        # --- line 29: top-k index set J ---
        k_top = self.k_top_schedule[t]
        # Clamp to valid range
        k_top = max(k_top, 1)
        k_top = min(k_top, n)

        # argpartition gives indices of the k_top largest values (unordered)
        top_k_indices = np.argpartition(w, -k_top)[-k_top:]

        # --- lines 30-33: restrict to J ---
        w_hat = np.zeros_like(w)
        w_hat[top_k_indices] = w[top_k_indices]

        # --- lines 34-41: normalize or fallback ---
        total = w_hat.sum()
        if total > 0:
            return w_hat / total
        else:
            # Fallback: uniform over J
            logging.debug(
                "pe_top_k round %d: all thresholded weights are zero, "
                "falling back to uniform over top-%d set",
                t, k_top,
            )
            w_hat[top_k_indices] = 1.0
            return w_hat / w_hat.sum()

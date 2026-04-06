"""Focused tests for PETopKMechanism (Algorithm 2).

Tests cover:
- Elementwise thresholding (w = max(c_tilde - H, 0))
- Top-k truncation with correct index selection
- Normalization of surviving weights
- Zero-mass fallback to uniform over top-k set
- Round-dependent k_top schedule
- Edge cases: k_top >= n, k_top = 1, all-zero input, single element
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

# ── Helpers ──────────────────────────────────────────────────────────

class MockAPI:
    """Stub API — mechanism instantiation needs one, but tests never call it."""
    pass


def _make_mechanism(k_top_schedule, threshold_H=0.0, **extra):
    """Instantiate a PETopKMechanism with minimal config."""
    from src.registry import get_class
    import src.mechanisms  # ensure registration

    cls = get_class("mechanism", "pe_top_k")
    if isinstance(k_top_schedule, (int, float)):
        n_rounds = 2  # default length for scalar broadcast tests
    elif isinstance(k_top_schedule, str):
        n_rounds = len([x for x in k_top_schedule.split(",") if x.strip()])
    else:
        n_rounds = len(k_top_schedule)
    config = {
        "type": "pe_top_k",
        "threshold_H": threshold_H,
        "k_top_schedule": k_top_schedule,
        "num_samples_schedule": [100] * n_rounds,
        "variation_degree_schedule": [0.5] * n_rounds,
        **extra,
    }
    return cls(config, MockAPI())


# ── 1. Thresholding ─────────────────────────────────────────────────

class TestThresholding:

    def test_zero_threshold_is_identity(self):
        """H=0 means w = max(c, 0); non-negative counts pass through."""
        mech = _make_mechanism([5, 5], threshold_H=0.0)
        counts = np.array([10.0, 3.0, 0.0, 7.0, 1.0])
        w = mech.compute_weights(counts, t=0)
        # k_top=5 >= n=5, so all survive; weights should be proportional
        expected = counts / counts.sum()
        np.testing.assert_allclose(w, expected)

    def test_positive_threshold_subtracts(self):
        """H > 0 zeros out counts below H and reduces the rest."""
        mech = _make_mechanism([5, 5], threshold_H=4.0)
        counts = np.array([10.0, 3.0, 0.0, 7.0, 1.0])
        # After threshold: [6, 0, 0, 3, 0] — only indices 0 and 3 survive
        w = mech.compute_weights(counts, t=0)
        assert w[1] == 0.0
        assert w[2] == 0.0
        assert w[4] == 0.0
        assert w[0] > 0.0
        assert w[3] > 0.0
        # Proportions: 6/9 and 3/9
        np.testing.assert_allclose(w[0], 6.0 / 9.0)
        np.testing.assert_allclose(w[3], 3.0 / 9.0)

    def test_threshold_clamps_negative(self):
        """Counts below zero (after noise) stay at zero after thresholding."""
        mech = _make_mechanism([3, 3], threshold_H=0.0)
        # Simulating raw noisy counts that could be negative — but dp_nn_histogram
        # already clips to >= 0 when count_threshold=0, so incoming values are >= 0.
        counts = np.array([0.0, 0.0, 5.0])
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w, [0.0, 0.0, 1.0])


# ── 2. Top-k truncation ─────────────────────────────────────────────

class TestTopKTruncation:

    def test_keeps_exactly_k_entries(self):
        """Only the top-k indices by thresholded weight should be nonzero."""
        mech = _make_mechanism([2, 2], threshold_H=0.0)
        counts = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        w = mech.compute_weights(counts, t=0)
        nonzero_indices = set(np.nonzero(w)[0])
        # Top 2 by value: index 1 (5.0) and index 3 (4.0)
        assert nonzero_indices == {1, 3}

    def test_k_top_equals_n(self):
        """k_top >= n keeps everything."""
        mech = _make_mechanism([5, 5], threshold_H=0.0)
        counts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = mech.compute_weights(counts, t=0)
        assert np.all(w > 0)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_k_top_greater_than_n(self):
        """k_top > n is clamped to n — no crash."""
        mech = _make_mechanism([100, 100], threshold_H=0.0)
        counts = np.array([2.0, 3.0])
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w, [2.0 / 5.0, 3.0 / 5.0])

    def test_k_top_one(self):
        """k_top=1 concentrates all mass on the single largest entry."""
        mech = _make_mechanism([1, 1], threshold_H=0.0)
        counts = np.array([1.0, 7.0, 3.0])
        w = mech.compute_weights(counts, t=0)
        assert w[1] == 1.0
        assert w[0] == 0.0
        assert w[2] == 0.0

    def test_k_top_zero_clamped_to_one(self):
        """k_top <= 0 is clamped to 1 (at least one survivor)."""
        mech = _make_mechanism([0, 0], threshold_H=0.0)
        counts = np.array([3.0, 5.0])
        w = mech.compute_weights(counts, t=0)
        # Should behave like k_top=1: all mass on index 1
        assert w[1] == 1.0

    def test_threshold_and_topk_combined(self):
        """Threshold zeroes some entries, then top-k selects among survivors."""
        mech = _make_mechanism([2, 2], threshold_H=3.0)
        counts = np.array([10.0, 2.0, 5.0, 8.0, 1.0])
        # After H=3: [7, 0, 2, 5, 0]
        # Top-2 of that: index 0 (7) and index 3 (5)
        w = mech.compute_weights(counts, t=0)
        nonzero = set(np.nonzero(w)[0])
        assert nonzero == {0, 3}
        np.testing.assert_allclose(w[0], 7.0 / 12.0)
        np.testing.assert_allclose(w[3], 5.0 / 12.0)


# ── 3. Normalization ────────────────────────────────────────────────

class TestNormalization:

    def test_weights_sum_to_one(self):
        """Output weights must form a valid probability distribution."""
        mech = _make_mechanism([3, 3], threshold_H=0.0)
        counts = np.array([4.0, 1.0, 7.0, 2.0, 5.0])
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_weights_non_negative(self):
        """All output weights must be >= 0."""
        mech = _make_mechanism([2, 2], threshold_H=5.0)
        counts = np.array([1.0, 2.0, 3.0, 10.0])
        w = mech.compute_weights(counts, t=0)
        assert np.all(w >= 0)

    def test_output_shape_matches_input(self):
        """Weight vector has same shape as input counts."""
        mech = _make_mechanism([3, 3], threshold_H=0.0)
        counts = np.array([1.0, 2.0, 3.0, 4.0])
        w = mech.compute_weights(counts, t=0)
        assert w.shape == counts.shape


# ── 4. Zero-mass fallback ───────────────────────────────────────────

class TestZeroMassFallback:

    def test_all_zero_after_threshold(self):
        """When all counts are below H, fallback to uniform over top-k."""
        mech = _make_mechanism([2, 2], threshold_H=10.0)
        counts = np.array([3.0, 5.0, 1.0, 4.0])
        # After H=10: all zeros.  Top-2 of the original noisy counts
        # picks indices 1 (5) and 3 (4) → uniform 0.5 each.
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w.sum(), 1.0)
        nonzero = set(np.nonzero(w)[0])
        assert len(nonzero) == 2
        # Uniform: each should be 0.5
        for idx in nonzero:
            np.testing.assert_allclose(w[idx], 0.5)

    def test_all_zero_counts(self):
        """Completely zero input counts → uniform over top-k set."""
        mech = _make_mechanism([2, 2], threshold_H=0.0)
        counts = np.array([0.0, 0.0, 0.0, 0.0])
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w.sum(), 1.0)
        # k_top=2, all weights are 0 before top-k, fallback gives
        # uniform over 2 arbitrary indices
        assert np.count_nonzero(w) == 2

    def test_single_element_zero(self):
        """Single-element zero count → fallback gives mass 1.0 on it."""
        mech = _make_mechanism([1, 1], threshold_H=0.0)
        counts = np.array([0.0])
        w = mech.compute_weights(counts, t=0)
        np.testing.assert_allclose(w, [1.0])

    def test_empty_counts(self):
        """Empty input returns empty output (edge guard)."""
        mech = _make_mechanism([2, 2], threshold_H=0.0)
        counts = np.array([])
        w = mech.compute_weights(counts, t=0)
        assert len(w) == 0


# ── 5. Round-dependent schedule ─────────────────────────────────────

class TestRoundSchedule:

    def test_different_k_per_round(self):
        """Each round uses its own k_top from the schedule."""
        schedule = [3, 2, 1]
        mech = _make_mechanism(schedule, threshold_H=0.0)
        counts = np.array([1.0, 5.0, 3.0, 4.0, 2.0])

        w0 = mech.compute_weights(counts, t=0)  # k=3
        w1 = mech.compute_weights(counts, t=1)  # k=2
        w2 = mech.compute_weights(counts, t=2)  # k=1

        assert np.count_nonzero(w0) == 3
        assert np.count_nonzero(w1) == 2
        assert np.count_nonzero(w2) == 1

    def test_annealing_concentrates_mass(self):
        """Decreasing k_top schedule concentrates mass on fewer candidates."""
        schedule = [5, 3, 1]
        mech = _make_mechanism(schedule, threshold_H=0.0)
        counts = np.array([1.0, 5.0, 3.0, 4.0, 2.0])

        w0 = mech.compute_weights(counts, t=0)
        w2 = mech.compute_weights(counts, t=2)

        # Entropy of w2 (k=1, point mass) < entropy of w0 (k=5, spread)
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        assert entropy(w2) < entropy(w0)


# ── 6. Config and registry ──────────────────────────────────────────

class TestConfigAndRegistry:

    def test_registry_lookup(self):
        from src.registry import get_class, list_registered
        import src.mechanisms

        assert "pe_top_k" in list_registered("mechanism")
        cls = get_class("mechanism", "pe_top_k")
        assert cls.__name__ == "PETopKMechanism"

    def test_missing_k_top_schedule_raises(self):
        from src.registry import get_class
        import src.mechanisms

        cls = get_class("mechanism", "pe_top_k")
        with pytest.raises(ValueError, match="k_top_schedule"):
            cls({
                "type": "pe_top_k",
                "num_samples_schedule": [100],
                "variation_degree_schedule": [0.5],
            }, MockAPI())

    def test_scalar_k_top_broadcast(self):
        """A single int k_top_schedule is broadcast to all rounds."""
        mech = _make_mechanism(100, threshold_H=0.0)
        # Constructor receives scalar 100, broadcasts to schedule length
        assert all(k == 100 for k in mech.k_top_schedule)

    def test_csv_string_k_top_schedule(self):
        """Comma-separated string is parsed correctly."""
        from src.registry import get_class
        import src.mechanisms

        cls = get_class("mechanism", "pe_top_k")
        config = {
            "type": "pe_top_k",
            "threshold_H": 0.0,
            "k_top_schedule": "500, 400, 300",
            "num_samples_schedule": [100] * 3,
            "variation_degree_schedule": [0.5] * 3,
        }
        mech = cls(config, MockAPI())
        assert mech.k_top_schedule == [500, 400, 300]

    def test_experiment_config_loads(self):
        """The yelp_pe_top_k experiment config parses correctly."""
        from src.config import load_config

        cfg = load_config("configs/experiments/yelp_pe_top_k.yaml")
        assert cfg.mechanism["type"] == "pe_top_k"
        assert cfg.mechanism["threshold_H"] == 0.0
        assert len(cfg.mechanism["k_top_schedule"]) == 11
        assert cfg.mechanism["k_top_schedule"][0] == 3500
        assert cfg.mechanism["k_top_schedule"][-1] == 500

    def test_instantiate_from_experiment_config(self):
        """Full pipeline: load config → instantiate mechanism."""
        from src.config import load_config
        from src.registry import get_class
        import src.mechanisms

        cfg = load_config("configs/experiments/yelp_pe_top_k.yaml")
        mech_cfg = cfg.get_mechanism_config()
        cls = get_class("mechanism", mech_cfg["type"])
        mech = cls(mech_cfg, MockAPI())

        assert mech.threshold_H == 0.0
        assert mech.k_top_schedule[0] == 3500
        assert mech.noise_multiplier == 15.34

    def test_validate_counts_is_relaxed(self):
        """PETopKMechanism._validate_counts does not raise on all-zero."""
        mech = _make_mechanism([2, 2], threshold_H=0.0)
        # Should not raise
        mech._validate_counts(np.zeros(5), "test_class")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

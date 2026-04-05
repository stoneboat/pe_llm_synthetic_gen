"""Smoke tests for the refactored framework.

Tests config parsing, registry lookups, object instantiation,
and basic pipeline structure without running expensive generation.
"""

import os
import sys
import collections

# Ensure src/ is on the path (mimicking the original working directory approach)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ============================================================
# Test 1: YAML config loading and parsing
# ============================================================

def test_load_yelp_config():
    """Yelp experiment config loads and parses correctly."""
    from src.config import load_config

    cfg = load_config("configs/experiments/yelp_original.yaml")

    assert cfg.name == "yelp_original_non_dp"
    assert cfg.dataset["name"] == "yelp"
    assert cfg.mechanism["type"] == "original_aug_pe"
    assert cfg.noise_multiplier == 0.0
    assert cfg.num_iterations == 20
    assert cfg.num_syn_samples == 5000
    assert cfg.L == 7
    assert cfg.feature_extractor == "stsb-roberta-base-v2"

    # Check schedules were built
    assert len(cfg.num_samples_schedule) == 21  # 20 iterations + 1
    assert cfg.num_samples_schedule[0] == 5000 * 7  # num_syn_samples * L
    assert len(cfg.variation_degree_schedule) == 21

    # Check mechanism config got propagated
    mech = cfg.get_mechanism_config()
    assert mech["noise_multiplier"] == 0.0
    assert mech["L"] == 7
    assert mech["select_syn_mode"] == "rank"


def test_load_pubmed_config():
    """PubMed experiment config loads and parses correctly."""
    from src.config import load_config

    cfg = load_config("configs/experiments/pubmed_original.yaml")

    assert cfg.name == "pubmed_original_non_dp"
    assert cfg.dataset["name"] == "pubmed"
    assert cfg.num_syn_samples == 2000
    assert cfg.feature_extractor == "sentence-t5-base"
    assert cfg.num_iterations == 10
    assert len(cfg.num_samples_schedule) == 11


def test_load_dp_config():
    """DP config with nonzero noise loads correctly."""
    from src.config import load_config

    cfg = load_config("configs/experiments/yelp_original_dp_eps1.yaml")

    assert cfg.noise_multiplier == 15.34
    assert cfg.num_iterations == 10


def test_config_override():
    """CLI overrides are applied correctly."""
    from src.config import load_config

    cfg = load_config("configs/experiments/yelp_original.yaml",
                       overrides={"noise_multiplier": 8.0, "seed": 123})

    assert cfg.noise_multiplier == 8.0
    assert cfg.seed == 123


# ============================================================
# Test 2: Registry lookups
# ============================================================

def test_registry_datasets():
    """All expected datasets are registered."""
    from src.registry import get_class, list_registered
    import src.data_adapters  # triggers registration

    registered = list_registered("dataset")
    assert "yelp" in registered
    assert "pubmed" in registered
    assert "openreview" in registered

    cls = get_class("dataset", "yelp")
    assert cls.__name__ == "YelpDataset"


def test_registry_mechanisms():
    """All expected mechanisms are registered."""
    from src.registry import get_class, list_registered
    import src.mechanisms

    registered = list_registered("mechanism")
    assert "original_aug_pe" in registered


def test_registry_tasks():
    """All expected tasks are registered."""
    from src.registry import get_class, list_registered
    import src.tasks

    registered = list_registered("task")
    assert "embedding_metrics" in registered
    assert "classification" in registered


# ============================================================
# Test 3: Object instantiation
# ============================================================

def test_instantiate_yelp_dataset():
    """YelpDataset can be instantiated from config."""
    from src.registry import get_class
    import src.data_adapters

    cls = get_class("dataset", "yelp")
    ds = cls({"name": "yelp"})
    assert ds.name == "yelp"
    assert ds.train_data_file == "data/yelp/train.csv"
    assert ds.get_label_columns() == ["label1", "label2"]
    assert ds.get_csv_header() == ["text", "label1", "label2"]


def test_instantiate_pubmed_dataset():
    """PubMedDataset can be instantiated from config."""
    from src.registry import get_class
    import src.data_adapters

    cls = get_class("dataset", "pubmed")
    ds = cls({"name": "pubmed"})
    assert ds.name == "pubmed"
    assert ds.get_csv_header() == ["text"]


def test_instantiate_mechanism():
    """OriginalAugPEMechanism can be instantiated from config."""
    from src.registry import get_class
    import src.mechanisms

    cls = get_class("mechanism", "original_aug_pe")

    # Use a mock api (just needs to exist for instantiation)
    class MockAPI:
        pass

    mech = cls({
        "type": "original_aug_pe",
        "noise_multiplier": 0.0,
        "L": 7,
        "init_L": 7,
        "select_syn_mode": "rank",
        "num_samples_schedule": [35000] * 21,
        "variation_degree_schedule": [0.5] * 21,
    }, MockAPI())

    assert mech.combine_divide_L == 7
    assert mech.noise_multiplier == 0.0
    assert mech.select_syn_mode == "rank"


def test_instantiate_metric_task():
    """EmbeddingMetricsTask can be instantiated."""
    from src.registry import get_class
    import src.tasks

    cls = get_class("task", "embedding_metrics")
    task = cls({"type": "embedding_metrics", "embedding_model": "stsb-roberta-base-v2"})
    assert task.config["embedding_model"] == "stsb-roberta-base-v2"


# ============================================================
# Test 4: Mechanism weight computation
# ============================================================

def test_original_mechanism_weights():
    """Original mechanism returns counts unchanged."""
    import numpy as np
    from src.registry import get_class
    import src.mechanisms

    cls = get_class("mechanism", "original_aug_pe")

    class MockAPI:
        pass

    mech = cls({
        "num_samples_schedule": [100] * 2,
        "variation_degree_schedule": [0.5] * 2,
    }, MockAPI())

    counts = np.array([10.0, 5.0, 0.0, 3.0])
    weights = mech.compute_weights(counts, t=1)
    np.testing.assert_array_equal(weights, counts)


def test_original_mechanism_select_rank():
    """Rank selection picks top-count indices."""
    import numpy as np
    from src.registry import get_class
    import src.mechanisms

    cls = get_class("mechanism", "original_aug_pe")

    class MockAPI:
        pass

    mech = cls({
        "select_syn_mode": "rank",
        "num_samples_schedule": [100] * 2,
        "variation_degree_schedule": [0.5] * 2,
    }, MockAPI())

    counts = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
    indices = mech._select_survivors(counts, current_idx=0, selected_size=2)
    # Top 2 by count: index 1 (5.0) and index 3 (4.0)
    assert indices == [1, 3]


def test_original_mechanism_select_prob():
    """Prob selection samples with replacement."""
    import numpy as np
    from src.registry import get_class
    import src.mechanisms

    cls = get_class("mechanism", "original_aug_pe")

    class MockAPI:
        pass

    mech = cls({
        "select_syn_mode": "prob",
        "num_samples_schedule": [100] * 2,
        "variation_degree_schedule": [0.5] * 2,
    }, MockAPI())

    np.random.seed(42)
    counts = np.array([0.0, 0.0, 10.0])  # all weight on index 2
    indices = mech._select_survivors(counts, current_idx=0, selected_size=5)
    assert all(i == 2 for i in indices)


# ============================================================
# Test 5: Full config -> object pipeline
# ============================================================

def test_full_config_to_objects():
    """Load config and instantiate all objects end-to-end."""
    from src.config import load_config
    from src.registry import get_class
    import src.data_adapters
    import src.mechanisms
    import src.tasks

    cfg = load_config("configs/experiments/yelp_original.yaml")

    # Dataset
    ds_cls = get_class("dataset", cfg.dataset["name"])
    ds = ds_cls(cfg.dataset)
    assert ds.name == "yelp"

    # Mechanism
    mech_cfg = cfg.get_mechanism_config()
    mech_cls = get_class("mechanism", mech_cfg["type"])

    class MockAPI:
        pass

    mech = mech_cls(mech_cfg, MockAPI())
    assert mech.combine_divide_L == 7

    # Tasks
    for task_cfg in cfg.tasks:
        task_cls = get_class("task", task_cfg["type"])
        task = task_cls(task_cfg)
        assert task.name in ["embedding_metrics", "classification"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

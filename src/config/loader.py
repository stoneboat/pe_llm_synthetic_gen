"""YAML configuration loader.

Loads experiment configs with support for:
- Separate dataset, mechanism, and task config files
- Inline overrides in experiment configs
- Schedule parsing (num_samples, variation_degree)
- Default merging
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning empty dict if not found."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_schedule(value, num_rounds: int, dtype=int) -> list:
    """Parse a schedule value into a list of length num_rounds + 1.

    Accepts:
    - A single value (repeated for all rounds)
    - A comma-separated string
    - A list
    """
    if isinstance(value, list):
        return [dtype(v) for v in value]
    if isinstance(value, str):
        parts = [dtype(x.strip()) for x in value.split(',') if x.strip()]
        if len(parts) == 1:
            return parts * (num_rounds + 1)
        return parts
    # Single numeric value
    return [dtype(value)] * (num_rounds + 1)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Metadata
    name: str = "experiment"
    seed: int = 42
    result_folder: str = "result/experiment"
    tag: str = ""

    # Dataset config
    dataset: Dict[str, Any] = field(default_factory=dict)

    # Mechanism config
    mechanism: Dict[str, Any] = field(default_factory=dict)

    # API / model config
    api: Dict[str, Any] = field(default_factory=dict)

    # Evaluation tasks
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    # Privacy parameters
    noise_multiplier: float = 0.0
    num_nearest_neighbor: int = 1
    nn_mode: str = "L2"
    count_threshold: float = 0.0

    # PE loop parameters
    num_iterations: int = 10
    num_syn_samples: int = 5000
    L: int = 7
    init_L: int = 7
    lookahead_degree: int = 0
    select_syn_mode: str = "rank"
    save_syn_mode: str = "selected"
    variation_degree: float = 0.5
    compute_fid: bool = True
    donnot_keep_last_iter: bool = False
    lookahead_self: bool = False

    # Feature extractor
    feature_extractor: str = "stsb-roberta-base-v2"
    feature_extractor_batch_size: int = 1024

    # Checkpoint / resume
    data_checkpoint_path: str = ""
    data_checkpoint_step: int = -1

    # Wandb
    log_online: bool = False
    wandb_key: str = ""
    project: str = "text-API"

    # Computed schedules (populated by finalize())
    num_samples_schedule: List[int] = field(default_factory=list)
    variation_degree_schedule: List[float] = field(default_factory=list)

    def finalize(self):
        """Compute derived fields after loading."""
        num_samples_total = self.num_syn_samples * self.L

        # Build schedules
        if not self.num_samples_schedule:
            self.num_samples_schedule = [num_samples_total] * (self.num_iterations + 1)
        else:
            self.num_samples_schedule = _parse_schedule(
                self.num_samples_schedule, self.num_iterations, int)

        if not self.variation_degree_schedule:
            self.variation_degree_schedule = [self.variation_degree] * (self.num_iterations + 1)
        else:
            self.variation_degree_schedule = _parse_schedule(
                self.variation_degree_schedule, self.num_iterations, float)

        # Ensure schedule lengths match
        expected_len = self.num_iterations + 1
        if len(self.num_samples_schedule) == 1:
            self.num_samples_schedule = self.num_samples_schedule * expected_len
        if len(self.variation_degree_schedule) == 1:
            self.variation_degree_schedule = self.variation_degree_schedule * expected_len

        # Propagate shared params into mechanism config
        mech = self.mechanism
        mech.setdefault("noise_multiplier", self.noise_multiplier)
        mech.setdefault("num_nearest_neighbor", self.num_nearest_neighbor)
        mech.setdefault("nn_mode", self.nn_mode)
        mech.setdefault("count_threshold", self.count_threshold)
        mech.setdefault("select_syn_mode", self.select_syn_mode)
        mech.setdefault("save_syn_mode", self.save_syn_mode)
        mech.setdefault("L", self.L)
        mech.setdefault("init_L", self.init_L)
        mech.setdefault("lookahead_degree", self.lookahead_degree)
        mech.setdefault("lookahead_self", self.lookahead_self)
        mech.setdefault("donnot_keep_last_iter", self.donnot_keep_last_iter)
        mech.setdefault("compute_fid", self.compute_fid)
        mech.setdefault("num_samples_schedule", self.num_samples_schedule)
        mech.setdefault("variation_degree_schedule", self.variation_degree_schedule)
        mech.setdefault("feature_extractor", self.feature_extractor)
        mech.setdefault("feature_extractor_batch_size", self.feature_extractor_batch_size)

        # Disable wandb if no key
        if not self.wandb_key:
            self.log_online = False

    def get_mechanism_config(self) -> dict:
        """Return the full mechanism config dict."""
        return self.mechanism

    def get_dataset_config(self) -> dict:
        """Return the full dataset config dict."""
        return self.dataset

    def get_api_config(self) -> dict:
        """Return the full API config dict."""
        return self.api


def load_config(config_path: str, overrides: Optional[Dict] = None) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    The YAML may reference external dataset/mechanism/task configs via
    `_include` keys, which are loaded and merged.

    Args:
        config_path: path to the experiment YAML.
        overrides: optional dict of overrides to apply last.

    Returns:
        Fully populated ExperimentConfig.
    """
    raw = _load_yaml(config_path)

    # Resolve includes for dataset, mechanism, tasks
    config_dir = os.path.dirname(config_path)
    root_dir = os.path.dirname(os.path.dirname(config_dir))  # project root

    for section in ['dataset', 'mechanism']:
        if isinstance(raw.get(section), dict):
            include = raw[section].pop('_include', None)
            if include:
                include_path = os.path.join(root_dir, include) if not os.path.isabs(include) else include
                base = _load_yaml(include_path)
                raw[section] = _deep_merge(base, raw[section])

    if 'tasks' in raw and isinstance(raw['tasks'], list):
        resolved_tasks = []
        for task in raw['tasks']:
            if isinstance(task, dict):
                include = task.pop('_include', None)
                if include:
                    include_path = os.path.join(root_dir, include) if not os.path.isabs(include) else include
                    base = _load_yaml(include_path)
                    task = _deep_merge(base, task)
            resolved_tasks.append(task)
        raw['tasks'] = resolved_tasks

    # Apply overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Build ExperimentConfig
    cfg = ExperimentConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg.finalize()
    return cfg

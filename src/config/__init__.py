"""Configuration layer: YAML-based experiment configuration."""

from .loader import load_config, ExperimentConfig

__all__ = ["load_config", "ExperimentConfig"]

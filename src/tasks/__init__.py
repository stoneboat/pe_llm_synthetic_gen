"""Task layer: modular downstream utility evaluation tasks."""

from .base import UtilityTask

# Lazy registration to avoid pulling in heavy dependencies at import time.
def _ensure_registered():
    from .embedding_metrics import EmbeddingMetricsTask  # noqa: F401
    from .classification import ClassificationTask  # noqa: F401

_ensure_registered()

__all__ = ["UtilityTask"]

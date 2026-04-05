"""Dataset layer: modular dataset adapters for DP synthetic text experiments."""

from .base import DatasetAdapter

# Lazy registration: import submodules to trigger @register decorators.
def _ensure_registered():
    from .yelp import YelpDataset  # noqa: F401
    from .pubmed import PubMedDataset  # noqa: F401
    from .openreview import OpenReviewDataset  # noqa: F401

_ensure_registered()

__all__ = ["DatasetAdapter"]

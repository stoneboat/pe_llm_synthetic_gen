"""Mechanism layer: modular DP synthetic text generation mechanisms."""

from .base import Mechanism

# Lazy import to avoid pulling in faiss/torch at module load time.
# The class is registered via decorator when first imported.
def _ensure_registered():
    from .original_aug_pe import OriginalAugPEMechanism  # noqa: F401
    from .pe_top_k import PETopKMechanism  # noqa: F401

_ensure_registered()

__all__ = ["Mechanism"]

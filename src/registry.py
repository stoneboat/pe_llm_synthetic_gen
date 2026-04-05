"""Simple registry pattern for datasets, mechanisms, and tasks."""

from typing import Dict, Type, TypeVar

T = TypeVar("T")

_REGISTRIES: Dict[str, Dict[str, Type]] = {}


def register(kind: str, name: str):
    """Decorator to register a class under a given kind and name.

    Usage:
        @register("dataset", "yelp")
        class YelpDataset(DatasetAdapter): ...
    """
    def decorator(cls):
        if kind not in _REGISTRIES:
            _REGISTRIES[kind] = {}
        _REGISTRIES[kind][name] = cls
        cls._registry_name = name
        return cls
    return decorator


def get_class(kind: str, name: str) -> Type:
    """Look up a registered class by kind and name."""
    if kind not in _REGISTRIES or name not in _REGISTRIES[kind]:
        available = list(_REGISTRIES.get(kind, {}).keys())
        raise KeyError(f"No {kind} registered as '{name}'. Available: {available}")
    return _REGISTRIES[kind][name]


def list_registered(kind: str) -> list:
    """List all registered names for a given kind."""
    return list(_REGISTRIES.get(kind, {}).keys())

"""Runtime environment helpers for stable local execution."""

from __future__ import annotations

import os
import site
from pathlib import Path


def _candidate_cuda_lib_dirs() -> list[str]:
    candidates: list[str] = []
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
    except Exception:
        pass

    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue

        preferred = [
            nvidia_root / "cu13" / "lib",
            nvidia_root / "cudnn" / "lib",
            nvidia_root / "cuda_nvrtc" / "lib",
        ]
        for lib_dir in preferred:
            if lib_dir.is_dir():
                lib_dir_str = str(lib_dir)
                if lib_dir_str not in seen:
                    candidates.append(lib_dir_str)
                    seen.add(lib_dir_str)

        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dir_str = str(lib_dir)
                if lib_dir_str not in seen:
                    candidates.append(lib_dir_str)
                    seen.add(lib_dir_str)
    return candidates


def configure_torch_runtime() -> None:
    """Prefer PyTorch's bundled CUDA libraries over system CUDA paths.

    This avoids loading incompatible system copies of cuDNN/NVRTC when the
    Python environment already contains the matching CUDA runtime libraries.
    """

    bundled_dirs = _candidate_cuda_lib_dirs()
    if not bundled_dirs:
        return

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]

    def _is_conflicting_cuda_path(path: str) -> bool:
        lower = path.lower()
        return "/usr/local/cuda" in lower or "cudnn" in lower or "cuda" in lower and "site-packages/nvidia" not in lower

    sanitized_current = [
        p for p in current
        if p not in bundled_dirs and not _is_conflicting_cuda_path(p)
    ]
    merged = bundled_dirs + sanitized_current
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

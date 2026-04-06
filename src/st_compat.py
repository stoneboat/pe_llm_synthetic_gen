"""Compatibility helpers for importing sentence-transformers.

This repo was originally developed against older sentence-transformers /
transformers combinations. Some newer sentence-transformers releases
expect `transformers.is_torch_npu_available`, which is absent in
transformers 4.30.x. We provide a small fallback so the import succeeds.
"""


def ensure_sentence_transformers_compat():
    import transformers

    if not hasattr(transformers, "is_torch_npu_available"):
        def _is_torch_npu_available():
            return False

        transformers.is_torch_npu_available = _is_torch_npu_available

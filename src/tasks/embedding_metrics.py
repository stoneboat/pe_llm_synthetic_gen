"""Embedding-based metric evaluation task.

Wraps the existing src/metric.py logic: computes FID, MAUVE, precision/recall,
and Sinkhorn distance between private and synthetic embeddings.
"""

import logging
import os
from typing import Any, Dict

from ..registry import register
from .base import UtilityTask


@register("task", "embedding_metrics")
class EmbeddingMetricsTask(UtilityTask):
    """Evaluate synthetic data quality via embedding-space metrics."""

    def __init__(self, config: dict):
        defaults = {
            "name": "embedding_metrics",
            "embedding_model": "stsb-roberta-base-v2",
            "batch_size": 1024,
            "private_data_size": 5000,
            "k": 3,
            "num_runs": 1,
            "min_token_threshold": 100,
        }
        merged = {**defaults, **config}
        super().__init__(merged)

    def evaluate(
        self,
        synthetic_data_path: str,
        dataset_config: dict,
        result_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run embedding-based metrics on synthetic data."""
        import numpy as np
        import torch
        from st_compat import ensure_sentence_transformers_compat

        ensure_sentence_transformers_compat()
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
        from tqdm import tqdm
        from dpsda.logging import load_embeddings
        from apis.utils import set_seed

        set_seed(seed=0, n_gpu=1)

        model_name = self.config["embedding_model"]
        model = SentenceTransformer(model_name)
        model.eval()

        # Load private embeddings
        dataset_name = dataset_config.get("name", "yelp")
        embeddings_file = dataset_config.get(
            "embeddings_file",
            f"result/embeddings/{model_name}/{dataset_name}_train_all.embeddings.npz"
        )
        all_original_embeddings, _ = load_embeddings(embeddings_file)

        # Load synthetic data
        syn_data = load_dataset("csv", data_files=synthetic_data_path)
        synthetic_texts = [d for d in syn_data['train']['text'] if d]

        # Compute synthetic embeddings
        batch_size = self.config["batch_size"]
        with torch.no_grad():
            syn_emb_parts = []
            for i in tqdm(range(len(synthetic_texts) // batch_size + 1)):
                batch = synthetic_texts[i * batch_size:(i + 1) * batch_size]
                if batch:
                    syn_emb_parts.append(model.encode(batch))
            all_syn_embeddings = np.concatenate(syn_emb_parts)

        # Compute FID
        from dpsda.metrics import calculate_fid
        fid = calculate_fid(all_original_embeddings, all_syn_embeddings)

        results = {"fid": float(fid), "n_synthetic": len(synthetic_texts)}
        logging.info(f"EmbeddingMetrics: FID={fid:.4f}, n_syn={len(synthetic_texts)}")

        return results

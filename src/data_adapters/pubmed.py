"""PubMed dataset adapter."""

import collections
import csv
import logging
from typing import List, Optional

import numpy as np
from datasets import load_dataset

from ..registry import register
from .base import DatasetAdapter, DatasetResult


def _load_dataset_with_special(data_file: str, gen: bool):
    if gen:
        try:
            return load_dataset("csv", data_files=data_file,
                                quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
        except Exception:
            return load_dataset("csv", data_files=data_file)
    return load_dataset("csv", data_files=data_file)


@register("dataset", "pubmed")
class PubMedDataset(DatasetAdapter):
    """PubMed abstract dataset with single pseudo-label."""

    def __init__(self, config: dict):
        defaults = {
            "name": "pubmed",
            "train_data_file": "data/pubmed/train.csv",
            "dev_data_file": "data/pubmed/dev.csv",
            "test_data_file": "data/pubmed/test.csv",
            "embeddings_file": "",
            "embedding_model": "sentence-t5-base",
            "variation_type": "pubmed_rephrase_tone",
            "subcategory_source": "data/pubmed/writers.txt",
        }
        merged = {**defaults, **config}
        super().__init__(merged)

    def load_data(self, data_file=None, num_samples=-1,
                  subsample_one_class=False, gen=False) -> DatasetResult:
        data_file = data_file or self.train_data_file
        logging.info(f"Loading PubMed data from {data_file}")

        raw_datasets = _load_dataset_with_special(data_file, gen)

        # PubMed subsampling: random sample if requested
        if not subsample_one_class and num_samples >= 0:
            training_dataset = raw_datasets['train']
            indices = list(range(len(training_dataset)))
            sample_indices = np.random.choice(indices, size=num_samples, replace=False)
            np.random.shuffle(sample_indices)
            raw_datasets['train'] = training_dataset.select(sample_indices)
        elif subsample_one_class:
            # PubMed has no real class structure; subsample is just random
            training_dataset = raw_datasets['train']
            indices = list(range(len(training_dataset)))
            if num_samples >= 0:
                sample_indices = np.random.choice(indices, size=num_samples, replace=False)
                raw_datasets['train'] = training_dataset.select(sample_indices)

        prompt_counter = collections.Counter()
        prompt_indexer = {}
        train_data = []
        train_labels = []

        for i, line in enumerate(raw_datasets['train']):
            prompt = "pubmed"
            prompt_counter[prompt] += 1
            if prompt not in prompt_indexer:
                prompt_indexer[prompt] = [i]
            else:
                prompt_indexer[prompt].append(i)
            train_data.append(line['text'])
            train_labels.append(prompt)

        return DatasetResult(
            samples=train_data,
            labels=train_labels,
            label_counter=prompt_counter,
            label_indexer=prompt_indexer,
        )

    def get_label_columns(self) -> List[str]:
        return []  # PubMed uses a single pseudo-label

    def get_csv_header(self) -> List[str]:
        return ['text']

    def format_sample_row(self, text: str, label: str) -> List[str]:
        return [text]

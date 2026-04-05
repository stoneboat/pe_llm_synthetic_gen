"""OpenReview dataset adapter."""

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


def _sample_dataset_openreview(dataset, sample_size: int, subsample_one_class: bool):
    if not subsample_one_class and sample_size < 0:
        return dataset
    training_dataset = dataset['train']

    if subsample_one_class:
        area = "Area: Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)"
        recommendation = "Recommendation: 8: accept, good paper"
        indices = np.where(
            (np.array(training_dataset['label1']) == area) &
            (np.array(training_dataset['label2']) == recommendation)
        )[0]
        if sample_size < 0:
            sample_indices = indices
        else:
            sample_indices = np.random.choice(indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
    else:
        indices = list(range(len(training_dataset)))
        sample_indices = np.random.choice(indices, size=sample_size, replace=False)
        np.random.shuffle(sample_indices)

    training_dataset = training_dataset.select(sample_indices)
    dataset['train'] = training_dataset
    return dataset


@register("dataset", "openreview")
class OpenReviewDataset(DatasetAdapter):
    """OpenReview paper review dataset with area + recommendation labels."""

    def __init__(self, config: dict):
        defaults = {
            "name": "openreview",
            "train_data_file": "data/openreview/iclr23_reviews_train.csv",
            "dev_data_file": "",
            "test_data_file": "",
            "embeddings_file": "",
            "embedding_model": "stsb-roberta-base-v2",
            "variation_type": "openreview_rephrase_tone",
            "subcategory_source": "data/openreview/writers.txt",
        }
        merged = {**defaults, **config}
        super().__init__(merged)

    def load_data(self, data_file=None, num_samples=-1,
                  subsample_one_class=False, gen=False) -> DatasetResult:
        data_file = data_file or self.train_data_file
        logging.info(f"Loading OpenReview data from {data_file}")

        raw_datasets = _load_dataset_with_special(data_file, gen)
        original_data = _sample_dataset_openreview(
            raw_datasets, sample_size=num_samples,
            subsample_one_class=subsample_one_class)

        prompt_counter = collections.Counter()
        prompt_indexer = {}
        train_data = []
        train_labels = []

        for i, line in enumerate(original_data['train']):
            prompt = f"{line['label1']}\t{line['label2']}"
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
        return ['label1', 'label2']

    def get_csv_header(self) -> List[str]:
        return ['text', 'label1', 'label2']

    def format_sample_row(self, text: str, label: str) -> List[str]:
        labels = label.strip().split("\t")
        return [text] + labels

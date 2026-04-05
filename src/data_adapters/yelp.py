"""Yelp dataset adapter."""

import collections
import csv
import logging
from typing import List, Optional

import numpy as np
from datasets import load_dataset

from ..registry import register
from .base import DatasetAdapter, DatasetResult


def _load_dataset_with_special(data_file: str, gen: bool):
    """Load CSV dataset, handling special characters in generated text."""
    if gen:
        try:
            return load_dataset("csv", data_files=data_file,
                                quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
        except Exception:
            return load_dataset("csv", data_files=data_file)
    return load_dataset("csv", data_files=data_file)


def _sample_dataset(dataset, label_column_name: str, sample_size: int,
                    subsample_one_class: bool):
    """Subsample dataset, preserving label distribution."""
    if not subsample_one_class and sample_size < 0:
        return dataset
    training_dataset = dataset['train']
    sample_indices = []

    if subsample_one_class:
        label1 = 'Business Category: Restaurants'
        label2 = 'Review Stars: 5.0'
        indices = np.where(
            (np.array(training_dataset['label1']) == label1) &
            (np.array(training_dataset['label2']) == label2)
        )[0]
        if sample_size < 0:
            sample_indices = indices
        else:
            sample_indices = np.random.choice(indices, size=sample_size, replace=False)
            np.random.shuffle(sample_indices)
    else:
        label_list = training_dataset.unique(label_column_name)
        for label in label_list:
            indices = np.where(
                np.array(training_dataset[label_column_name]) == label
            )[0]
            sample_num = round(sample_size * (len(indices) / len(training_dataset)))
            sample_indices.append(np.random.choice(indices, size=sample_num, replace=False))
        sample_indices = np.concatenate(sample_indices)
        np.random.shuffle(sample_indices)

    training_dataset = training_dataset.select(sample_indices)
    dataset['train'] = training_dataset
    return dataset


@register("dataset", "yelp")
class YelpDataset(DatasetAdapter):
    """Yelp review dataset with two-level label structure."""

    def __init__(self, config: dict):
        defaults = {
            "name": "yelp",
            "train_data_file": "data/yelp/train.csv",
            "dev_data_file": "data/yelp/dev.csv",
            "test_data_file": "data/yelp/test.csv",
            "embeddings_file": "",
            "embedding_model": "stsb-roberta-base-v2",
            "variation_type": "yelp_rephrase_tone",
            "subcategory_source": "data/yelp/subcategories",
        }
        merged = {**defaults, **config}
        super().__init__(merged)

    def load_data(self, data_file=None, num_samples=-1,
                  subsample_one_class=False, gen=False) -> DatasetResult:
        data_file = data_file or self.train_data_file
        logging.info(f"Loading Yelp data from {data_file}")

        raw_datasets = _load_dataset_with_special(data_file, gen)
        original_data = _sample_dataset(
            raw_datasets, label_column_name='label1',
            sample_size=num_samples, subsample_one_class=subsample_one_class)

        prompt_counter = collections.Counter()
        prompt_indexer = {}
        label_columns = ['label1', 'label2']

        for i, line in enumerate(original_data['train']):
            prompt = "\t".join([line[idx] for idx in label_columns])
            prompt_counter[prompt] += 1
            if prompt not in prompt_indexer:
                prompt_indexer[prompt] = [i]
            else:
                prompt_indexer[prompt].append(i)

        train_data = [d for d in original_data['train']['text']]
        train_labels = [
            "\t".join([line[idx] for idx in label_columns])
            for line in original_data['train']
        ]

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

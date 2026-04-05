"""Downstream classification task.

Wraps the existing src/utility_eval/run_classification.py as a task.
This task trains a classifier on synthetic data and evaluates on real test data.
"""

import logging
import os
import subprocess
import sys
from typing import Any, Dict

from ..registry import register
from .base import UtilityTask


@register("task", "classification")
class ClassificationTask(UtilityTask):
    """Train a classifier on synthetic data, evaluate on real test/dev data."""

    def __init__(self, config: dict):
        defaults = {
            "name": "classification",
            "model_name": "roberta-base",
            "max_seq_length": 512,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "num_epochs": 5,
            "label_column": "label2",
            "seed": 0,
            "clean_dataset": True,
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
        """Run downstream classification evaluation.

        Delegates to run_classification.py via subprocess for isolation.
        """
        dev_file = dataset_config.get("dev_data_file", "")
        test_file = dataset_config.get("test_data_file", "")

        if not dev_file or not test_file:
            logging.warning("Classification task requires dev and test files.")
            return {"error": "missing dev/test files"}

        output_dir = os.path.join(result_dir, "classification")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "src/utility_eval/run_classification.py",
            "--report_to", "none",
            "--model_name_or_path", self.config["model_name"],
            "--output_dir", output_dir,
            "--train_file", synthetic_data_path,
            "--validation_file", dev_file,
            "--test_file", test_file,
            "--do_train", "--do_eval", "--do_predict",
            "--max_seq_length", str(self.config["max_seq_length"]),
            "--per_device_train_batch_size", str(self.config["batch_size"]),
            "--per_device_eval_batch_size", str(self.config["batch_size"]),
            "--learning_rate", str(self.config["learning_rate"]),
            "--num_train_epochs", str(self.config["num_epochs"]),
            "--overwrite_output_dir", "--overwrite_cache",
            "--save_strategy", "epoch", "--save_total_limit", "2",
            "--load_best_model_at_end",
            "--logging_strategy", "epoch",
            "--seed", str(self.config["seed"]),
            "--metric_for_best_model", "accuracy_all",
            "--greater_is_better", "True",
            "--evaluation_strategy", "epoch",
            "--label_column_name", self.config["label_column"],
        ]

        if self.config.get("clean_dataset", True):
            cmd.append("--clean_dataset")

        logging.info(f"Running classification: {' '.join(cmd[:10])}...")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            logging.error(f"Classification failed: {result.stderr[:500]}")
            return {"error": result.stderr[:500]}

        return {"output_dir": output_dir, "status": "completed"}

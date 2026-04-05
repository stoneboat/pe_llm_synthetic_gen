"""Config-driven metric evaluation entry point.

Usage:
    python -m src.cli.evaluate --config configs/experiments/yelp_original.yaml
    python -m src.cli.evaluate --config configs/experiments/yelp_original.yaml --iteration 10
"""

import argparse
import logging
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_evaluate(config, iteration: int = -1):
    """Run embedding-based metric evaluation from an ExperimentConfig."""
    from src.registry import get_class
    import src.tasks

    dataset_cfg = config.get_dataset_config()

    # Find which iterations to evaluate
    if iteration >= 0:
        iterations = [iteration]
    else:
        # Evaluate all available iterations
        iterations = []
        for i in range(config.num_iterations + 1):
            samples_path = os.path.join(config.result_folder, str(i), 'samples.csv')
            if os.path.isfile(samples_path):
                iterations.append(i)

    if not iterations:
        logging.warning(f"No synthetic data found in {config.result_folder}")
        return

    # Run metric tasks
    for task_cfg in config.tasks:
        task_type = task_cfg.get("type", "")
        if task_type != "embedding_metrics":
            continue

        task_cls = get_class("task", task_type)
        # Merge dataset embedding info into task config
        task_cfg.setdefault("embedding_model",
                            dataset_cfg.get("embedding_model", config.feature_extractor))
        task = task_cls(task_cfg)

        for it in iterations:
            samples_path = os.path.join(config.result_folder, str(it), 'samples.csv')
            if not os.path.isfile(samples_path):
                continue
            result_dir = os.path.join(config.result_folder, str(it))
            logging.info(f"Evaluating iteration {it}: {samples_path}")
            metrics = task.evaluate(
                synthetic_data_path=samples_path,
                dataset_config=dataset_cfg,
                result_dir=result_dir,
            )
            logging.info(f"Iteration {it} metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic data metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Specific iteration to evaluate (-1 for all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    from src.config import load_config
    config = load_config(args.config)
    run_evaluate(config, iteration=args.iteration)


if __name__ == "__main__":
    main()

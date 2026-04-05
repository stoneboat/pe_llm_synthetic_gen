"""Config-driven downstream utility evaluation entry point.

Usage:
    python -m src.cli.downstream --config configs/experiments/yelp_original.yaml
    python -m src.cli.downstream --config configs/experiments/yelp_original.yaml --iteration 10
"""

import argparse
import logging
import os
import sys


def run_downstream(config, iteration: int = -1):
    """Run downstream evaluation tasks from an ExperimentConfig."""
    from src.registry import get_class
    import src.tasks

    dataset_cfg = config.get_dataset_config()

    # Find which iterations to evaluate
    if iteration >= 0:
        iterations = [iteration]
    else:
        iterations = []
        for i in range(config.num_iterations + 1):
            samples_path = os.path.join(config.result_folder, str(i), 'samples.csv')
            if os.path.isfile(samples_path):
                iterations.append(i)

    if not iterations:
        logging.warning(f"No synthetic data found in {config.result_folder}")
        return

    # Run non-metric tasks (classification, etc.)
    for task_cfg in config.tasks:
        task_type = task_cfg.get("type", "")
        if task_type == "embedding_metrics":
            continue  # handled by evaluate

        task_cls = get_class("task", task_type)
        task = task_cls(task_cfg)

        for it in iterations:
            samples_path = os.path.join(config.result_folder, str(it), 'samples.csv')
            if not os.path.isfile(samples_path):
                continue
            result_dir = os.path.join(config.result_folder, str(it))
            logging.info(f"Running {task_type} on iteration {it}")
            metrics = task.evaluate(
                synthetic_data_path=samples_path,
                dataset_config=dataset_cfg,
                result_dir=result_dir,
            )
            logging.info(f"Iteration {it} {task_type}: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Run downstream utility evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Specific iteration to evaluate (-1 for all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    from src.config import load_config
    config = load_config(args.config)
    run_downstream(config, iteration=args.iteration)


if __name__ == "__main__":
    main()

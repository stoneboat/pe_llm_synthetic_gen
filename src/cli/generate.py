"""Config-driven generation entry point.

Usage:
    python -m src.cli.generate --config configs/experiments/yelp_original.yaml
    python -m src.cli.generate --config configs/experiments/yelp_original.yaml --override noise_multiplier=15.34
"""

import argparse
import logging
import os
import sys

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_api_from_config(api_config: dict):
    """Construct the text generation API from config dict.

    Builds CLI-style args and delegates to the existing HFAPI constructor
    to preserve full backward compatibility with the original API code.
    """
    api_type = api_config.get("type", "HFGPT")

    if api_type == "HFGPT":
        from apis import get_api_class_from_name
        api_class = get_api_class_from_name("HFGPT")

        # Build the CLI arg list that HFAPI.from_command_line_args expects
        cli_args = [
            "--model_type", str(api_config.get("model_type", "gpt2")),
            "--length", str(api_config.get("max_new_tokens", 64)),
            "--temperature", str(api_config.get("temperature", 1.0)),
            "--top_k", str(api_config.get("top_k", 50)),
            "--top_p", str(api_config.get("top_p", 0.9)),
            "--repetition_penalty", str(api_config.get("repetition_penalty", 1.0)),
            "--random_sampling_batch_size", str(api_config.get("random_sampling_batch_size", 64)),
            "--variation_batch_size", str(api_config.get("variation_batch_size", 256)),
            "--seed", str(api_config.get("seed", 42)),
            "--variation_type", str(api_config.get("variation_type", "yelp_rephrase_tone")),
            "--mlm_probability", str(api_config.get("mlm_probability", 0.5)),
        ]
        if api_config.get("do_sample", True):
            cli_args.append("--do_sample")
        if api_config.get("fp16", True):
            cli_args.append("--fp16")
        if api_config.get("use_subcategory", True):
            cli_args.append("--use_subcategory")
        if api_config.get("no_cuda", False):
            cli_args.append("--no_cuda")
        if api_config.get("dry_run", False):
            cli_args.append("--dry_run")

        api = api_class.from_command_line_args(cli_args)
        return api
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


def run_generate(config):
    """Run the generation pipeline from an ExperimentConfig."""
    from dpsda.logging import (
        setup_logging, load_embeddings, log_samples,
        log_prompt_generation, compute_fid as compute_fid_fn
    )
    from dpsda.feature_extractor import extract_features
    from src.registry import get_class

    # Import to trigger registration
    import src.data_adapters
    import src.mechanisms

    os.makedirs(config.result_folder, exist_ok=True)
    setup_logging(os.path.join(config.result_folder, 'log.log'))
    logging.info(f"Experiment config: {config.name}")
    logging.info(f"Result folder: {config.result_folder}")

    # Check if already finished
    if config.data_checkpoint_step >= len(config.num_samples_schedule) - 1:
        logging.info(f"Already finished {config.data_checkpoint_step} PE iterations!")
        return

    # 1. Instantiate dataset adapter
    dataset_cfg = config.get_dataset_config()
    dataset_cls = get_class("dataset", dataset_cfg["name"])
    dataset = dataset_cls(dataset_cfg)

    # 2. Load private data
    private_data = dataset.load_data(
        num_samples=dataset_cfg.get("num_private_samples", -1),
        subsample_one_class=dataset_cfg.get("subsample_one_class", False),
    )

    if dataset_cfg.get("num_private_samples", -1) > 0:
        log_samples(
            samples=private_data.samples,
            additional_info=private_data.labels,
            folder=f'{config.result_folder}/train',
        )

    private_classes = list(private_data.label_counter.keys())
    logging.info(f"Private: {len(private_classes)} classes, {len(private_data.samples)} samples")

    # 3. Load or compute private embeddings
    embeddings_file = dataset_cfg.get("embeddings_file", "")
    if embeddings_file and os.path.exists(embeddings_file):
        logging.info(f"Loading cached embeddings from {embeddings_file}")
        all_private_features, _ = load_embeddings(embeddings_file)
        # Truncate data to match embeddings if needed
        private_data.samples = private_data.samples[:len(all_private_features)]
    else:
        logging.info("Computing private embeddings")
        all_private_features = extract_features(
            data=private_data.samples,
            batch_size=config.feature_extractor_batch_size,
            model_name=config.feature_extractor,
        )

    # 4. Build API
    api = build_api_from_config(config.get_api_config())

    # 5. Instantiate mechanism
    mech_cfg = config.get_mechanism_config()
    mech_type = mech_cfg.get("type", "original_aug_pe")
    mech_cls = get_class("mechanism", mech_type)
    mechanism = mech_cls(mech_cfg, api)

    # 6. Generate initial synthetic samples (or load checkpoint)
    if config.data_checkpoint_path:
        logging.info(f"Loading checkpoint from {config.data_checkpoint_path}")
        ckpt_data = dataset.load_data(
            data_file=config.data_checkpoint_path, num_samples=-1, gen=True)
        seed_syn_samples = ckpt_data.samples
        seed_additional_info = ckpt_data.labels
        sync_labels_counter = ckpt_data.label_counter

        if config.data_checkpoint_step < 0:
            raise ValueError("data_checkpoint_step must be >= 0 when using checkpoint")
        start_t = config.data_checkpoint_step + 1
    else:
        logging.info("Generating initial synthetic samples")
        num_initial = config.num_samples_schedule[0]
        seed_syn_samples, seed_additional_info, sync_labels_counter = mechanism.generate_initial(
            num_samples=num_initial,
            label_counter=private_data.label_counter,
        )

        # Save initial prompt/generation pairs
        os.makedirs(f'{config.result_folder}/0', exist_ok=True)
        log_prompt_generation(
            fname=f'{config.result_folder}/0/prompt_generation.jsonl',
            prompts=[""] * len(seed_syn_samples),
            generations=np.stack([seed_syn_samples], axis=1),
        )

        if config.data_checkpoint_step >= 0:
            logging.info("Ignoring data_checkpoint_step for fresh generation")
        start_t = 1

    # Save initial samples
    log_samples(
        samples=seed_syn_samples,
        additional_info=seed_additional_info,
        folder=f'{config.result_folder}/{start_t - 1}',
    )

    if config.compute_fid:
        syn_features = extract_features(
            data=seed_syn_samples,
            batch_size=config.feature_extractor_batch_size,
            model_name=config.feature_extractor,
        )
        compute_fid_fn(
            syn_features, all_private_features, config.feature_extractor,
            folder=config.result_folder, step=start_t - 1, log_online=config.log_online)

    # 7. Expand initial pool if L > 1
    if mechanism.init_combine_divide_L > 1:
        # Try to load existing expanded pool from checkpoint
        if config.data_checkpoint_path:
            parent_dir = os.path.dirname(config.data_checkpoint_path)
            all_data_ckpt = os.path.join(parent_dir + "_all", 'samples.csv')
            if os.path.isfile(all_data_ckpt):
                logging.info(f"Loading expanded pool from {all_data_ckpt}")
                expanded = dataset.load_data(data_file=all_data_ckpt, num_samples=-1, gen=True)
                syn_samples = expanded.samples
                additional_info = expanded.labels
                sync_labels_counter = expanded.label_counter
            else:
                syn_samples, additional_info, sync_labels_counter = mechanism.expand_initial_pool(
                    seed_syn_samples, seed_additional_info, sync_labels_counter, private_classes)
                log_samples(samples=syn_samples, additional_info=additional_info,
                            folder=f'{config.result_folder}/-1')
        else:
            syn_samples, additional_info, sync_labels_counter = mechanism.expand_initial_pool(
                seed_syn_samples, seed_additional_info, sync_labels_counter, private_classes)
            log_samples(samples=syn_samples, additional_info=additional_info,
                        folder=f'{config.result_folder}/-1')
    else:
        syn_samples = seed_syn_samples
        additional_info = seed_additional_info

    logging.info(f"Initial pool: {len(syn_samples)} samples, {len(additional_info)} labels")

    # 8. Main PE loop
    for t in range(start_t, len(config.num_samples_schedule)):
        result = mechanism.run_round(
            t=t,
            syn_samples=syn_samples,
            additional_info=additional_info,
            sync_labels_counter=sync_labels_counter,
            private_classes=private_classes,
            all_private_features=all_private_features,
            private_labels_indexer=private_data.label_indexer,
            result_folder=config.result_folder,
        )

        # Update for next round
        syn_samples = result.next_round_samples
        additional_info = result.next_round_labels
        sync_labels_counter = result.updated_label_counter

        # Save round outputs
        log_samples(
            samples=result.selected_samples,
            additional_info=result.selected_labels,
            folder=f'{config.result_folder}/{t}',
        )

        if config.compute_fid:
            syn_features = extract_features(
                data=result.selected_samples,
                batch_size=config.feature_extractor_batch_size,
                model_name=config.feature_extractor,
            )
            compute_fid_fn(
                syn_features, all_private_features, config.feature_extractor,
                folder=config.result_folder, step=t, log_online=config.log_online)

        # Save full next-round pool
        log_samples(
            samples=syn_samples,
            additional_info=additional_info,
            folder=f'{config.result_folder}/{t}_all',
        )

    logging.info("Generation complete.")


def main():
    parser = argparse.ArgumentParser(description="DP synthetic text generation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Key=value overrides (e.g. noise_multiplier=15.34)")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for ov in args.override:
        key, _, val = ov.partition("=")
        # Try to parse as number
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        overrides[key] = val

    from src.config import load_config
    config = load_config(args.config, overrides=overrides if overrides else None)
    run_generate(config)


if __name__ == "__main__":
    main()

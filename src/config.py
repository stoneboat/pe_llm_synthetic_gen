"""
Configuration dataclasses for AUG-PE experiments.

Defaults match the paper's hyperparameter tables (Tables 13-14, Appendix B).
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class PEConfig:
    """Private Evolution algorithm configuration."""

    dataset: str = "yelp"
    num_syn_samples: int = 5000
    L: int = 7
    K: int = 0
    num_iterations: int = 20
    noise_multiplier: float = 0.0
    select_mode: str = "rank"
    feature_extractor: str = "stsb-roberta-base-v2"
    feature_extractor_batch_size: int = 1024
    num_nearest_neighbor: int = 1
    nn_mode: str = "L2"
    count_threshold: float = 0.0
    variation_degree: float = 0.5

    @classmethod
    def yelp_gpt2(cls, epsilon: float = math.inf) -> "PEConfig":
        sigma_map = {math.inf: 0.0, 4.0: 4.24, 2.0: 8.03, 1.0: 15.34}
        T = 10 if epsilon < math.inf else 20
        return cls(
            dataset="yelp",
            num_syn_samples=5000,
            L=7, K=0,
            num_iterations=T,
            noise_multiplier=sigma_map.get(epsilon, 0.0),
            feature_extractor="stsb-roberta-base-v2",
        )

    @classmethod
    def yelp_gpt2_large(cls, epsilon: float = math.inf) -> "PEConfig":
        cfg = cls.yelp_gpt2(epsilon)
        return cfg

    @classmethod
    def openreview_gpt2(cls, epsilon: float = math.inf) -> "PEConfig":
        sigma_map = {math.inf: 0.0, 4.0: 3.38, 2.0: 6.22, 1.0: 11.60}
        T = 10
        return cls(
            dataset="openreview",
            num_syn_samples=2000,
            L=7, K=0,
            num_iterations=T,
            noise_multiplier=sigma_map.get(epsilon, 0.0),
            feature_extractor="stsb-roberta-base-v2",
        )

    @classmethod
    def pubmed_gpt2(cls, epsilon: float = math.inf) -> "PEConfig":
        sigma_map = {math.inf: 0.0, 4.0: 3.75, 2.0: 7.01, 1.0: 13.26}
        T = 10
        return cls(
            dataset="pubmed",
            num_syn_samples=2000,
            L=7, K=0,
            num_iterations=T,
            noise_multiplier=sigma_map.get(epsilon, 0.0),
            feature_extractor="sentence-t5-base",
        )


@dataclass
class GenerationConfig:
    """LLM generation configuration."""

    model_name: str = "gpt2"
    temperature: float = 1.4
    max_new_tokens: int = 64
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    fp16: bool = True
    seed: int = 42
    random_sampling_batch_size: int = 1024
    variation_batch_size: int = 1024
    variation_type: str = "yelp_rephrase_tone"
    use_subcategory: bool = True

    @classmethod
    def for_model(cls, model_name: str, dataset: str = "yelp") -> "GenerationConfig":
        """Auto-configure batch sizes and dataset-specific params."""
        batch_sizes = {
            "gpt2": 1024, "gpt2-medium": 128, "gpt2-large": 64,
        }
        temperatures = {
            "yelp": 1.4, "openreview": 1.2, "pubmed": 1.0,
        }
        max_tokens = {
            "yelp": 64, "openreview": 448, "pubmed": 448,
        }
        variation_types = {
            "yelp": "yelp_rephrase_tone",
            "openreview": "openreview_rephrase_tone",
            "pubmed": "pubmed_rephrase_tone",
        }
        bs = batch_sizes.get(model_name, 8)
        return cls(
            model_name=model_name,
            temperature=temperatures.get(dataset, 1.0),
            max_new_tokens=max_tokens.get(dataset, 64),
            random_sampling_batch_size=bs,
            variation_batch_size=bs,
            variation_type=variation_types.get(dataset, "yelp_rephrase_tone"),
        )


@dataclass
class EvalConfig:
    """Downstream evaluation configuration."""

    downstream_model: str = "roberta-base"
    max_seq_length: int = 512
    batch_size: int = 64
    learning_rate: float = 3e-5
    num_epochs: int = 5
    weight_decay: float = 0.0
    eval_metric: str = "accuracy"

    @classmethod
    def for_dataset(cls, dataset: str) -> "EvalConfig":
        if dataset == "yelp":
            return cls(
                downstream_model="roberta-base",
                batch_size=64, learning_rate=3e-5, num_epochs=5,
            )
        elif dataset == "openreview":
            return cls(
                downstream_model="roberta-base",
                batch_size=64, learning_rate=3e-5, num_epochs=10,
            )
        elif dataset == "pubmed":
            return cls(
                downstream_model="google/bert_uncased_L-4_H-256_A-4",
                batch_size=32, learning_rate=3e-4, num_epochs=20,
                weight_decay=0.01,
            )
        return cls()


def build_cli_args(pe_cfg: PEConfig, gen_cfg: GenerationConfig,
                   result_folder: str = "result/yelp/gpt2",
                   train_data_file: Optional[str] = None,
                   train_embeddings_file: str = "") -> list[str]:
    """Build the command-line arguments for main.py from config dataclasses."""
    if train_data_file is None:
        train_data_file = f"data/{pe_cfg.dataset}/train.csv"

    num_samples = pe_cfg.num_syn_samples * pe_cfg.L

    args = [
        "--api", "HFGPT",
        "--dataset", pe_cfg.dataset,
        "--train_data_file", train_data_file,
        "--noise", str(pe_cfg.noise_multiplier),
        "--model_type", gen_cfg.model_name,
        "--length", str(gen_cfg.max_new_tokens),
        "--temperature", str(gen_cfg.temperature),
        "--top_k", str(gen_cfg.top_k),
        "--top_p", str(gen_cfg.top_p),
        "--repetition_penalty", str(gen_cfg.repetition_penalty),
        "--random_sampling_batch_size", str(gen_cfg.random_sampling_batch_size),
        "--variation_batch_size", str(gen_cfg.variation_batch_size),
        "--select_syn_mode", pe_cfg.select_mode,
        "--num_samples_schedule", str(num_samples),
        "--combine_divide_L", str(pe_cfg.L),
        "--init_combine_divide_L", str(pe_cfg.L),
        "--variation_degree_schedule", str(pe_cfg.variation_degree),
        "--lookahead_degree", str(pe_cfg.K),
        "--epochs", str(pe_cfg.num_iterations),
        "--feature_extractor", pe_cfg.feature_extractor,
        "--feature_extractor_batch_size", str(pe_cfg.feature_extractor_batch_size),
        "--mlm_probability", str(pe_cfg.variation_degree),
        "--variation_type", gen_cfg.variation_type,
        "--result_folder", result_folder,
        "--seed", str(gen_cfg.seed),
        "--noise_multiplier", str(pe_cfg.noise_multiplier),
    ]
    if gen_cfg.do_sample:
        args.append("--do_sample")
    if gen_cfg.fp16:
        args.append("--fp16")
    if gen_cfg.use_subcategory:
        args.append("--use_subcategory")
    if train_embeddings_file:
        args.extend(["--train_data_embeddings_file", train_embeddings_file])

    return args

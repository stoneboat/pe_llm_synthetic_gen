# AUG-PE: Differentially Private Synthetic Text via Foundation Model APIs

Reproduction of the paper ["Differentially Private Synthetic Data via Foundation Model APIs 2: Text"](https://arxiv.org/abs/2403.01749) (Xie et al., ICML 2024 Spotlight).

The core algorithm code is from the [original authors' repository](https://github.com/AI-secure/aug-pe), reorganized under `src/`. The repository now includes a config-driven experiment framework with YAML configs and CLI entry points under `src/cli/`, while preserving the original `src/main.py` pipeline for backward compatibility.

## Quick Start

```bash
# 1. Set up the environment (venv, dependencies, data download, model prefetch)
bash scripts/local_scripts/install_gpu.sh

# 2. Activate
source /tmp/python-venv/pe-venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src:.
export LD_LIBRARY_PATH="/tmp/python-venv/pe-venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"

# 3. Precompute private data embeddings (~40 min for full Yelp)
bash scripts/embeddings.sh --yelp

# 4. Run AUG-PE generation (GPT-2, Yelp, non-DP, 20 iterations)
export CUDA_VISIBLE_DEVICES=0
python -m src.cli.generate --config configs/experiments/yelp_original.yaml

# 5. Evaluate downstream classification accuracy
python -m src.cli.downstream --config configs/experiments/yelp_original.yaml

# 6. Compute embedding distribution metrics (FID, precision, recall)
python -m src.cli.evaluate --config configs/experiments/yelp_original.yaml
```

## Primary Commands

The recommended interface is the YAML-driven CLI.

### Generate synthetic text

Non-DP Yelp:

```bash
PYTHONPATH=src:. python -m src.cli.generate --config configs/experiments/yelp_original.yaml
```

DP Yelp (`epsilon = 1`, `sigma = 15.34`):

```bash
PYTHONPATH=src:. python -m src.cli.generate --config configs/experiments/yelp_original_dp_eps1.yaml
```

If you see `nvrtc: error: failed to open libnvrtc-builtins.so.13.0`, your shell is usually picking up system CUDA libraries before the venv's bundled CUDA 13 runtime. Export the venv CUDA library path shown above before running the CLI.

### Evaluate saved generations

All available iterations:

```bash
PYTHONPATH=src:. python -m src.cli.evaluate --config configs/experiments/yelp_original.yaml
PYTHONPATH=src:. python -m src.cli.downstream --config configs/experiments/yelp_original.yaml
```

Single iteration, for example iteration `10`:

```bash
PYTHONPATH=src:. python -m src.cli.evaluate --config configs/experiments/yelp_original.yaml --iteration 10
PYTHONPATH=src:. python -m src.cli.downstream --config configs/experiments/yelp_original.yaml --iteration 10
```

### CLI overrides

```bash
PYTHONPATH=src:. python -m src.cli.generate \
  --config configs/experiments/yelp_original.yaml \
  --override noise_multiplier=15.34 seed=123
```

## Config-Driven Framework

The new framework separates:

- `src/data_adapters/` for dataset loading and labeling
- `src/mechanisms/` for generation mechanisms
- `src/tasks/` for evaluation tasks
- `src/config/` for YAML loading
- `src/cli/` for runnable entry points

Reusable configs live under:

- `configs/datasets/`
- `configs/mechanisms/`
- `configs/tasks/`
- `configs/experiments/`

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── paper/                               # Reports and reference PDFs
├── configs/                             # YAML experiment configs
│   ├── datasets/
│   ├── mechanisms/
│   ├── tasks/
│   └── experiments/
│
├── src/                                 # Python source code
│   ├── cli/                             #   Config-driven entry points
│   │   ├── generate.py
│   │   ├── evaluate.py
│   │   └── downstream.py
│   ├── data_adapters/                   #   Dataset adapters
│   ├── mechanisms/                      #   Mechanism implementations
│   ├── tasks/                           #   Evaluation tasks
│   ├── config/                          #   YAML config loader
│   ├── registry.py                      #   Registry pattern
│   ├── main.py                          #   Original monolithic entry point (preserved)
│   ├── metric.py                        #   Legacy distribution metrics entry point
│   ├── pre_comp_emb.py                  #   Precompute embeddings
│   ├── legacy_config.py                 #   Legacy dataclass configs
│   ├── dp_accounting.py                 #   Privacy budget computation (Theorem 2)
│   ├── apis/                            #   LLM generation APIs
│   │   ├── hf_api.py                    #     HuggingFace GPT-2 (RANDOM_API + VARIATION_API)
│   │   └── utils.py                     #     Prompts, tones, subcategories
│   ├── dpsda/                           #   DP algorithm components
│   │   ├── dp_counter.py                #     DP nearest-neighbor histogram (faiss)
│   │   ├── feature_extractor.py         #     Sentence-transformer embeddings
│   │   ├── data_loader.py               #     Dataset loading
│   │   └── logging.py                   #     Sample saving, FID logging
│   └── utility_eval/                    #   Legacy downstream evaluation scripts
│
├── data/
│   ├── yelp/                            #   Yelp reviews
│   ├── pubmed/                          #   PubMed abstracts
│   └── openreview/                      #   OpenReview reviews
│
├── scripts/                             # Legacy helper scripts (preserved)
│   ├── local_scripts/install_gpu.sh
│   ├── local_scripts/sanity_check.sh
│   ├── hf/yelp/
│   ├── hf/pubmed/
│   ├── embeddings.sh
│   └── download_data.sh
│
└── Notebook/demo.ipynb                  # Step-by-step Python walkthrough
```

## Algorithm Overview

AUG-PE generates differentially private synthetic text using only inference API access to an LLM:

1. **RANDOM_API**: Generate initial samples from the LLM using class-label prompts
2. **Iterate T times**:
   - Embed synthetic samples using a sentence-transformer
   - Each private sample votes for its nearest synthetic neighbor (DP histogram with Gaussian noise)
   - Select top samples by vote count (rank-based selection)
   - Generate variations via VARIATION_API (paraphrasing with random tones)
   - Combine selected samples + variations for next iteration
3. **Output**: The selected samples from the final iteration

Privacy is ensured by adding Gaussian noise to the histogram at each iteration. The total privacy cost composes over `T` iterations via the adaptive composition theorem for Gaussian mechanisms.

In the new framework, the original algorithm is represented by the `original_aug_pe` mechanism and configured through YAML experiment files such as [`configs/experiments/yelp_original.yaml`](/home/ubuntu/pe_llm_synthetic_gen/configs/experiments/yelp_original.yaml) and [`configs/experiments/yelp_original_dp_eps1.yaml`](/home/ubuntu/pe_llm_synthetic_gen/configs/experiments/yelp_original_dp_eps1.yaml).

## Key Hyperparameters (Yelp + GPT-2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Nsyn | 5,000 | Number of synthetic samples |
| L | 7 | Number of variations + 1 |
| K | 0 | Lookahead degree (self-embedding) |
| T | 20 (non-DP) / 10 (DP) | PE iterations |
| temperature | 1.4 | LLM generation temperature |
| max_tokens | 64 | Max new tokens per generation |
| embedding | stsb-roberta-base-v2 | Sentence-transformer model |
| sigma | 15.34 / 8.03 / 4.24 | Noise multiplier for epsilon = 1 / 2 / 4 |

## Privacy Accounting

```python
from src.dp_accounting import compute_sigma, compute_epsilon

# Compute required sigma for epsilon=1 on Yelp
sigma = compute_sigma(epsilon=1.0, T=10, delta=1/(1939290 * math.log(1939290)))
# sigma ≈ 15.34
```

## Hardware

Tested on NVIDIA L40S (46 GB VRAM). GPT-2 in FP16 uses ~0.5 GB, leaving ample room for embeddings and nearest-neighbor search.

## Backward Compatibility

The original entry point is still preserved:

```bash
cd src
python main.py --api HFGPT --dataset yelp ...
```

The YAML-driven CLI is now the recommended interface for new runs.

## Citation

```bibtex
@inproceedings{xie2024differentially,
  title={Differentially Private Synthetic Data via Foundation Model {API}s 2: Text},
  author={Xie, Chulin and Lin, Zinan and Backurs, Arturs and Gopi, Sivakanth and Yu, Da
          and Inan, Huseyin A and Nori, Harsha and Jiang, Haotian and Zhang, Huishuai
          and Lee, Yin Tat and Li, Bo and Yekhanin, Sergey},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
}
```

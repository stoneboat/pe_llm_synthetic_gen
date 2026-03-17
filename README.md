# AUG-PE: Differentially Private Synthetic Text via Foundation Model APIs

Reproduction of the paper ["Differentially Private Synthetic Data via Foundation Model APIs 2: Text"](https://arxiv.org/abs/2403.01749) (Xie et al., ICML 2024 Spotlight).

The core algorithm code is from the [original authors' repository](https://github.com/AI-secure/aug-pe), reorganized with all Python source under `src/`. This repo adds environment setup for an NVIDIA L40S GPU, configuration dataclasses, privacy accounting utilities, and a step-by-step demo notebook.

## Quick Start

```bash
# 1. Set up the environment (venv, dependencies, data download, model prefetch)
bash scripts/local_scripts/install_gpu.sh

# 2. Activate
source /tmp/python-venv/pe-venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 3. Precompute private data embeddings (~40 min for full Yelp)
bash scripts/embeddings.sh --yelp

# 4. Run AUG-PE generation (GPT-2, Yelp, non-DP, 20 iterations)
export CUDA_VISIBLE_DEVICES=0
bash scripts/hf/yelp/generate.sh

# 5. Evaluate downstream classification accuracy
bash scripts/hf/yelp/downstream.sh

# 6. Compute embedding distribution metrics (FID, precision, recall)
bash scripts/hf/yelp/metric.sh
```

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── paper/                               # Reference PDF
│
├── src/                                 # All Python source code
│   ├── main.py                          #   Core AUG-PE entry point
│   ├── metric.py                        #   Distribution metrics entry point
│   ├── pre_comp_emb.py                  #   Precompute embeddings
│   ├── config.py                        #   Dataclass configs (paper hyperparameters)
│   ├── dp_accounting.py                 #   Privacy budget computation (Theorem 2)
│   ├── apis/                            #   LLM generation APIs
│   │   ├── hf_api.py                    #     HuggingFace GPT-2 (RANDOM_API + VARIATION_API)
│   │   └── utils.py                     #     Prompts, tones, subcategories
│   ├── dpsda/                           #   DP algorithm components
│   │   ├── dp_counter.py                #     DP nearest-neighbor histogram (faiss)
│   │   ├── feature_extractor.py         #     Sentence-transformer embeddings
│   │   ├── data_loader.py               #     Dataset loading
│   │   └── logging.py                   #     Sample saving, FID logging
│   └── utility_eval/                    #   Downstream evaluation
│       └── run_classification.py        #     RoBERTa text classification
│
├── data/
│   ├── yelp/                            #   Yelp reviews (train.csv downloaded separately)
│   └── pubmed/                          #   PubMed abstracts
│
├── scripts/
│   ├── local_scripts/install_gpu.sh     # Environment setup
│   ├── local_scripts/sanity_check.sh    # Quick validation run
│   ├── hf/yelp/generate.sh              # AUG-PE generation (GPT-2 + Yelp)
│   ├── hf/yelp/downstream.sh            # Downstream evaluation
│   ├── hf/yelp/metric.sh                # FID / precision / recall
│   ├── hf/pubmed/                       # PubMed experiment scripts
│   ├── embeddings.sh                    # Embedding precomputation
│   └── download_data.sh                 # Dataset download helper
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

Privacy is ensured by adding Gaussian noise to the histogram at each iteration. The total privacy cost composes over T iterations via the adaptive composition theorem for Gaussian mechanisms.

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

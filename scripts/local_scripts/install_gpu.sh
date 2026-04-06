#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="/tmp/python-venv/pe-venv"

echo "=== AUG-PE Reproduction: Environment Setup ==="
echo "Repo directory: $REPO_DIR"
echo ""

# ── 1. GPU verification ──────────────────────────────────────────────
echo "[1/7] Checking GPU..."
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "  WARNING: nvidia-smi not available. Continuing without GPU verification."
fi

# ── 2. Python venv ───────────────────────────────────────────────────
echo "[2/7] Setting up Python venv at $VENV_DIR..."
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  Reusing existing venv."
else
    if [ -d "$VENV_DIR" ]; then
        echo "  Removing broken venv directory..."
        rm -rf "$VENV_DIR"
    fi
    python3 -m venv "$VENV_DIR"
    echo "  Created new venv."
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 3. Install dependencies ──────────────────────────────────────────
echo "[3/7] Installing Python dependencies..."
pip install -r "$REPO_DIR/requirements.txt" --quiet 2>&1 | tail -5

# Verify critical imports
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'  Transformers {transformers.__version__}')"
python3 -c "import sentence_transformers; print(f'  Sentence-Transformers {sentence_transformers.__version__}')"

# ── 4. Download Yelp train.csv ───────────────────────────────────────
echo "[4/7] Downloading datasets..."
YELP_TRAIN="$REPO_DIR/data/yelp/train.csv"
if [ -f "$YELP_TRAIN" ]; then
    echo "  Yelp train.csv already exists ($(du -h "$YELP_TRAIN" | cut -f1))."
else
    echo "  Downloading Yelp train.csv (~1.2 GB)..."
    gdown "https://drive.google.com/uc?id=1epLuBxCk5MGnm1GiIfLcTcr-tKgjCrc2" -O "$YELP_TRAIN" || {
        echo "  WARNING: gdown failed. Try manually from: https://drive.google.com/uc?id=1epLuBxCk5MGnm1GiIfLcTcr-tKgjCrc2"
    }
fi

PUBMED_TRAIN="$REPO_DIR/data/pubmed/train.csv"
if [ -f "$PUBMED_TRAIN" ]; then
    echo "  PubMed train.csv already exists ($(du -h "$PUBMED_TRAIN" | cut -f1))."
else
    echo "  Downloading PubMed train.csv (~117 MB)..."
    gdown "https://drive.google.com/uc?id=12-zV93MQNPvM_ORUoahZ2n4odkkOXD-r" -O "$PUBMED_TRAIN" || {
        echo "  WARNING: gdown failed. Try manually from: https://drive.google.com/uc?id=12-zV93MQNPvM_ORUoahZ2n4odkkOXD-r"
    }
fi

# ── 5. Prefetch models ───────────────────────────────────────────────
echo "[5/7] Prefetching models to HuggingFace cache..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('  Downloading GPT-2...')
AutoTokenizer.from_pretrained('gpt2')
AutoModelForCausalLM.from_pretrained('gpt2')
print('  GPT-2 ready.')
" 2>/dev/null

python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading stsb-roberta-base-v2...')
SentenceTransformer('stsb-roberta-base-v2')
print('  Embedding model ready.')
" 2>/dev/null

# ── 6. Jupyter kernel ────────────────────────────────────────────────
echo "[6/7] Registering Jupyter kernel..."
python3 -m ipykernel install --user --name pe-venv --display-name "Python (PE)" 2>/dev/null || true
echo "  Kernel 'Python (PE)' registered."

# ── 7. Environment variables ─────────────────────────────────────────
echo "[7/7] Setting environment variables..."
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo "  export PYTORCH_ALLOC_CONF=expandable_segments:True"
echo "  export TOKENIZERS_PARALLELISM=false"
echo "  export LD_LIBRARY_PATH=\"$VENV_DIR/lib/python3.10/site-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH}\""
echo ""
echo "Quick sanity check:"
echo "  cd $REPO_DIR"
echo "  bash scripts/hf/yelp/generate.sh"

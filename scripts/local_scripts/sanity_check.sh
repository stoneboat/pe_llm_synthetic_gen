#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check: GPT-2, Yelp, small samples, 3 iterations
# Expected runtime: ~10-15 minutes on L40S

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

mlm_prob=0.5
var_type="yelp_rephrase_tone"
feat_ext="stsb-roberta-base-v2"
length=64
temperature=1.4
num_seed_samples=200
lookahead_degree=0
L=2
init_L=${L}
num_samples=$((L*num_seed_samples))
epochs=3
select_syn_mode=rank
model_type=gpt2
noise=0
batch_size=256
feature_extractor_batch_size=512

result_folder="result/sanity_check"
rm -rf "$result_folder"

echo "=== AUG-PE Sanity Check ==="
echo "  Model: $model_type"
echo "  Dataset: yelp (subsampled to 2000 private samples)"
echo "  Nsyn: $num_seed_samples, L: $L, T: $epochs"
echo "  Total synthetic pool: $num_samples"
echo ""

python src/main.py \
    --train_data_file "data/yelp/train.csv" \
    --api HFGPT \
    --dataset yelp \
    --noise_multiplier ${noise} \
    --model_type ${model_type} \
    --do_sample \
    --length ${length} \
    --random_sampling_batch_size ${batch_size} \
    --variation_batch_size ${batch_size} \
    --fp16 \
    --temperature ${temperature} \
    --select_syn_mode ${select_syn_mode} \
    --num_samples_schedule ${num_samples} \
    --combine_divide_L ${L} \
    --init_combine_divide_L ${init_L} \
    --variation_degree_schedule ${mlm_prob} \
    --lookahead_degree ${lookahead_degree} \
    --epochs ${epochs} \
    --use_subcategory \
    --feature_extractor ${feat_ext} \
    --feature_extractor_batch_size ${feature_extractor_batch_size} \
    --mlm_probability ${mlm_prob} \
    --variation_type ${var_type} \
    --result_folder ${result_folder} \
    --num_private_samples 2000 \
    --compute_fid True

echo ""
echo "=== Sanity check complete ==="
echo "Results saved to: $result_folder"

if [ -f "$result_folder/fid.csv" ]; then
    echo ""
    echo "FID scores across iterations:"
    cat "$result_folder/fid.csv"
fi

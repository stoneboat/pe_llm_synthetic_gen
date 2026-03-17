#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 --dataset_name"
  exit 1
fi

# Use a case statement to handle different datasets
case $1 in
  --pubmed)
    python src/pre_comp_emb.py --dataset pubmed --model_name_or_path 'sentence-t5-base'
    ;;
  --yelp)
    python src/pre_comp_emb.py --dataset yelp --model_name_or_path 'stsb-roberta-base-v2'
    ;;
  *)
    echo "Invalid dataset. Available datasets are: --pubmed, --yelp"
    exit 1
    ;;
esac

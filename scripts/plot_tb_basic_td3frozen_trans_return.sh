#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache

# Optional: pick a GPU
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

python -m eval.plot_tb_avg_return \
  --logdirs \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td3_frozen \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td3_unfreeze1 \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td3_unfreeze2 \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td3_fullft \
  --tag Eval_AverageReturn \
  --output ./plots/basic_eval_td3frozen_trans_return.png
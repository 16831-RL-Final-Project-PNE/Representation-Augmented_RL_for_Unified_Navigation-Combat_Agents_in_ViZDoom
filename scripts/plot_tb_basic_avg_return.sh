#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache

# Optional: pick a GPU
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

python -m eval.plot_tb_avg_return \
  --logdirs \
    ./logs/data/patrick/logs/tb_basic_ppo \
    ./logs/data/patrick/logs/tb_basic_ppo_dinov2_run2 \
    ./logs/data/patrick/logs/tb_basic_ppo_dinov3_run2 \
    ./logs/data/patrick/logs/tb_basic_ppo_rnd_run2 \
    ./logs/data/patrick/logs/tb_basic_ppo_dinov2_rnd_run2 \
    ./logs/data/patrick/logs/tb_basic_ppo_dinov3_rnd_run2 \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td1_frozen \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td2_frozen \
    ./logs/data/patrick/logs/tb_basic_ppo_jepa_td3_frozen \
  --tag Eval_AverageReturn \
  --output ./plots/basic_eval_avg_return.png
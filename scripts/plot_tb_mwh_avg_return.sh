#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache

# Optional: pick a GPU
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

python -m eval.plot_tb_avg_return \
  --logdirs \
    ./logs/tb_mwh_ppo_v2 \
    ./logs/data/patrick/logs/tb_mwh_ppo_dinov2 \
    ./logs/data/patrick/logs/tb_mwh_ppo_dinov3 \
    ./logs/data/patrick/logs/tb_mwh_ppo_rnd \
    ./logs/data/patrick/logs/tb_mwh_ppo_dinov2_rnd \
    ./logs/data/patrick/logs/tb_mwh_ppo_dinov3_rnd \
    ./logs/data/patrick/logs/tb_mwh_ppo_jepa_td1_frozen \
    ./logs/data/patrick/logs/tb_mwh_ppo_jepa_td2_frozen \
    ./logs/data/patrick/logs/tb_mwh_ppo_jepa_td3_frozen \
  --tag Eval_AverageReturn \
  --output ./plots_mwh/mwh_eval_avg_return.png
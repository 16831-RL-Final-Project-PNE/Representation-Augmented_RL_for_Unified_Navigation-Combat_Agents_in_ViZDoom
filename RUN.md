# Random Agent
# 1 run random agent training (Actually it is no training, just eval)
python -m scripts.train_random_basic --scenario basic --action_space usual

python -m scripts.train_random_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 20 \
  --steps_per_iteration 4096 \
  --log_root ./logs_mwh \
  --plot_root ./plots_mwh \
  --tb_dirname tb_run1 \
  --eval_log_name mwh_random_run1_eval.npz \
  --checkpoint_root ./checkpoints_mwh \
  --checkpoint_name mwh_random_run1 \
  --save_every 0

# 2 use unified evaluator to plot average eval return vs interations
python -m eval.evaluation \
  --log_path ./logs/basic_random_eval.npz \
  --out ./plots/basic_random_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs_mwh/mwh_random_run1_eval.npz \
  --out ./plots_mwh/mwh_random_run1_eval.png \
  --annotate_last_only

# 3 output the playing game GIF using random agent
python -m scripts.eval_random_play \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_random_v7/best.gif \
  --gif_dir ./out/basic_random_v7/eps \
  --fps 12 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

python -m scripts.eval_random_play \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_random_v8/best.gif \
  --gif_dir ./out/basic_random_v8/eps \
  --fps 12 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 47

python -m scripts.eval_random_play \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_random_v3/best.gif \
  --gif_dir ./out/mwh_random_v3/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

python -m scripts.eval_random_play \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --max_gif_frames 800 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_random_v4/best.gif \
  --gif_dir ./out/mwh_random_v4/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0

python -m scripts.eval_random_play \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --max_gif_frames 800 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_random_v5/best.gif \
  --gif_dir ./out/mwh_random_v5/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 0


# PPO Agent
# 1 run ppo agent training
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo \
  --save_every 80

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.05 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_rnd_eval_run2.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_rnd_run2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_rnd_run2 \
  --save_every 80

python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 400 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir ./logs \
  --eval_log_name mwh_ppo_eval_v2.npz \
  --tb_log_dir ./logs/tb_mwh_ppo_v2 \
  --checkpoint_dir ./checkpoints_mwh \
  --checkpoint_name mwh_ppo_v2 \
  --save_every 80

# attempting
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.01 \
  --rnd_ext_coef 2.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-5 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_deterministic \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_rnd_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name mwh_ppo_rnd \
  --save_every 80

# need modified filename
CUDA_VISIBLE_DEVICES=1 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_dinov3_run2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_dinov3_run2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov3_run2 \
  --save_every 80 \
  --backbone dinov3 \
  --freeze_backbone

# already run
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 3e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.02 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_deterministic \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_dinov3_rnd_eval_run2.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_dinov3_rnd_run2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov3_rnd_run2 \
  --save_every 80

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_dinov3_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_dinov3 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_dinov3 \
  --save_every 80 \
  --backbone dinov3 \
  --freeze_backbone

# attempting
CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.01 \
  --rnd_ext_coef 2.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-5 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_deterministic \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_dinov3_rnd_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_dinov3_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name mwh_ppo_dinov3_rnd \
  --save_every 80

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir ./logs \
  --eval_log_name basic_ppo_dinov2_run2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_dinov2_run2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov2_run2 \
  --save_every 80 \
  --backbone dinov2 \
  --freeze_backbone

# attempting
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --backbone dinov2 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 3e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.02 \
  --rnd_ext_coef 1.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 1e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_dinov2_rnd_eval_run2.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_dinov2_rnd_run2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_dinov2_rnd_run2 \
  --save_every 80


CUDA_VISIBLE_DEVICES=7 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_dinov2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_dinov2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_dinov2 \
  --save_every 80 \
  --backbone dinov2 \
  --freeze_backbone

# attempting
CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone dinov2 \
  --freeze_backbone \
  --feat_dim 256 \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --use_rnd \
  --rnd_int_coef 0.01 \
  --rnd_ext_coef 2.0 \
  --rnd_gamma 0.99 \
  --rnd_lr 3e-4 \
  --rnd_weight_decay 1e-4 \
  --rnd_batch_size 256 \
  --rnd_epochs 1 \
  --rnd_int_decay \
  --eval_deterministic \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_dinov2_rnd_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_dinov2_rnd \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name mwh_ppo_dinov2_rnd \
  --save_every 80


# 2 use unified evaluator to plot average eval return vs interations

# load checkpoint, use deterministic evaluation to run episodes again and record rewards
python -m scripts.eval_ppo_checkpoint \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_iter199.pt \
  --eval_log_path ./logs/basic_ppo_eval_det.npz \
  --episodes 50

python -m eval.evaluation \
  --log_path ./logs/basic_ppo_eval_det.npz \
  --out ./plots/basic_ppo_eval_det.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/basic_ppo_eval.npz \
  --out ./plots/basic_ppo_eval.png \
  --annotate_last_only

# load checkpoint, use deterministic evaluation to run episodes again and record rewards
python -m scripts.eval_ppo_checkpoint \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_rnd_final.pt \
  --eval_log_path /data/patrick/16831RL/npzfolder/basic_ppo_rnd_eval_det.npz \
  --episodes 200

python -m eval.evaluation \
  --log_path /data/patrick/16831RL/npzfolder/basic_ppo_rnd_eval_det.npz \
  --out ./plots/basic_ppo_rnd_eval_det.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/mwh_ppo_eval.npz \
  --out ./plots_mwh/mwh_ppo_eval.png \
  --annotate_last_only

python -m eval.evaluation \
  --log_path ./logs/mwh_ppo_eval_v2.npz \
  --out ./plots_mwh/mwh_ppo_eval_v2.png \
  --annotate_last_only

# 3 output the playing game GIF using ppo agent
python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/basic_ppo_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 320x240 \
  --gif ./out/basic_ppo_v4/best.gif \
  --gif_dir ./out/basic_ppo_v4/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints/basic_ppo_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_v5/best.gif \
  --gif_dir ./out/basic_ppo_v5/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_final.pt \
  --scenario basic \
  --action_space usual \
  --episodes 15 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_v6/best.gif \
  --gif_dir ./out/basic_ppo_v6/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_dinov3_run2_final.pt \
  --scenario basic \
  --action_space usual \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_dinov3/best.gif \
  --gif_dir ./out/basic_ppo_dinov3/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_dinov3_rnd_run2_final.pt \
  --scenario basic \
  --action_space usual \
  --backbone dinov3 \
  --freeze_backbone \
  --feat_dim 256 \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_dinov3_rnd/best.gif \
  --gif_dir ./out/basic_ppo_dinov3_rnd/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_jepa_td3_frozen_final.pt \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --freeze_backbone \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_jepa_td3_frozen/best.gif \
  --gif_dir ./out/basic_ppo_jepa_td3_frozen/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_jepa_td3_unfreeze2_final.pt \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --freeze_backbone \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_jepa_td3_unfreeze2/best.gif \
  --gif_dir ./out/basic_ppo_jepa_td3_unfreeze2/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/basic_ppo_jepa_td3_fullft_final.pt \
  --scenario basic \
  --action_space usual \
  --backbone cnn \
  --freeze_backbone \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/basic_ppo_jepa_td3_fullft/best.gif \
  --gif_dir ./out/basic_ppo_jepa_td3_fullft/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints_mwh/mwh_ppo_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_v1/best.gif \
  --gif_dir ./out/mwh_ppo_v1/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint ./checkpoints_mwh/mwh_ppo_v2_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_v2/best.gif \
  --gif_dir ./out/mwh_ppo_v2/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints_mwh/mwh_ppo_dinov3_iter_80.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone dinov3 \
  --freeze_backbone \
  --episodes 10 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_dinov3/best.gif \
  --gif_dir ./out/mwh_ppo_dinov3/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints/mwh_ppo_dinov3_rnd_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone dinov3 \
  --freeze_backbone \
  --episodes 5 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_dinov3_rnd/best.gif \
  --gif_dir ./out/mwh_ppo_dinov3_rnd/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --seed 100 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints_mwh/mwh_ppo_jepa_td3run3_frozen_iter_80.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --freeze_backbone \
  --episodes 10 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_jepa_td3run3_frozen/best.gif \
  --gif_dir ./out/mwh_ppo_jepa_td3run3_frozen/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints_mwh/mwh_ppo_jepa_td3_frozen_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --freeze_backbone \
  --episodes 10 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_jepa_td3_frozen/best.gif \
  --gif_dir ./out/mwh_ppo_jepa_td3_frozen/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints_mwh/mwh_ppo_jepa_td3_unfreeze2_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --freeze_backbone \
  --episodes 10 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_jepa_td3_unfreeze2/best.gif \
  --gif_dir ./out/mwh_ppo_jepa_td3_unfreeze2/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

python -m scripts.eval_ppo_basic_play \
  --checkpoint /data/patrick/16831RL/checkpoints_mwh/mwh_ppo_jepa_td3_fullft_final.pt \
  --scenario my_way_home \
  --action_space no_shoot \
  --backbone cnn \
  --freeze_backbone \
  --episodes 10 \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 --height 84 \
  --base_res 800x600 \
  --gif ./out/mwh_ppo_jepa_td3_fullft/best.gif \
  --gif_dir ./out/mwh_ppo_jepa_td3_fullft/eps \
  --fps 15 \
  --gif_scale 1 \
  --gif_repeat 1 \
  --deterministic

# 4. Collect JEPA Rollout Frames
# basic env
CUDA_VISIBLE_DEVICES=3 python -m scripts.collect_jepa_frames \
  --scenario basic \
  --action_space usual \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 \
  --height 84 \
  --base_res 320x240 \
  --seed 0 \
  --num_steps 50000 \
  --trained_ckpt "" \
  --trained_rollout_prob 0.0 \
  --max_episode_steps 300 \
  --out_path /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy

CUDA_VISIBLE_DEVICES=3 python -m scripts.collect_jepa_frames \
  --scenario basic \
  --action_space usual \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 \
  --height 84 \
  --base_res 320x240 \
  --seed 1 \
  --num_steps 50000 \
  --trained_ckpt /data/patrick/16831RL/checkpoints/basic_ppo_final.pt \
  --trained_rollout_prob 1.0 \
  --max_episode_steps 300 \
  --out_path /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy

CUDA_VISIBLE_DEVICES=3 python -m scripts.collect_jepa_frames \
  --scenario my_way_home \
  --action_space no_shoot \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 \
  --height 84 \
  --base_res 320x240 \
  --seed 0 \
  --num_steps 100000 \
  --trained_ckpt "" \
  --trained_rollout_prob 0.0 \
  --max_episode_steps 16384 \
  --out_path /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy

CUDA_VISIBLE_DEVICES=3 python -m scripts.collect_jepa_frames \
  --scenario my_way_home \
  --action_space no_shoot \
  --frame_repeat 4 \
  --frame_stack 4 \
  --width 84 \
  --height 84 \
  --base_res 320x240 \
  --seed 1 \
  --num_steps 100000 \
  --trained_ckpt /data/patrick/16831RL/checkpoints/mwh_ppo_rnd_final.pt \
  --trained_rollout_prob 1.0 \
  --max_episode_steps 16384 \
  --out_path /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy


# 5. I-JEPA Pretraining
CUDA_VISIBLE_DEVICES=4 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_wmup0p1_td0_varw2_std1_nmb5_mask_0p6.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.10 \
  --mask_ratio 0.6 \
  --temporal_delta 0 \
  --var_weight 2.0 \
  --covar_weight 1.0 \
  --std_target 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e150_wmup0p1_td0_varw2_std1_nmb5_mask_0p6

CUDA_VISIBLE_DEVICES=6 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td1.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 1 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td1

CUDA_VISIBLE_DEVICES=2 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td2.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 2 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td2

CUDA_VISIBLE_DEVICES=2 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3

CUDA_VISIBLE_DEVICES=2 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 200 \
  --lr 1.5e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 0.8 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4

# Cannot learn a useful representation, loss_recon = 1.68, far away from typical useful range of 0.3-0.6
CUDA_VISIBLE_DEVICES=2 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_random_50k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_basic_expert_50k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p5_covw1_std1_mask0p7_td3_run5.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 200 \
  --lr 1.5e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.7 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 0.5 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa \
  --wandb_run_name basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p5_covw1_std1_mask0p7_td3_run5

CUDA_VISIBLE_DEVICES=4 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6

CUDA_VISIBLE_DEVICES=6 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td1.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 1 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td1

CUDA_VISIBLE_DEVICES=2 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td2.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 2 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td2

CUDA_VISIBLE_DEVICES=4 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 2.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3

CUDA_VISIBLE_DEVICES=4 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw1_std1_covw1_mask0p6_td3_run3.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 150 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 1.0 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw1_std1_covw1_mask0p6_td3_run3

CUDA_VISIBLE_DEVICES=0 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 200 \
  --lr 1.5e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.6 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 0.8 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4

CUDA_VISIBLE_DEVICES=0 python -m scripts.pretrain_jepa \
  --frames_paths \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_random_100k.npy \
    /data/patrick/16831RL/jepa_rollout_colllect/jepa_frames_mwh_expert_100k.npy \
  --out_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p7_td3_run5.pt \
  --in_channels 12 \
  --feat_dim 256 \
  --batch_size 128 \
  --epochs 200 \
  --lr 1.5e-3 \
  --min_lr 1e-5 \
  --warmup_ratio 0.1 \
  --mask_ratio 0.7 \
  --temporal_delta 3 \
  --num_blocks 5 \
  --var_weight 0.8 \
  --std_target 1.0 \
  --covar_weight 1.0 \
  --momentum 0.996 \
  --device cuda \
  --num_workers 4 \
  --use_wandb \
  --wandb_project vizdoom-jepa-mwh \
  --wandb_run_name mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p7_td3_run5



# 6 JEPA Downstream Post-training
# Basic
# A: JEPA conv frozen（linear probe RL）
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_forzen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr2en3_wmup0p15_nmb5_varw0p85_covw0p7_std0p8_mask0p8.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

# B: JEPA conv only unfreezing last 1 (jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_partial1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_partial1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_partial1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr2en3_wmup0p15_nmb5_varw0p85_covw0p7_std0p8_mask0p8.pt \
  --jepa_partial_unfreeze 1

# C: JEPA conv only unfreezing last 2 (jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_partial2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_partial2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_partial2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr2en3_wmup0p15_nmb5_varw0p85_covw0p7_std0p8_mask0p8.pt \
  --jepa_partial_unfreeze 2

# D: JEPA conv only unfreezing last 3 (full fine-tuning, jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_fullft_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr2en3_wmup0p15_nmb5_varw0p85_covw0p7_std0p8_mask0p8.pt

# E: mask = 0.6, td=1
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td1_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td1_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td1_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td1.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td1_unfreeze1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td1_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td1_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td1.pt \
  --jepa_partial_unfreeze 1

# F: mask = 0.6, td=2
CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td2_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td2_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td2_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td2.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

# G: mask = 0.6, td=3
CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3_unfreeze1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3.pt \
  --jepa_partial_unfreeze 1

CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3_unfreeze2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3_unfreeze2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3_unfreeze2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3.pt \
  --jepa_partial_unfreeze 2

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3_fullft_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_covw1_std1_mask0p6_td3.pt

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3run4_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3run4_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3run4_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3run4_unfreeze1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3run4_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3run4_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 1

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3run4_unfreeze2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3run4_unfreeze2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3run4_unfreeze2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 2

CUDA_VISIBLE_DEVICES=1 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_td3run4_fullft_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_td3run4_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_td3run4_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_covw1_std1_mask0p6_td3_run4.pt



# F mask = 0.6, td=0
CUDA_VISIBLE_DEVICES=1 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_frozen_mask0p6_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_frozen_mask0p6 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_frozen_mask0p6 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_wmup0p1_td0_varw2_std1_nmb5_mask_0p6.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --clip_coef 0.1 \
  --value_coef 0.25 \
  --entropy_coef 0.01 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_ppo_jepa_mask0p6_unfreeze1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_ppo_jepa_mask0p6_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_ppo_jepa_mask0p6_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/basic_cnn_jepa_coswarm_e150_wmup0p1_td0_varw2_std1_nmb5_mask_0p6.pt \
  --jepa_partial_unfreeze 1


# My_Way_Home
# A: JEPA conv frozen（linear probe RL）
CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_frozen_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1en3_wmup0p1_nmb5_varw1_std0p85_covw1_mask0p8.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

# B: JEPA conv only unfreezing last 1 (jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_partial1_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_partial1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_partial1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1en3_wmup0p1_nmb5_varw1_std0p85_covw1_mask0p8.pt \
  --jepa_partial_unfreeze 1

# C: JEPA conv only unfreezing last 2 (jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_partial2_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_partial2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_partial2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1en3_wmup0p1_nmb5_varw1_std0p85_covw1_mask0p8.pt \
  --jepa_partial_unfreeze 2

# D: JEPA conv only unfreezing last 3 (full fine-tuning, jepa_partial_unfreeze) layers + head
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_fullft_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1en3_wmup0p1_nmb5_varw1_std0p85_covw1_mask0p8.pt \
  --jepa_partial_unfreeze 3



# E: mask=0.6, td=1
CUDA_VISIBLE_DEVICES=6 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td1_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td1_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td1_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td1.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td1_unfreeze1.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td1_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td1_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td1.pt \
  --jepa_partial_unfreeze 1 


# F: mask=0.6, td=2
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td2_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td2_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td2_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td2.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

# G: mask=0.6, td=3
CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run3_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run3_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run3_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw1_std1_covw1_mask0p6_td3_run3.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run4_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run4_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run4_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run5_frozen.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run5_frozen \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run5_frozen \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p7_td3_run5.pt \
  --freeze_backbone \
  --jepa_partial_unfreeze 0

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3_unfreeze1.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3.pt \
  --jepa_partial_unfreeze 1

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run4_unfreeze1.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run4_unfreeze1 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run4_unfreeze1 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4.pt \
  --jepa_partial_unfreeze 1

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3_unfreeze2.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_unfreeze2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3_unfreeze2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3.pt \
  --jepa_partial_unfreeze 2

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run4_unfreeze2.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run4_unfreeze2 \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run4_unfreeze2 \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4.pt \
  --jepa_partial_unfreeze 2

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3_fullft.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e150_lr1en3_wmup0p1_nmb5_varw2_std1_covw1_mask0p6_td3.pt

CUDA_VISIBLE_DEVICES=2 python -m scripts.train_ppo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 200 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --clip_coef 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_ppo_jepa_td3run4_fullft.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_ppo_jepa_td3run4_fullft \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_ppo_jepa_td3run4_fullft \
  --save_every 80 \
  --jepa_ckpt /data/patrick/16831RL/checkpoints/mwh_cnn_jepa_coswarm_e200_lr1p5en3_wmup0p1_nmb5_varw0p8_std1_covw1_mask0p6_td3_run4.pt

# GRPO Agent
# 1 run grpo agent training
CUDA_VISIBLE_DEVICES=5 python -m scripts.train_grpo_basic \
  --scenario basic \
  --action_space usual \
  --total_iterations 200 \
  --steps_per_iteration 8192 \
  --batch_size 128 \
  --learning_rate 5e-5 \
  --clip_coef 0.08 \
  --entropy_coef 0.01 \
  --grpo_epochs 2 \
  --grpo_group_size 16 \
  --grpo_target_kl 0.015 \
  --grpo_kl_beta 0.01 \
  --grpo_use_return_per_step \
  --grpo_use_std_norm \
  --grpo_adv_clip 3.0 \
  --grpo_ref_mode ema \
  --grpo_ref_tau 0.005 \
  --eval_deterministic \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name basic_grpo_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_basic_grpo \
  --checkpoint_dir /data/patrick/16831RL/checkpoints \
  --checkpoint_name basic_grpo \
  --save_every 80

CUDA_VISIBLE_DEVICES=5 python -m scripts.train_grpo_basic \
  --scenario my_way_home \
  --action_space no_shoot \
  --total_iterations 300 \
  --steps_per_iteration 16384 \
  --batch_size 256 \
  --learning_rate 5e-5 \
  --clip_coef 0.05 \
  --entropy_coef 0.02 \
  --grpo_epochs 2 \
  --grpo_group_size 32 \
  --grpo_target_kl 0.02 \
  --grpo_kl_beta 0.02 \
  --grpo_use_return_per_step \
  --grpo_adv_clip 5.0 \
  --grpo_ref_mode ema \
  --grpo_ref_tau 0.01 \
  --eval_episodes 10 \
  --eval_interval 1 \
  --eval_log_dir /data/patrick/16831RL/logs \
  --eval_log_name mwh_grpo_eval.npz \
  --tb_log_dir /data/patrick/16831RL/logs/tb_mwh_grpo \
  --checkpoint_dir /data/patrick/16831RL/checkpoints_mwh \
  --checkpoint_name mwh_grpo \
  --save_every 80

# Plot the learning curves
./scripts/plot_tb_basic_avg_return.sh 2 

./scripts/plot_tb_mwh_avg_return.sh 2 

./scripts/plot_tb_basic_td3frozen_trans_return.sh 2 

./scripts/plot_tb_mwh_td3frozen_trans_return.sh 2 
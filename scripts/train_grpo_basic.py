# scripts/train_grpo_basic.py
import argparse
import os

from env.doom_env import DoomEnv
from train.rl_trainer import RLTrainer
from configs.grpo_config import GRPOConfig


def main():
    parser = argparse.ArgumentParser()

    # ----- Environment & scenario -----
    parser.add_argument(
        "--scenario",
        type=str,
        default="basic",
        choices=["basic", "my_way_home"],
        help="ViZDoom scenario to run.",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="no_shoot",
        choices=["usual", "no_shoot"],
        help="ViZDoom action space to run.",
    )
    parser.add_argument("--frame_repeat", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument(
        "--base_res",
        type=str,
        default="320x240",
        choices=["160x120", "320x240", "800x600"],
        help="Native ViZDoom render resolution.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=42)

    # ----- GRPO training hyper-parameters -----
    parser.add_argument("--total_iterations", type=int, default=200)
    parser.add_argument("--refine_iterations", type=int, default=None)
    parser.add_argument("--steps_per_iteration", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # ----- GRPO knobs -----
    parser.add_argument("--grpo_group_size", type=int, default=8, help="g: number of trajectories per group.")
    parser.add_argument(
        "--grpo_use_std_norm",
        action="store_true",
        default=False,
        help="If set, A_i = (R_i - mean_group) / std_group. (default True)",
    )
    parser.add_argument(
        "--grpo_adv_norm_in_update",
        action="store_true",
        default=False,
        help="Additionally normalize advantages inside update() (usually keep False).",
    )

    parser.add_argument(
        "--grpo_length_normalize",
        action="store_true",
        default=False,
        help="Normalize each Ai on length of the trajectory",
    )

    parser.add_argument(
        "--grpo_use_return_per_step",
        action="store_true",
        default=False,
        help="Divide each return by the length of the trajectory",
    )

    parser.add_argument("--grpo_epochs", type=int, default=2,
                        help="Number of epochs in update() of GRPO agent.")

    parser.add_argument("--grpo_adv_clip", type=float, default=5.0,
                        help="clipped values of Ai to prevent from drifting.")

    parser.add_argument("--grpo_target_kl", type=float, default=0.01,
                        help="KL target value to early stopping in update() of GRPO agent.")

    parser.add_argument("--grpo_kl_beta", type=float, default=0.02,
                        help="coefficient of KL divergence term in update() of GRPO agent.")

    parser.add_argument("--grpo_ref_mode", type=str, default="ema", choices=["ema", "hard"])
    parser.add_argument("--grpo_ref_tau", type=float, default=0.01)
    parser.add_argument("--grpo_ref_update_every", type=int, default=10)
    parser.add_argument("--grpo_eps", type=float, default=1e-8)

    # ----- Evaluation settings -----
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument(
        "--eval_log_dir",
        type=str,
        default="./logs",
        help="Directory to store eval npz logs.",
    )
    parser.add_argument(
        "--eval_log_name",
        type=str,
        default="basic_grpo_eval.npz",
        help="Filename for eval npz log.",
    )
    parser.add_argument(
        "--tb_log_dir",
        type=str,
        default="./logs/tb_basic_grpo",
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--eval_deterministic",
        action="store_true",
        default=False,
        help="Use deterministic policy during eval (default: False).",
    )
    # ----- Checkpoint settings -----
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save GRPO checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="basic_grpo",
        help="Checkpoint file prefix (e.g., basic_grpo_final.pt).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="0 = only final checkpoint; >0 = also every N iterations.",
    )

    parser.add_argument(
        "--feat_dim",
        type=int,
        default=256,
        help="feature dimension output from encoder part.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="cnn",
        choices=["cnn"],
    ) #GRPO: cnn only now

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the image encoder.",
    )

    # ----- RND options -----
    parser.add_argument(
        "--use_rnd",
        action="store_true",
        help="Enable RND intrinsic reward for exploration.",
    )
    parser.add_argument("--rnd_int_coef", type=float, default=1.0,
                        help="Coefficient on intrinsic (RND) reward.")
    parser.add_argument("--rnd_ext_coef", type=float, default=1.0,
                        help="Coefficient on extrinsic reward when mixing.")
    parser.add_argument("--rnd_gamma", type=float, default=0.99,
                        help="EMA factor for intrinsic reward std.")
    parser.add_argument("--rnd_lr", type=float, default=1e-4,
                        help="Learning rate for RND predictor.")
    parser.add_argument("--rnd_weight_decay", type=float, default=1e-4,
                        help="Weight decay for RND predictor AdamW.")
    parser.add_argument("--rnd_batch_size", type=int, default=256,
                        help="RND predictor batch size per rollout.")
    parser.add_argument("--rnd_epochs", type=int, default=1,
                        help="Number of passes over RND data per rollout.")

    parser.add_argument(
        "--rnd_int_decay",
        action="store_true",
        help="If set, linearly decay rnd_int_coef from its initial value to 0 over training.",
    )
    parser.add_argument(
        "--jepa_ckpt",
        type=str,
        default=None,
        help="Optional path to JEPA checkpoint (.pt) whose encoder_state_dict will initialize the CNN backbone.",
    )
    parser.add_argument(
        "--jepa_partial_unfreeze",
        type=int, 
        default=0,
        help="# 0 = all frozen conv block, only head will be unfrozen; >0 means freezing last k conv block + head. For JEPA-pretrained CNN backbone."
    )

    args = parser.parse_args()

    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    eval_log_path = os.path.join(args.eval_log_dir, args.eval_log_name)

    train_env = DoomEnv(
        scenario=args.scenario,
        action_space=args.action_space,
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.seed,
        window_visible=False,
        sound_enabled=False,
        base_res=args.base_res,
    )

    eval_env = DoomEnv(
        scenario=args.scenario,
        action_space=args.action_space,
        frame_repeat=args.frame_repeat,
        frame_stack=args.frame_stack,
        width=args.width,
        height=args.height,
        seed=args.eval_seed,
        window_visible=False,
        sound_enabled=False,
        base_res=args.base_res,
    )

    # ----- GRPO config -----
    config = GRPOConfig(
        total_iterations=args.total_iterations,
        refine_iterations=args.refine_iterations,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        eval_log_path=eval_log_path,
        log_dir=args.tb_log_dir,
        eval_deterministic=args.eval_deterministic,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        save_every=args.save_every,
        backbone=args.backbone,
        feat_dim=args.feat_dim,
        freeze_backbone=args.freeze_backbone,
        use_rnd=args.use_rnd,
        rnd_int_coef=args.rnd_int_coef,
        rnd_ext_coef=args.rnd_ext_coef,
        rnd_gamma=args.rnd_gamma,
        rnd_lr=args.rnd_lr,
        rnd_weight_decay=args.rnd_weight_decay,
        rnd_batch_size=args.rnd_batch_size,
        rnd_epochs=args.rnd_epochs,
        rnd_int_decay=args.rnd_int_decay,
        jepa_ckpt=args.jepa_ckpt,
        jepa_partial_unfreeze=args.jepa_partial_unfreeze,
        # GRPO config
        grpo_group_size=args.grpo_group_size,
        grpo_use_std_norm=args.grpo_use_std_norm,
        grpo_adv_norm_in_update=args.grpo_adv_norm_in_update,
        grpo_use_return_per_step=args.grpo_use_return_per_step,
        grpo_length_normalize=args.grpo_length_normalize,
        grpo_adv_clip=args.grpo_adv_clip,
        grpo_kl_beta=args.grpo_kl_beta,
        grpo_target_kl=args.grpo_target_kl,
        grpo_epochs=args.grpo_epochs,
        grpo_ref_mode=args.grpo_ref_mode,
        grpo_ref_tau=args.grpo_ref_tau,
        grpo_ref_update_every=args.grpo_ref_update_every,
        grpo_eps=args.grpo_eps,
    )

    trainer = RLTrainer(
        env=train_env,
        eval_env=eval_env,
        agent_type="grpo",
        config=config,
        device=None,
    )
    trainer.train()


if __name__ == "__main__":
    main()

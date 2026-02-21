# configs/grpo_config.py
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    total_iterations: int = 200
    refine_iterations: int = None
    steps_per_iteration: int = 8192
    batch_size: int = 64
    learning_rate: float = 1e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    grpo_epochs: int = 4
    clip_coef: float = 0.1
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    eval_episodes: int = 10
    eval_interval: int = 1
    eval_log_path: str = "./logs/eval_returns.npz"

    # TensorBoard log directory
    log_dir: str = "./logs/tb_grpo"

    # ---- NEW: model checkpoint options ----
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "grpo_basic"   # prefix
    save_every: int = 0                  # 0 = only final, >0 = also every N iterations

    # whether to use deterministic policy during evaluation
    eval_deterministic: bool = True

    # encoder model
    feat_dim: int = 256
    backbone: str = "cnn" # "cnn" / "dinov2" / "dinov3"
    freeze_backbone: bool = False

    # ---------------------------
    # RND exploration (optional)
    # ---------------------------
    # When False, RLTrainer behaves exactly as before (no intrinsic reward).
    use_rnd: bool = False

    # Reward mixing:
    # total_reward = rnd_ext_coef * extrinsic + rnd_int_coef * normalized_intrinsic
    rnd_int_coef: float = 1.0
    rnd_ext_coef: float = 1.0

    # EMA smoothing for std of intrinsic reward
    rnd_gamma: float = 0.99

    # RND optimizer hyperparameters
    rnd_lr: float = 1e-4
    rnd_weight_decay: float = 1e-4

    # RND training schedule per rollout
    rnd_batch_size: int = 256
    rnd_epochs: int = 1

    # Whether to linearly decay rnd_int_coef from its initial value to 0
    rnd_int_decay: bool = False

    # ---------------------------
    # JEPA-pretrained encoder
    # ---------------------------
    jepa_ckpt: str | None = None     # e.g. "/data/.../mwh_cnn_jepa_coswarm_...pt"
    jepa_partial_unfreeze: int = 0   # 0 = all frozen conv block, only head will be unfrozen；>0 means freezing last k conv block + head

    # ---------------------------
    # GRPO (on-policy, trajectory-level advantage)
    # ---------------------------
    # group size g: number of trajectories in a group to compute baseline
    grpo_group_size: int = 8

    # If True: A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
    # If False: A_i = (R_i - mean(R_group))
    grpo_use_std_norm: bool = True

    # Numerical stability
    grpo_eps: float = 1e-8

    # Optional: additionally normalize advantages inside update()
    # Usually keep False because A_i already standardized at group level.
    grpo_adv_norm_in_update: bool = False    

    # --- stability knobs ---
    # Use return-per-step for grouping baseline (recommended for ViZDoom w/ living penalty)
    grpo_use_return_per_step: bool = True

    # Length-normalize per-timestep advantage: adv_t = A_i / T_i
    grpo_length_normalize: bool = False

    # Clip trajectory-level advantage after z-score (prevents exploding updates)
    grpo_adv_clip: float = 5.0  # try 3-10

    # KL penalty strength (beta). 0.0 = disable KL penalty
    grpo_kl_beta: float = 0.01

    # Early stop update if policy changes too much in one iteration
    grpo_target_kl: float | None = 0.02  # try 0.02; set None to disable

    # Reference policy update mode
    # "ema": ref <- (1-tau)*ref + tau*policy  (every update call)
    # "hard": ref <- policy                   (every grpo_ref_update_every updates)
    grpo_ref_mode: str = "ema"
    grpo_ref_tau: float = 0.01
    grpo_ref_update_every: int = 10   # used only when mode == "hard"
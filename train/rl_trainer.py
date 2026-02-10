# train/rl_trainer.py
from collections import OrderedDict
from typing import Optional, Dict, List
from tqdm.auto import trange
import time

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train.replay_buffer import RolloutBuffer
from eval.evaluation import evaluate_policy, EvalLogger
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
from agents.grpo_agent import GRPOAgent
from configs.ppo_config import PPOConfig


class RLTrainer:
    """
    Generic RL trainer that can work with different agents.
    For now, it implements PPO training for PPOAgent.

    Optional RND integration:
      - If config.use_rnd is False (default), behavior is identical
        to the original RLTrainer (no intrinsic reward).
      - If config.use_rnd is True and agent_type == "ppo", we use a
        Random Network Distillation model to produce intrinsic rewards
        and shape the reward before computing returns for PPO.
    """

    def __init__(
        self,
        env,
        eval_env,
        agent_type: str,
        config: PPOConfig,
        device: Optional[str] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Observation shape from env: (frame_stack, 3, H, W)
        self.obs_shape = env.observation_shape
        self.n_actions = env.action_space_n

        self.agent_type = agent_type.lower()
        self.agent = self._build_agent(self.agent_type).to(self.device)

        # Optimizer: only needed for trainable agents (e.g., PPO).
        if agent_type == "random":
            self.optimizer = None
        else:
            # AdamW for PPO
            self.optimizer = torch.optim.AdamW(
                self.agent.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-4,
            )

        self.buffer = RolloutBuffer(
            buffer_size=self.config.steps_per_iteration,
            obs_shape=self.obs_shape,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        self.eval_logger = EvalLogger()

        # Training progress
        self.total_envsteps = 0
        self.start_time = time.time()
        self.initial_return: Optional[float] = None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # ------------------------
        # Optional RND components
        # ------------------------
        # Ensure backward compatibility with older configs that do not
        # have RND fields.
        self.use_rnd = bool(getattr(self.config, "use_rnd", False)) and (self.agent_type == "ppo")

        # Store initial intrinsic coefficient & whether to decay it
        self.rnd_int_coef_initial: float = float(getattr(self.config, "rnd_int_coef", 0.0))
        self.rnd_int_decay: bool = bool(getattr(self.config, "rnd_int_decay", False))

        if self.use_rnd:
            from exploration.rnd_model import RNDModel  # imported only when needed

            frame_stack, c, h, w = self.obs_shape
            in_channels = frame_stack * c
            rnd_lr = float(getattr(self.config, "rnd_lr", 1e-4))
            rnd_weight_decay = float(getattr(self.config, "rnd_weight_decay", 1e-4))

            self.rnd_model = RNDModel(
                obs_shape=(in_channels, h, w),
                lr=rnd_lr,
                weight_decay=rnd_weight_decay,
                device=str(self.device),
            )
            # Running std for intrinsic reward normalization
            self.rnd_running_std: float = 1.0
        else:
            self.rnd_model = None
            self.rnd_running_std = 1.0

    def _build_agent(self, agent_type: str):
        """
        Simple agent factory. Extend this to add more agent types.
        """
        if agent_type == "ppo":
            return PPOAgent(
                obs_shape=self.obs_shape,
                n_actions=self.n_actions,
                feat_dim=self.config.feat_dim,
                backbone=self.config.backbone,        # "cnn" / "dinov2" / "dinov3"
                freeze_backbone=self.config.freeze_backbone,
                jepa_ckpt_path=getattr(self.config, "jepa_ckpt", None),
                jepa_partial_unfreeze=getattr(self.config, "jepa_partial_unfreeze", 0),
            )
        elif agent_type == "grpo":
            return GRPOAgent(
                obs_shape=self.obs_shape,
                n_actions=self.n_actions,
                feat_dim=self.config.feat_dim,
                backbone=self.config.backbone,        # will be "cnn" here
                freeze_backbone=self.config.freeze_backbone,
                jepa_ckpt_path=getattr(self.config, "jepa_ckpt", None),
                jepa_partial_unfreeze=getattr(self.config, "jepa_partial_unfreeze", 0),
            )            
        elif agent_type == "random":
            # Stateless random policy
            return RandomAgent(n_actions=self.n_actions)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert stacked frames from DoomEnv into a (1, C, H, W) float tensor in [0,1].
        Input:  obs is (T, 3, H, W) uint8 from DoomEnv._get_obs()
        Output: (1, T*3, H, W) float32 on self.device
        """
        obs_t = torch.from_numpy(obs)  # (T, 3, H, W)
        if obs_t.ndim != 4:
            raise ValueError(f"Expected obs with 4 dims (T,3,H,W), got {obs_t.shape}")
        t, c, h, w = obs_t.shape

        # Use reshape instead of view to handle non-contiguous tensors safely.
        obs_t = obs_t.reshape(1, t * c, h, w).float() / 255.0
        return obs_t.to(self.device)

    def collect_rollout(self) -> Dict[str, List[float] | float]:
        """
        Collect one rollout of length steps_per_iteration into the buffer.
        Also track per-episode returns and lengths for training metrics.

        If RND is enabled, we will:
          - compute intrinsic rewards for all collected states,
          - normalize them using an EMA of their std,
          - mix them with extrinsic rewards,
          - overwrite buffer.rewards with the mixed reward,
          - train the RND predictor once on this rollout.
        """
        self.buffer.reset()
        obs = self.env.reset()

        train_returns: List[float] = []
        train_ep_lens: List[int] = []
        ep_ret = 0.0
        ep_len = 0

        rnd_loss: Optional[float] = None
        rnd_int_mean: Optional[float] = None
        rnd_int_std: Optional[float] = None

        # Use tqdm to show per-iteration progress over environment steps
        for _ in trange(
            self.config.steps_per_iteration,
            desc="Collecting rollout",
            leave=False,
        ):
            obs_tensor = self._obs_to_tensor(obs)

            with torch.no_grad():
                actions, log_probs, values = self.agent.act(obs_tensor, deterministic=False)

            action = int(actions.item())
            value = float(values.item())
            log_prob = float(log_probs.item())

            next_obs, reward, done, _info = self.env.step(action)

            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            ep_ret += float(reward)
            ep_len += 1

            obs = next_obs
            if done:
                train_returns.append(ep_ret)
                train_ep_lens.append(ep_len)
                ep_ret = 0.0
                ep_len = 0
                obs = self.env.reset()

        # If final episode did not terminate within this rollout, record it as a partial episode.
        if ep_len > 0:
            train_returns.append(ep_ret)
            train_ep_lens.append(ep_len)

        # --------------------------
        # RND reward shaping (PPO)
        # --------------------------
        if self.use_rnd and self.buffer.size > 0:
            frame_stack, c, h, w = self.obs_shape
            n = self.buffer.size

            # Observations: (N, T, 3, H, W) uint8
            obs_batch = self.buffer.observations[:n]  # uint8
            # Flatten frame stack into channels: (N, C, H, W) where C = T * 3
            obs_flat = obs_batch.reshape(n, frame_stack * c, h, w)

            # Compute intrinsic reward
            with torch.no_grad():
                int_rewards = self.rnd_model.compute_intrinsic_reward(obs_flat)  # (N,)
            # On device; bring to CPU for mixing/logging
            int_rewards_cpu = int_rewards.detach().cpu()

            # Normalize intrinsic reward with EMA of std
            batch_std = float(int_rewards_cpu.std().item()) + 1e-8
            rnd_gamma = float(getattr(self.config, "rnd_gamma", 0.99))
            self.rnd_running_std = (
                rnd_gamma * self.rnd_running_std
                + (1.0 - rnd_gamma) * batch_std
            )
            norm_factor = self.rnd_running_std + 1e-8
            int_rewards_norm_cpu = int_rewards_cpu / norm_factor

            # Mix extrinsic and intrinsic
            ext_rewards = self.buffer.rewards[:n]  # (N,) float32 on CPU
            int_coef = float(getattr(self.config, "rnd_int_coef", 1.0))
            ext_coef = float(getattr(self.config, "rnd_ext_coef", 1.0))

            mixed_rewards = ext_rewards.clone()
            mask = (ext_rewards == 0.0)  # only add intrinsic reward when there is no env reward
            mixed_rewards[mask] += int_coef * int_rewards_norm_cpu[mask]
            self.buffer.rewards[:n] = mixed_rewards.to(self.buffer.rewards.device)

            # Train RND predictor on this rollout
            rnd_batch_size = int(getattr(self.config, "rnd_batch_size", 256))
            rnd_epochs = int(getattr(self.config, "rnd_epochs", 1))
            rnd_loss = float(self.rnd_model.update(
                obs_flat,
                batch_size=rnd_batch_size,
                epochs=rnd_epochs,
            ))

            rnd_int_mean = float(int_rewards_norm_cpu.mean().item())
            rnd_int_std = float(int_rewards_norm_cpu.std().item())

        # Compute returns and advantages using (possibly modified) rewards
        if self.agent_type == "grpo":
            # policy-only GRPO: last_value is unused; pass 0 for API compatibility
            last_value = 0.0
            g = int(getattr(self.config, "grpo_group_size", 8))
            use_std = bool(getattr(self.config, "grpo_use_std_norm", True))
            eps = float(getattr(self.config, "grpo_eps", 1e-8))
            self.buffer.compute_returns_and_group_advantages(
                last_value=last_value,
                group_size=g,
                use_std_norm=use_std,
                eps=eps,
                use_return_per_step=bool(getattr(self.config, "grpo_use_return_per_step", True)),
                length_normalize=bool(getattr(self.config, "grpo_length_normalize", True)),
                adv_clip=getattr(self.config, "grpo_adv_clip", 5.0),
            )
        else:
            # Bootstrap value for the last state
            obs_tensor = self._obs_to_tensor(obs)
            with torch.no_grad():
                last_value = float(self.agent.get_value(obs_tensor).item())
            self.buffer.compute_returns_and_advantages(last_value)

        self.total_envsteps += self.config.steps_per_iteration

        out: Dict[str, List[float] | float] = {
            "train_returns": train_returns,
            "train_ep_lens": train_ep_lens,
        }
        if rnd_loss is not None:
            out["RND_Loss"] = rnd_loss
            out["RND_IntReward_Mean"] = rnd_int_mean
            out["RND_IntReward_Std"] = rnd_int_std

        return out

    def update_policy(self) -> Dict[str, float]:
        """
        Perform one policy/value update step.

        This method is intentionally algorithm-agnostic:
        - For non-trainable agents (e.g., RandomAgent), this is a no-op that
          returns zero losses for logging.
        - For trainable agents (e.g., PPOAgent, future DrQv2Agent, etc.),
          the actual update logic is delegated to the agent via its `update`
          method.

        The agent is responsible for:
        - reading batches from the provided rollout buffer,
        - computing its own losses,
        - calling `optimizer.step()`,
        - and returning a dict of scalar logs.
        """
        # Non-learning agents: no update
        if self.agent_type == "random":
            return {
                "Loss_Policy": 0.0,
                "Loss_Value": 0.0,
                "Loss_Entropy": 0.0,
            }

        # Trainable agents are expected to implement `update(...)`
        if not hasattr(self.agent, "update"):
            raise AttributeError(
                f"Agent of type '{self.agent_type}' does not implement an "
                f"`update(buffer, optimizer, config)` method."
            )

        # Delegate the actual RL algorithm to the agent
        logs = self.agent.update(
            buffer=self.buffer,
            optimizer=self.optimizer,
            config=self.config,
        )

        # Ensure we always return a dict with the standard keys
        default_logs = {
            "Loss_Policy": 0.0,
            "Loss_Value": 0.0,
            "Loss_Entropy": 0.0,
        }
        if logs is None:
            return default_logs

        default_logs.update(logs)
        return default_logs

    def _log_iteration(
        self,
        iteration: int,
        train_returns: List[float],
        train_ep_lens: List[int],
        eval_returns: List[float],
        eval_ep_lens: List[int],
        train_logs: Dict[str, float],
    ) -> None:
        """
        Aggregate metrics and write to TensorBoard + stdout,
        using homework-style keys.
        """
        logs = OrderedDict()

        # Eval stats
        eval_returns_np = np.asarray(eval_returns, dtype=np.float32)
        eval_ep_lens_np = np.asarray(eval_ep_lens, dtype=np.float32)
        logs["Eval_AverageReturn"] = float(eval_returns_np.mean())
        logs["Eval_StdReturn"] = float(eval_returns_np.std())
        logs["Eval_MaxReturn"] = float(eval_returns_np.max())
        logs["Eval_MinReturn"] = float(eval_returns_np.min())
        logs["Eval_AverageEpLen"] = float(eval_ep_lens_np.mean())

        # Train stats
        train_returns_np = np.asarray(train_returns, dtype=np.float32)
        train_ep_lens_np = np.asarray(train_ep_lens, dtype=np.float32)
        logs["Train_AverageReturn"] = float(train_returns_np.mean())
        logs["Train_StdReturn"] = float(train_returns_np.std())
        logs["Train_MaxReturn"] = float(train_returns_np.max())
        logs["Train_MinReturn"] = float(train_returns_np.min())
        logs["Train_AverageEpLen"] = float(train_ep_lens_np.mean())

        logs["Train_EnvstepsSoFar"] = float(self.total_envsteps)
        logs["TimeSinceStart"] = float(time.time() - self.start_time)

        # Initial_DataCollection_AverageReturn (use first iteration's train return)
        if self.initial_return is None:
            self.initial_return = logs["Train_AverageReturn"]
        logs["Initial_DataCollection_AverageReturn"] = float(self.initial_return)

        # Add training losses from the agent (PPO)
        logs.update(train_logs)

        # If RND logs were attached to train_logs (via train()), they are already here.
        # If you want them guaranteed, you can add them in RLTrainer.train().

        # Print + TensorBoard
        for key, value in logs.items():
            print(f"{key} : {value}")
            self.writer.add_scalar(key, value, iteration)
        self.writer.flush()
        print("Done logging...\n")

        # Also push eval mean/std to EvalLogger for later plotting
        self.eval_logger.add(
            iteration,
            logs["Eval_AverageReturn"],
            logs["Eval_StdReturn"],
        )

    def save_model(self, tag: str) -> str:
        """
        Save agent checkpoint. `tag` is usually an iteration index or 'final'.
        Returns the full path to the saved file.
        """
        filename = f"{self.config.checkpoint_name}_{tag}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)

        torch.save(
            {
                "agent_state_dict": self.agent.state_dict(),
                "agent_type": self.agent_type,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
                "config": self.config,
            },
            path,
        )
        print(f"Saved model checkpoint to {path}")
        return path

    def train(self) -> None:
        """
        Main training loop:
        - collect rollouts
        - update policy
        - evaluate periodically and log mean returns vs iteration
        - optionally save checkpoints every N iterations
        """
        for iteration in range(self.config.total_iterations):
            print(f"\n\n********** Iteration {iteration} ************")

            # if RND is set, let rnd_int_coef decay to 0 through iterations
            if self.use_rnd and self.rnd_int_decay:
                # frac changes from 0 to 1
                frac = iteration / max(self.config.total_iterations - 1, 1)
                # linear decay：iteration=0, initial; last iteration,  0
                current_coef = self.rnd_int_coef_initial * (1.0 - frac)
                self.config.rnd_int_coef = current_coef
                # debug
                # print(f"[RND] iter={iteration}, rnd_int_coef={current_coef:.4f}")

            # 1) collect rollout + train metrics (and possibly RND metrics)
            rollout_info = self.collect_rollout()
            train_returns = rollout_info["train_returns"]
            train_ep_lens = rollout_info["train_ep_lens"]

            # 2) Agent update
            train_logs = self.update_policy()

            # Attach RND logs (if present) into train_logs for logging
            rnd_loss = rollout_info.get("RND_Loss")
            if rnd_loss is not None:
                train_logs["RND_Loss"] = float(rollout_info["RND_Loss"])
                train_logs["RND_IntReward_Mean"] = float(rollout_info["RND_IntReward_Mean"])
                train_logs["RND_IntReward_Std"] = float(rollout_info["RND_IntReward_Std"])

            # 3) evaluation for logging (uses env's extrinsic reward only)
            eval_returns, eval_ep_lens = evaluate_policy(
                env=self.eval_env,
                agent=self.agent,
                num_episodes=self.config.eval_episodes,
                device=self.device,
                deterministic=self.config.eval_deterministic,
                return_raw=True,
            )

            # 4) aggregate + log
            self._log_iteration(
                iteration=iteration,
                train_returns=train_returns,
                train_ep_lens=train_ep_lens,
                eval_returns=eval_returns,
                eval_ep_lens=eval_ep_lens,
                train_logs=train_logs,
            )

            # 5) optional periodic checkpoint saving
            if (
                getattr(self.config, "save_every", 0) > 0
                and (iteration + 1) % self.config.save_every == 0
            ):
                # use iteration index (1-based) in filename
                self.save_model(tag=f"iter_{iteration + 1}")

        # Save final model
        self.save_model(tag="final")

        # Save eval log (for npz + plotting)
        self.eval_logger.save(self.config.eval_log_path)
        print(f"Saved eval log to {self.config.eval_log_path}")
        self.writer.close()

# train/replay_buffer.py
from typing import Tuple, Dict, Iterator

import numpy as np
import torch


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.
    Stores transitions and computes GAE advantages and returns.
    Observations are stored as (T, 3, H, W) uint8 arrays.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        obs_shape: (frame_stack, 3, H, W)
        """
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=torch.uint8
        )
        self.actions = torch.zeros(self.buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)

        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32)

        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.pos

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        obs: (frame_stack, 3, H, W) numpy array (uint8)
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("RolloutBuffer is full; call reset() before adding more.")

        self.observations[self.pos].copy_(torch.from_numpy(obs))
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """
        Compute GAE advantages and returns in-place.
        last_value: bootstrap value for the final state after the rollout.
        """
        last_value_t = torch.tensor(last_value, dtype=torch.float32)
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0.0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value_t
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            )
            advantages[t] = last_advantage

        self.advantages[: self.size] = advantages[: self.size]
        self.returns[: self.size] = self.advantages[: self.size] + self.values[: self.size]

    def compute_returns_and_group_advantages(
        self,
        last_value: float,
        group_size: int,
        use_std_norm: bool = True,
        eps: float = 1e-8,
        use_return_per_step: bool = True,
        length_normalize: bool = True,
        adv_clip: float | None = 5.0,
    ) -> None:
        """
        GRPO-style computation:

        1) Split the rollout buffer into trajectories using `dones`.
        2) Shuffle trajectories, chunk into groups of size g.
        3) For each trajectory i in a group:
             A_i = (R_i - mean(R_group)) / (std(R_group) + eps)   if use_std_norm
                 = (R_i - mean(R_group))                         otherwise
           Assign this same A_i to every timestep in that trajectory.
        4) Compute value targets as discounted return-to-go (with bootstrap last_value if truncated).

        Note:
        - This reuses the same collected rollout for multiple PPO-style epochs.
        - `done` is necessary because within one rollout you may have multiple episodes due to resets.
        """
        n = self.size
        if n <= 0:
            return

        if group_size < 2:
            raise ValueError(f"GRPO requires group_size >= 2, got {group_size}")

        # ---- 1) collect trajectory segments and trajectory returns ----
        traj_ranges = []  # list of (start_idx, end_idx)
        traj_lens = []
        traj_scores = []

        start = 0
        ret = 0.0
        for t in range(n):
            ret += float(self.rewards[t].item())
            done_t = float(self.dones[t].item())
            if done_t >= 0.5:
                s, e = start, t
                L = e-s+1
                traj_ranges.append((start, t))
                traj_lens.append(L)

                if use_return_per_step:
                    traj_scores.append(ret / (L+1e-8))
                else:
                    traj_scores.append(ret)

                start = t + 1
                ret = 0.0

        # partial trajectory (not done) at end of rollout
        if start < n:
            s, e = start, n - 1
            L = e-s+1
            traj_ranges.append((start, n - 1))
            traj_lens.append(L)

            if use_return_per_step:
                traj_scores.append(ret / (L+1e-8))
            else:
                traj_scores.append(ret)

        num_traj = len(traj_scores)
        if num_traj < 2:
            # Not enough trajectories to form a meaningful group baseline
            self.advantages[:n].zero_()
        else:
            # ---- 2) shuffle trajectories and form groups ----
            perm = np.random.permutation(num_traj)

            groups = []
            i = 0
            while i < num_traj:
                g = perm[i : i + group_size]
                if len(g) == 1 and len(groups) > 0:
                    # avoid a singleton group by merging into previous group
                    groups[-1] = np.concatenate([groups[-1], g])
                else:
                    groups.append(g)
                i += group_size

            # ---- 3) assign trajectory-level A_i to timesteps ----
            advantages = torch.zeros_like(self.rewards)
            traj_scores_np = np.asarray(traj_scores, dtype=np.float32)

            for g in groups:
                r = traj_scores_np[g]
                mean_r = float(r.mean())
                std_r = float(r.std())

                for idx in g:
                    Ai = (traj_scores_np[idx] - mean_r)
                    if use_std_norm:
                        Ai = Ai / (std_r + eps)

                    #clip after normailzation (important)
                    if adv_clip is not None:
                        Ai = float(np.clip(Ai, -adv_clip, adv_clip))

                    s, e = traj_ranges[idx]
                    L = traj_lens[idx]

                    if length_normalize:
                        advantages[s : e + 1] = Ai / max(L, 1)
                    else:
                        advantages[s : e + 1] = float(Ai)

            self.advantages[:n] = advantages[:n]

        # ---- 4) discounted return-to-go for value target ----
        last_value_t = torch.tensor(float(last_value), dtype=torch.float32)
        returns = torch.zeros_like(self.rewards)

        for t in reversed(range(n)):
            if t == n - 1:
                next_return = last_value_t
            else:
                next_return = returns[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            returns[t] = self.rewards[t] + self.gamma * next_return * next_non_terminal

        self.returns[:n] = returns[:n]


    def get_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield minibatches as dictionaries.
        Observations are returned as float32 normalized to [0,1]
        with shape (B, C, H, W) where C = 3 * frame_stack.
        """
        inds = np.arange(self.size)
        np.random.shuffle(inds)

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_idx = inds[start:end]

            obs_batch = self.observations[batch_idx].to(self.device)  # (B, T, 3, H, W)
            b, t, c, h, w = obs_batch.shape
            obs_batch = obs_batch.reshape(b, t * c, h, w).float() / 255.0

            yield {
                "observations": obs_batch,
                "actions": self.actions[batch_idx].to(self.device),
                "log_probs": self.log_probs[batch_idx].to(self.device),
                "advantages": self.advantages[batch_idx].to(self.device),
                "returns": self.returns[batch_idx].to(self.device),
                "values": self.values[batch_idx].to(self.device),
            }

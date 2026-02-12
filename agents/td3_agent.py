#!/usr/bin/env python3
"""
td3_allinone.py

- If env.action_space is Box (continuous): runs standard TD3.
- If env.action_space is Discrete: runs a TD3-inspired discrete counterpart: Clipped Double DQN
  (twin Q networks + min target to reduce overestimation, Double-DQN next-action selection).

Dependencies:
- torch, numpy
- gymnasium or gym

Run examples:
  # Continuous control (MuJoCo-style, if installed):
  python td3_allinone.py --env HalfCheetah-v4 --steps 300000

  # Discrete control:
  python td3_allinone.py --env CartPole-v1 --steps 50000
"""

import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except Exception:
    import gym


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, discrete: bool):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

        self.discrete = discrete
        if discrete:
            self.act_buf = np.zeros((size, 1), dtype=np.int64)
        else:
            self.act_buf = np.zeros((size, act_dim), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if self.discrete:
            self.act_buf[self.ptr] = act
        else:
            self.act_buf[self.ptr] = act

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idx], device=device, dtype=torch.float32)     # (B, obs_dim)
        obs2 = torch.as_tensor(self.obs2_buf[idx], device=device, dtype=torch.float32)  # (B, obs_dim)
        rew = torch.as_tensor(self.rew_buf[idx], device=device, dtype=torch.float32)    # (B, 1)
        done = torch.as_tensor(self.done_buf[idx], device=device, dtype=torch.float32)  # (B, 1)

        if self.discrete:
            act = torch.as_tensor(self.act_buf[idx], device=device, dtype=torch.int64)  # (B, 1)
        else:
            act = torch.as_tensor(self.act_buf[idx], device=device, dtype=torch.float32)  # (B, act_dim)

        return obs, act, rew, obs2, done


# -----------------------------
# Networks
# -----------------------------
def mlp(sizes, activation=nn.ReLU, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorContinuous(nn.Module):
    """
    Deterministic actor. Outputs bounded actions via tanh then affine to [low, high].
    """
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], activation=nn.ReLU, out_activation=nn.Identity)

        low = torch.as_tensor(act_low, dtype=torch.float32)
        high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_low", low)    # (act_dim,)
        self.register_buffer("act_high", high)  # (act_dim,)

        self.register_buffer("act_mid", (high + low) * 0.5)     # (act_dim,)
        self.register_buffer("act_half", (high - low) * 0.5)    # (act_dim,)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        raw = self.net(obs)                      # (B, act_dim)
        squashed = torch.tanh(raw)               # (B, act_dim) in [-1,1]
        act = self.act_mid + self.act_half * squashed  # (B, act_dim) in [low, high]
        return act


class CriticContinuous(nn.Module):
    """
    Q(s,a) for continuous actions.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU, out_activation=nn.Identity)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim), act: (B, act_dim)
        x = torch.cat([obs, act], dim=-1)  # (B, obs_dim+act_dim)
        q = self.net(x)                    # (B, 1)
        return q


class QNetworkDiscrete(nn.Module):
    """
    Q(s) -> Q-values for all discrete actions.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions], activation=nn.ReLU, out_activation=nn.Identity)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        q_all = self.net(obs)  # (B, n_actions)
        return q_all


# -----------------------------
# TD3 (Continuous)
# -----------------------------
@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise_std: float = 0.2    # in action units relative to action range (we’ll scale)
    target_noise_clip: float = 0.5   # in action units relative to action range (we’ll scale)
    explore_noise_std: float = 0.1   # for behavior policy
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3


class TD3Continuous:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device, cfg: TD3Config):
        self.device = device
        self.cfg = cfg

        self.actor = ActorContinuous(obs_dim, act_dim, act_low, act_high).to(device)
        self.actor_t = ActorContinuous(obs_dim, act_dim, act_low, act_high).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.q1 = CriticContinuous(obs_dim, act_dim).to(device)
        self.q2 = CriticContinuous(obs_dim, act_dim).to(device)
        self.q1_t = CriticContinuous(obs_dim, act_dim).to(device)
        self.q2_t = CriticContinuous(obs_dim, act_dim).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.critic_lr)

        # store action scale for noise scaling
        act_low_t = torch.as_tensor(act_low, device=device, dtype=torch.float32)   # (act_dim,)
        act_high_t = torch.as_tensor(act_high, device=device, dtype=torch.float32) # (act_dim,)
        self.act_range = (act_high_t - act_low_t)  # (act_dim,)
        self.act_low = act_low_t
        self.act_high = act_high_t

        self.update_step = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, noise: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
        a = self.actor(obs_t).squeeze(0)  # (act_dim,)
        if noise:
            # exploration noise scaled by action range
            eps = torch.randn_like(a) * (self.cfg.explore_noise_std * self.act_range)
            a = a + eps
        a = torch.clamp(a, self.act_low, self.act_high)
        return a.cpu().numpy()

    def _polyak(self, net: nn.Module, net_t: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_t in zip(net.parameters(), net_t.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self, rb: ReplayBuffer, batch_size: int):
        self.update_step += 1
        cfg = self.cfg

        obs, act, rew, obs2, done = rb.sample(batch_size, self.device)
        # obs: (B, obs_dim), act: (B, act_dim), rew: (B,1), obs2: (B, obs_dim), done: (B,1)

        with torch.no_grad():
            # target action with smoothing
            a2 = self.actor_t(obs2)  # (B, act_dim)

            # noise scaled by action range
            noise = torch.randn_like(a2) * (cfg.target_noise_std * self.act_range)  # (B, act_dim)
            noise = torch.clamp(noise, -cfg.target_noise_clip * self.act_range, cfg.target_noise_clip * self.act_range)
            a2 = a2 + noise
            a2 = torch.clamp(a2, self.act_low, self.act_high)  # (B, act_dim)

            q1_t = self.q1_t(obs2, a2)  # (B,1)
            q2_t = self.q2_t(obs2, a2)  # (B,1)
            q_t_min = torch.min(q1_t, q2_t)  # (B,1)

            y = rew + cfg.gamma * (1.0 - done) * q_t_min  # (B,1)

        # critic update
        q1 = self.q1(obs, act)  # (B,1)
        q2 = self.q2(obs, act)  # (B,1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        info = {"q_loss": float(q_loss.detach().cpu().item())}

        # delayed actor + target updates
        if (self.update_step % cfg.policy_delay) == 0:
            a_pi = self.actor(obs)  # (B, act_dim)
            actor_loss = -self.q1(obs, a_pi).mean()  # scalar

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            # Polyak update all targets
            self._polyak(self.actor, self.actor_t, cfg.tau)
            self._polyak(self.q1, self.q1_t, cfg.tau)
            self._polyak(self.q2, self.q2_t, cfg.tau)

            info["actor_loss"] = float(actor_loss.detach().cpu().item())
        else:
            info["actor_loss"] = None

        return info


# -----------------------------
# Discrete: Clipped Double DQN (TD3-inspired)
# -----------------------------
@dataclass
class DiscreteConfig:
    gamma: float = 0.99
    tau: float = 0.005          # Polyak for target Qs
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 50000
    lr: float = 1e-3


class ClippedDoubleDQN:
    """
    Twin Q networks for discrete actions + min target to reduce overestimation.
    Next action selection uses online Q1 argmax (Double DQN style).
    No actor network exists in discrete DQN-style methods.
    """
    def __init__(self, obs_dim: int, n_actions: int, device: torch.device, cfg: DiscreteConfig):
        self.device = device
        self.cfg = cfg
        self.n_actions = n_actions

        self.q1 = QNetworkDiscrete(obs_dim, n_actions).to(device)
        self.q2 = QNetworkDiscrete(obs_dim, n_actions).to(device)
        self.q1_t = QNetworkDiscrete(obs_dim, n_actions).to(device)
        self.q2_t = QNetworkDiscrete(obs_dim, n_actions).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)

        self.total_steps = 0

    def _polyak(self, net: nn.Module, net_t: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_t in zip(net.parameters(), net_t.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def _epsilon(self):
        cfg = self.cfg
        t = self.total_steps
        frac = min(1.0, t / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + frac * (cfg.epsilon_final - cfg.epsilon_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        self.total_steps += 1
        if (not eval_mode) and (np.random.rand() < self._epsilon()):
            return np.random.randint(self.n_actions)

        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
        q_all = self.q1(obs_t)  # (1, n_actions)
        a = int(torch.argmax(q_all, dim=1).item())
        return a

    def update(self, rb: ReplayBuffer, batch_size: int):
        cfg = self.cfg
        obs, act, rew, obs2, done = rb.sample(batch_size, self.device)
        # obs: (B, obs_dim)
        # act: (B, 1) int64
        # rew: (B, 1)
        # obs2: (B, obs_dim)
        # done: (B, 1)

        with torch.no_grad():
            # online Q1 selects next action (Double DQN selection)
            q1_next_online = self.q1(obs2)                 # (B, n_actions)
            next_a = torch.argmax(q1_next_online, dim=1, keepdim=True)  # (B,1) int64

            # target Qs evaluate that action
            q1_next_t = self.q1_t(obs2).gather(1, next_a)  # (B,1)
            q2_next_t = self.q2_t(obs2).gather(1, next_a)  # (B,1)
            q_next_min = torch.min(q1_next_t, q2_next_t)   # (B,1)

            y = rew + cfg.gamma * (1.0 - done) * q_next_min  # (B,1)

        # current Q predictions
        q1_all = self.q1(obs)               # (B, n_actions)
        q2_all = self.q2(obs)               # (B, n_actions)
        q1 = q1_all.gather(1, act)          # (B,1)
        q2 = q2_all.gather(1, act)          # (B,1)

        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # Polyak update target Qs every step (common for stability)
        self._polyak(self.q1, self.q1_t, cfg.tau)
        self._polyak(self.q2, self.q2_t, cfg.tau)

        return {"q_loss": float(loss.detach().cpu().item())}


# -----------------------------
# Training / Evaluation
# -----------------------------
def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        obs, _info = out
    else:
        obs = out
    return obs


def step_env(env, action):
    out = env.step(action)
    # gymnasium: (obs2, rew, terminated, truncated, info)
    # gym: (obs2, rew, done, info)
    if len(out) == 5:
        obs2, rew, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs2, rew, done, info = out
    return obs2, float(rew), float(done), info


@torch.no_grad()
def evaluate(env, agent, episodes: int, continuous: bool):
    returns = []
    for _ in range(episodes):
        obs = reset_env(env)
        done = 0.0
        ep_ret = 0.0
        while not done:
            if continuous:
                act = agent.select_action(obs, noise=False)
            else:
                act = agent.select_action(obs, eval_mode=True)
            obs, rew, done, _ = step_env(env, act)
            ep_ret += rew
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # common
    p.add_argument("--replay_size", type=int, default=1_000_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--start_steps", type=int, default=10000)     # random actions before learning
    p.add_argument("--update_after", type=int, default=1000)     # start gradient updates after this many steps
    p.add_argument("--updates_per_step", type=int, default=1)    # gradient steps per env step
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--eval_eps", type=int, default=5)

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    obs0 = reset_env(env)
    obs_dim = int(np.prod(obs0.shape))

    # flatten observation if needed
    def obs_flat(x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    # Determine action space type
    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
        act_dim = int(np.prod(env.action_space.shape))
        act_low = env.action_space.low.reshape(-1).astype(np.float32)
        act_high = env.action_space.high.reshape(-1).astype(np.float32)

        agent = TD3Continuous(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            device=device,
            cfg=TD3Config(),
        )
        rb = ReplayBuffer(obs_dim, act_dim, args.replay_size, discrete=False)
        print(f"[Mode] Continuous TD3 | obs_dim={obs_dim}, act_dim={act_dim}")

    elif isinstance(env.action_space, gym.spaces.Discrete):
        continuous = False
        n_actions = env.action_space.n

        agent = ClippedDoubleDQN(
            obs_dim=obs_dim,
            n_actions=n_actions,
            device=device,
            cfg=DiscreteConfig(),
        )
        rb = ReplayBuffer(obs_dim, act_dim=1, size=args.replay_size, discrete=True)
        print(f"[Mode] Discrete ClippedDoubleDQN (TD3-inspired) | obs_dim={obs_dim}, n_actions={n_actions}")

    else:
        raise ValueError("Unsupported action space type. Use Box or Discrete.")

    obs = obs_flat(obs0)
    ep_ret = 0.0
    ep_len = 0
    episodes = 0

    for t in range(1, args.steps + 1):
        # choose action
        if t <= args.start_steps:
            if continuous:
                a = env.action_space.sample()
                a = np.asarray(a, dtype=np.float32).reshape(-1)
            else:
                a = env.action_space.sample()
        else:
            if continuous:
                a = agent.select_action(obs, noise=True)
            else:
                a = agent.select_action(obs, eval_mode=False)

        # env step
        obs2, rew, done, _ = step_env(env, a)
        obs2 = obs_flat(obs2)

        rb.add(obs, a if continuous else np.array([[a]], dtype=np.int64), np.array([[rew]], dtype=np.float32), obs2, np.array([[done]], dtype=np.float32))

        obs = obs2
        ep_ret += rew
        ep_len += 1

        # reset episode
        if done:
            episodes += 1
            obs = obs_flat(reset_env(env))
            ep_ret = 0.0
            ep_len = 0

        # updates
        if t >= args.update_after:
            for _ in range(args.updates_per_step):
                info = agent.update(rb, args.batch)

        # eval
        if (t % args.eval_every) == 0:
            mean_ret, std_ret = evaluate(eval_env, agent, args.eval_eps, continuous=continuous)
            print(f"Step {t:7d} | EvalReturn {mean_ret:.2f} ± {std_ret:.2f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

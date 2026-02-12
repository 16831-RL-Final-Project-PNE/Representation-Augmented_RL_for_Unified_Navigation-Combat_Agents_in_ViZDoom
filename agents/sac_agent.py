#!/usr/bin/env python3
"""
Minimal, correct Soft Actor-Critic (SAC) in PyTorch for continuous action spaces.

This version annotates SHAPES on every “new tensor” line or key operation.

Conventions:
- B: batch size
- obs_dim: observation dimension
- act_dim: action dimension
"""

import argparse
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)


# -----------------------------
# Replay Buffer
# -----------------------------

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device, discrete_action: bool):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)  # (N, obs_dim)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)  # (N, obs_dim)

        if discrete_action:
            self.act_buf = np.zeros((size, 1), dtype=np.int64)      # (N, 1) int action id
        else:
            self.act_buf = np.zeros((size, act_dim), dtype=np.float32) # (N, act_dim)

        self.rew_buf  = np.zeros((size, 1), dtype=np.float32)        # (N, 1)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)        # (N, 1)

        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr]  = obs

        if self.discrete_action:
            self.act_buf[self.ptr, 0] = int(act)
        else:
            self.act_buf[self.ptr] = act

        self.rew_buf[self.ptr]  = rew
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        act_tensor = to_tensor(self.act_buf[idxs], self.device,
                       dtype=torch.int64 if self.discrete_action else torch.float32)

        batch = dict(
            obs=to_tensor(self.obs_buf[idxs], self.device),    # (B, obs_dim)
            obs2=to_tensor(self.obs2_buf[idxs], self.device),  # (B, obs_dim)
            act=act_tensor,  # discrete: (B,1) int64; continuous: (B,act_dim) float32
            act=to_tensor(self.act_buf[idxs], self.device),    # (B, act_dim)
            rew=to_tensor(self.rew_buf[idxs], self.device),    # (B, 1)
            done=to_tensor(self.done_buf[idxs], self.device),  # (B, 1)
        )
        return batch


# -----------------------------
# Networks
# -----------------------------

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class SquashedGaussianActor(nn.Module):
    """
    Actor outputs a Gaussian over pre-squash action u, then uses tanh to squash into [-1,1],
    then affine-rescales to env bounds.

    sample(obs) returns:
    - a_env: (B, act_dim)
    - logp_a_env: (B, 1)  where logp includes tanh + scaling corrections
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_low, act_high, eps=1e-12):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        act_low = np.asarray(act_low, dtype=np.float32)    # (act_dim,)
        act_high = np.asarray(act_high, dtype=np.float32)  # (act_dim,)
        scale = (act_high - act_low) / 2.0                 # (act_dim,)
        bias = (act_high + act_low) / 2.0                  # (act_dim,)

        self.register_buffer("act_scale", torch.as_tensor(scale, dtype=torch.float32))  # (act_dim,)
        self.register_buffer("act_bias", torch.as_tensor(bias, dtype=torch.float32))    # (act_dim,)

        # Constant correction term for affine scaling: sum log(scale)
        self.register_buffer("log_act_scale_sum", torch.log(self.act_scale+eps).sum())      # scalar

    def forward(self, obs):
        # obs: (B, obs_dim)
        h = self.net(obs)                                   # (B, hidden)
        mu = self.mu_layer(h)                               # (B, act_dim)
        log_std = self.log_std_layer(h)                     # (B, act_dim)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # (B, act_dim)
        std = torch.exp(log_std)                            # (B, act_dim)
        return mu, std

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)                      # (1, obs_dim)
        obs_t = obs_t.to(self.act_scale.device)             # (1, obs_dim)

        mu, std = self.forward(obs_t)                       # mu/std: (1, act_dim)
        if deterministic:
            u = mu                                          # (1, act_dim)
        else:
            eps = torch.randn_like(mu)                      # (1, act_dim)
            u = mu + std * eps                              # (1, act_dim)

        a = torch.tanh(u)                                   # (1, act_dim) in [-1,1]
        a_env = a * self.act_scale + self.act_bias          # (1, act_dim) in env bounds
        return a_env.cpu().numpy()[0]

    def sample(self, obs):
        # obs: (B, obs_dim)
        mu, std = self.forward(obs)                         # (B, act_dim), (B, act_dim)
        dist = torch.distributions.Normal(mu, std)

        # Reparameterized sample
        u = dist.rsample()                                  # (B, act_dim)
        a = torch.tanh(u)                                   # (B, act_dim) in [-1,1]
        a_env = a * self.act_scale + self.act_bias          # (B, act_dim)

        # log prob with tanh correction
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True) # (B, 1)

        eps = 1e-6
        # log|det d(tanh)/du| = sum log(1 - tanh(u)^2)
        log_det_tanh = torch.log(1.0 - a.pow(2) + eps).sum(dim=-1, keepdim=True)  # (B, 1)

        logp_a = logp_u - log_det_tanh                       # (B, 1)

        # scaling correction for a_env = a * scale + bias
        logp_a_env = logp_a - self.log_act_scale_sum         # (B, 1)

        return a_env, logp_a_env


class QCritic(nn.Module):
    """
    Q(s,a) -> scalar
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs, act):
        # obs: (B, obs_dim), act: (B, act_dim)
        x = torch.cat([obs, act], dim=-1)                   # (B, obs_dim + act_dim)
        q = self.q(x)                                       # (B, 1)
        return q


class CategoricalActor(nn.Module):
    """
    Discrete policy: pi(a|s) = Categorical(logits)
    act() returns int action
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                       activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs):
        # obs: (B, obs_dim)
        logits = self.net(obs)  # (B, act_dim)
        return logits

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        obs_t = obs_t.to(next(self.parameters()).device)

        logits = self.forward(obs_t)  # (1, act_dim)
        if deterministic:
            a = torch.argmax(logits, dim=-1)  # (1,)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()  # (1,)
        return int(a.item())

    def dist_info(self, obs):
        logits = self.forward(obs)                 # (B, act_dim)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, act_dim)
        probs = log_probs.exp()                    # (B, act_dim)
        return probs, log_probs

class DiscreteQCritic(nn.Module):
    """
    Q(s) -> (B, act_dim), then gather by action index
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                     activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs):
        # obs: (B, obs_dim)
        return self.q(obs)  # (B, act_dim)

# -----------------------------
# SAC Agent
# -----------------------------

@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    hidden_sizes: tuple = (256, 256)
    target_entropy: float = None  # if None, set to -act_dim
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 1
    updates_per_step: int = 1
    replay_size: int = 1_000_000
    learn_alpha: bool = True


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device, cfg: SACConfig, discrete_action: bool):
        self.device = device
        self.cfg = cfg
        self.discrete_action = discrete_action

        if self.discrete_action:
            self.actor = CategoricalActor(obs_dim, act_dim, cfg.hidden_sizes).to(device)
            self.q1 = DiscreteQCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
            self.q2 = DiscreteQCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)

            self.q1_tgt = DiscreteQCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
            self.q2_tgt = DiscreteQCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        else:
            self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes, act_low, act_high).to(device)
            self.q1 = QCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
            self.q2 = QCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)

            self.q1_tgt = QCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
            self.q2_tgt = QCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)

        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        for p in self.q1_tgt.parameters():
            p.requires_grad = False
        for p in self.q2_tgt.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)

        if cfg.target_entropy is None:
            cfg.target_entropy = float(np.log(act_dim)) if self.discrete_action else float(act_dim)
        self.target_entropy = float(cfg.target_entropy)

        # We learn log_alpha so alpha stays positive (alpha = exp(log_alpha))
        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)  # scalar
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.learn_alpha = cfg.learn_alpha

    @property
    def alpha(self):
        return self.log_alpha.exp()  # scalar

    def update(self, batch):
        """
        batch:
          obs:  (B, obs_dim)
          act:  (B, act_dim)
          rew:  (B, 1)
          obs2: (B, obs_dim)
          done: (B, 1)
        """
        obs  = batch["obs"]    # (B, obs_dim)
        act  = batch["act"]    # (B, act_dim)
        rew  = batch["rew"]    # (B, 1)
        obs2 = batch["obs2"]   # (B, obs_dim)
        done = batch["done"]   # (B, 1)

        if self.discrete_action:
            # act: (B,1) int64
            act_idx = act.long()  # (B,1)

            # ----- Critic target -----
            with torch.no_grad():
                probs2, log_probs2 = self.actor.dist_info(obs2)      # (B, A), (B, A)
                q1_t_all = self.q1_tgt(obs2)                         # (B, A)
                q2_t_all = self.q2_tgt(obs2)                         # (B, A)
                min_q_t_all = torch.min(q1_t_all, q2_t_all)          # (B, A)

                v2 = (probs2 * (min_q_t_all - self.alpha * log_probs2)).sum(dim=1, keepdim=True)  # (B,1)
                backup = rew + self.cfg.gamma * (1.0 - done) * v2    # (B,1)

            # ----- Critic updates -----
            q1_all = self.q1(obs)                                    # (B, A)
            q2_all = self.q2(obs)                                    # (B, A)
            q1_val = q1_all.gather(1, act_idx)                       # (B,1)
            q2_val = q2_all.gather(1, act_idx)                       # (B,1)

            q1_loss = F.mse_loss(q1_val, backup)
            q2_loss = F.mse_loss(q2_val, backup)

            self.q1_opt.zero_grad(set_to_none=True); q1_loss.backward(); self.q1_opt.step()
            self.q2_opt.zero_grad(set_to_none=True); q2_loss.backward(); self.q2_opt.step()

            # ----- Actor update (freeze critics) -----
            for p in self.q1.parameters(): p.requires_grad_(False)
            for p in self.q2.parameters(): p.requires_grad_(False)

            probs, log_probs = self.actor.dist_info(obs)             # (B, A), (B, A)
            min_q_all = torch.min(self.q1(obs), self.q2(obs))        # (B, A)
            actor_loss = (probs * (self.alpha * log_probs - min_q_all)).sum(dim=1).mean()

            self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()

            for p in self.q1.parameters(): p.requires_grad_(True)
            for p in self.q2.parameters(): p.requires_grad_(True)

            # ----- Alpha update -----
            alpha_loss = torch.tensor(0.0, device=self.device)
            if self.learn_alpha:
                entropy_est = -(probs * log_probs).sum(dim=1, keepdim=True)  # (B,1) positive
                alpha_loss = (self.log_alpha * (entropy_est - self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()

            # ----- Target update -----
            with torch.no_grad():
                for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                    p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                    p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

            info = {
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()) if self.learn_alpha else 0.0,
                "alpha": float(self.alpha.item()),
                "avg_logp": float((probs * log_probs).sum(dim=1).mean().item()),  # optional
            }
            return info
        else:

            # -------------------------
            # Critic target y (a.k.a. backup)
            # -------------------------
            with torch.no_grad():
                a2, logp_a2 = self.actor.sample(obs2)           # a2: (B, act_dim), logp_a2: (B, 1)
                q1_t = self.q1_tgt(obs2, a2)                    # (B, 1)
                q2_t = self.q2_tgt(obs2, a2)                    # (B, 1)
                min_q_t = torch.min(q1_t, q2_t)                 # (B, 1)

                # soft Q target = min(Q_tgt) - alpha * logpi
                soft_q_t = min_q_t - self.alpha * logp_a2       # (B, 1)

                # y = r + gamma*(1-done)*soft_q_t
                backup = rew + self.cfg.gamma * (1.0 - done) * soft_q_t  # (B, 1)

            # -------------------------
            # Critic updates
            # -------------------------
            q1_val = self.q1(obs, act)                          # (B, 1)
            q2_val = self.q2(obs, act)                          # (B, 1)

            q1_loss = F.mse_loss(q1_val, backup)                # scalar
            q2_loss = F.mse_loss(q2_val, backup)                # scalar

            self.q1_opt.zero_grad(set_to_none=True)
            q1_loss.backward()
            self.q1_opt.step()

            self.q2_opt.zero_grad(set_to_none=True)
            q2_loss.backward()
            self.q2_opt.step()

            # -------------------------
            # Actor update
            # -------------------------
            for p in self.q1.parameters(): p.requires_grad_(False) # No need to update critic
            for p in self.q2.parameters(): p.requires_grad_(False) # No need to update critic
            a_new, logp_a_new = self.actor.sample(obs)          # a_new: (B, act_dim), logp_a_new: (B, 1)
            q1_new = self.q1(obs, a_new)                        # (B, 1)
            q2_new = self.q2(obs, a_new)                        # (B, 1)
            min_q_new = torch.min(q1_new, q2_new)               # (B, 1)

            actor_loss = (self.alpha * logp_a_new - min_q_new).mean()  # scalar

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            for p in self.q1.parameters(): p.requires_grad_(True) # Release
            for p in self.q2.parameters(): p.requires_grad_(True) # Release

            # -------------------------
            # Temperature (alpha) update (optional)
            # -------------------------
            alpha_loss = torch.tensor(0.0, device=self.device)  # scalar
            if self.learn_alpha:
                # We treat (logp + target_entropy) as a constant target for alpha update,
                # so we detach it to avoid backprop into actor from alpha loss.
                entropy_est = -logp_a_new

                # target_entropy is typically negative (e.g., -act_dim).
                # This update adjusts alpha so that the policy entropy stays around the desired level.
                alpha_loss = ((self.log_alpha) * (entropy_est - self.target_entropy).detach()).mean()  # scalar

                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

            # -------------------------
            # Target critic Polyak update
            # -------------------------
            with torch.no_grad():
                for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                    p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                    p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

            info = {
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()) if self.learn_alpha else 0.0,
                "alpha": float(self.alpha.item()),
                "avg_logp": float(logp_a_new.mean().item()),
            }
            return info


# -----------------------------
# Training Loop
# -----------------------------

def make_env(env_name, seed):
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    env = gym.make(env_name)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except TypeError:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env

def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs2, rew, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs2, rew, done, info = out
    return obs2, float(rew), done, info

def evaluate(agent, env_name, seed, episodes=5):
    env = make_env(env_name, seed + 12345)
    rets = []
    for _ in range(episodes):
        obs = reset_env(env)
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.actor.act(obs, deterministic=True)
            obs, rew, done, _ = step_env(env, act)
            ep_ret += rew
        rets.append(ep_ret)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_every", type=int, default=50_000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_env(args.env, args.seed)
    obs = reset_env(env)

    assert len(env.observation_space.shape) == 1
    assert len(env.action_space.shape) == 1

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    act_low = env.action_space.low
    act_high = env.action_space.high

    cfg = SACConfig()
    agent = SACAgent(obs_dim, act_dim, act_low, act_high, device, cfg)
    buf = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device)

    start_time = time.time()

    for t in range(1, args.steps + 1):
        if t < cfg.start_steps:
            act = env.action_space.sample()
        else:
            act = agent.actor.act(obs, deterministic=False)

        obs2, rew, done, _ = step_env(env, act)
        buf.store(obs, act, rew, obs2, done)
        obs = obs2 if not done else reset_env(env)

        if (t >= cfg.update_after) and (t % cfg.update_every == 0) and (buf.size >= cfg.batch_size):
            batch = buf.sample_batch(cfg.batch_size)
            info = agent.update(batch)

        if (t % args.eval_every) == 0:
            mean_ret, std_ret = evaluate(agent, args.env, args.seed, episodes=5)
            elapsed = time.time() - start_time
            print(
                f"Step {t:>8d} | EvalReturn {mean_ret:>8.1f} ± {std_ret:<6.1f} | "
                f"alpha {agent.alpha.item():.3f} | elapsed {elapsed:.1f}s"
            )

    env.close()

if __name__ == "__main__":
    main()

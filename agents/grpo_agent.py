# agents/grpo_agent.py
from typing import Dict
import copy
import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from .nn_actor_critic import ConvEncoder  # reuse current CNN encoder


class PolicyNet(nn.Module):
    """CNN encoder + discrete policy head (no value head)."""
    def __init__(self, in_channels: int, n_actions: int, feat_dim: int):
        super().__init__()
        self.enc = ConvEncoder(in_channels=in_channels, feat_dim=feat_dim)
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        logits = self.pi(h)
        return logits


class GRPOAgent(nn.Module):
    """
    More "true GRPO" variant:
      - Policy-only (no critic / no value loss)
      - KL regularization against a frozen reference policy pi_ref
      - (Optional) keep old-vs-new approx_kl as early-stop only

    RLTrainer API compatibility:
      - act(...) returns (actions, log_probs, values_dummy)
      - get_value(...) returns zeros (dummy)
      - update(...) uses buffer.advantages (trajectory-level) + ref KL
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        feat_dim: int = 256,
        backbone: str = "cnn",
        freeze_backbone: bool = True,
        jepa_ckpt_path: str | None = None,
        jepa_partial_unfreeze: int = 0,
    ):
        """
        obs_shape: (frame_stack, 3, H, W) from DoomEnv.observation_shape
        n_actions: size of discrete action space

        backbone:
            "cnn"    -> use simple Conv encoder (ActorCritic)
        """
        super().__init__()
        frame_stack, c, h, w = obs_shape
        assert c == 3, f"Expected 3 channels per frame (RGB), got {c}"
        in_channels = frame_stack * c

        self.backbone = backbone.lower()
        if self.backbone != "cnn":
            raise ValueError(f"GRPOAgent supports backbone='cnn' only, got '{backbone}'.")

        # -------- policy net (trainable) --------
        self.policy = PolicyNet(in_channels=in_channels, n_actions=n_actions, feat_dim=feat_dim)

        # Force LazyLinear init once so deepcopy is safe/stable
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            _ = self.policy(dummy)

        # ---- Load JEPA-pretrained encoder weights into policy.enc (optional) ----
        if jepa_ckpt_path is not None:
            print(f"[GRPOAgent] Loading JEPA encoder from: {jepa_ckpt_path}")
            ckpt = torch.load(jepa_ckpt_path, map_location="cpu", weights_only=True)

            if "encoder_state_dict" in ckpt:
                enc_state = ckpt["encoder_state_dict"]
            elif "jepa_state_dict" in ckpt:
                # get encoder from jepa_state_dict
                raw_state = ckpt["jepa_state_dict"]
                enc_state = {
                    k.replace("encoder.", ""): v
                    for k, v in raw_state.items()
                    if k.startswith("encoder.")
                }
            else:
                raise KeyError(
                    "JEPA checkpoint must contain 'encoder_state_dict' or 'jepa_state_dict'."
                )

            missing, unexpected = self.policy.enc.load_state_dict(enc_state, strict=False)
            if missing:
                print(f"[GRPOAgent] Warning: missing keys in encoder_state_dict: {missing}")
            if unexpected:
                print(f"[GRPOAgent] Warning: unexpected keys in encoder_state_dict: {unexpected}")

        # ---- Freeze / partial fine-tune encoder (same semantics as your PPOAgent) ----
        if freeze_backbone and (jepa_ckpt_path is not None):
            # Freeze conv trunk; keep head trainable (same behavior as your PPOAgent).
            print("[GRPOAgent] Freezing ConvEncoder.conv (JEPA features frozen).")
            for p in self.policy.enc.conv.parameters():
                p.requires_grad = False

        elif jepa_partial_unfreeze > 0 and (jepa_ckpt_path is not None):
            # Freeze all, then unfreeze last k conv layers + head.
            print(f"[GRPOAgent] Partially fine-tuning encoder: last {jepa_partial_unfreeze} conv layers + head.")
            for p in self.policy.enc.parameters():
                p.requires_grad = False

            conv_modules = []
            for m in self.policy.enc.conv.modules():
                if isinstance(m, nn.Conv2d):
                    conv_modules.append(m)

            k = min(jepa_partial_unfreeze, len(conv_modules))
            for conv in conv_modules[-k:]:
                for p in conv.parameters():
                    p.requires_grad = True

            for p in self.policy.enc.head.parameters():
                p.requires_grad = True

        # -------- reference policy (frozen) --------
        self._update_calls = 0
        # IMPORTANT: create ref AFTER loading JEPA + freeze settings, so ref matches initial policy weights.
        self.ref_policy = copy.deepcopy(self.policy)
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.ref_policy.eval()

        self.n_actions = n_actions

    # --- rollout / eval API ---
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: (B, C, H, W)
        Returns: actions, log_probs, values (all tensors with batch dimension)
        """
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # dummy value for RLTrainer compatibility
        values_dummy = torch.zeros_like(actions, dtype=torch.float32, device=actions.device)
        return actions, log_probs, values_dummy

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W)
        Returns: values (B,)
        """
        # dummy value for RLTrainer compatibility
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used during PPO update to recompute log_probs, entropy, and values
        under the current policy parameters.
        """
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return dist, log_probs, entropy

    @torch.no_grad()
    def ref_dist(self, obs: torch.Tensor):
        logits_ref = self.ref_policy(obs)
        return Categorical(logits=logits_ref)

    @torch.no_grad()
    def _ref_hard_sync_(self):
        """ref <- policy (hard copy)"""
        self.ref_policy.load_state_dict(self.policy.state_dict(), strict=True)

    @torch.no_grad()
    def _ref_ema_update_(self, tau: float):
        """ref <- (1-tau)*ref + tau*policy"""
        for p_ref, p in zip(self.ref_policy.parameters(), self.policy.parameters()):
            p_ref.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # --- update ---
    def update(self, buffer, optimizer: torch.optim.Optimizer, config) -> Dict[str, float]:
        """
        Run one GRPO policy update on the collected on-policy rollout.

        This implementation is a “policy-only GRPO” variant:
        - Optimizes a PPO-style clipped surrogate objective using trajectory-level advantages
        that were precomputed in the rollout buffer (same A for all steps within a trajectory).
        - Adds a KL regularization term to a reference policy (pi_ref) to stabilize updates:
            loss = L_clip - entropy_coef * H(pi) + kl_beta * KL(pi_new || pi_ref)
        - Optionally applies an additional per-minibatch advantage normalization
        (typically disabled if advantages are already group-standardized).
        - Optionally performs early stopping based on the old-vs-new approximate KL
        (trust-region monitor) to prevent overly large policy updates.
        - Updates the reference policy after the PPO/GRPO epochs, either by EMA or hard sync,
        depending on config (grpo_ref_mode, grpo_ref_tau, grpo_ref_update_every).

        Args:
            buffer: RolloutBuffer containing observations, actions, old log-probs,
                and precomputed advantages (trajectory-level) for this iteration.
            optimizer: Optimizer for policy parameters.
            config: GRPOConfig (or compatible) with keys such as:
                grpo_epochs, batch_size, clip_coef, entropy_coef,
                grpo_kl_beta, grpo_target_kl, grpo_ref_mode, grpo_ref_tau,
                grpo_ref_update_every, grpo_adv_norm_in_update, max_grad_norm.

        Returns:
            Dict[str, float] with scalar logs, e.g.:
            - "Loss_Policy": mean clipped surrogate loss
            - "Loss_Entropy": mean entropy
            - "GRPO_KL_Ref": mean KL(pi_new || pi_ref)
            - "GRPO_KL_OldNew": mean approx KL(old || new) used for early-stop monitoring
        """
        policy_losses = []
        entropy_losses = []
        kl_refs = []
        approx_kls_oldnew = []

        kl_beta = float(getattr(config, "grpo_kl_beta", 0.0))
        target_kl = getattr(config, "grpo_target_kl", None)  # early stop on old-vs-new (optional)

        early_stop = False
        early_stop_kl = 0.0
        ran_mb = 0

        for epoch_idx in trange(
            config.grpo_epochs,
            desc="GRPO update",
            leave=False,
        ):
            for batch in buffer.get_minibatches(config.batch_size):
                ran_mb += 1
                obs_batch = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]

                # NOTE: GRPO typically relies on group-normalized A_i already.
                # Keep this off unless you want extra normalization.
                if bool(getattr(config, "grpo_adv_norm_in_update", False)):
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist_new, new_log_probs, entropy = self.evaluate_actions(obs_batch, actions)

                # ---- PPO-style clipped objective (still okay in GRPO implementations) ----
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = entropy.mean()

                # ---- KL to reference policy: KL(pi_new || pi_ref) ----
                if kl_beta > 0.0:
                    dist_ref = self.ref_dist(obs_batch)
                    kl_ref = kl_divergence(dist_new, dist_ref).mean()
                else:
                    kl_ref = torch.zeros((), device=obs_batch.device)

                loss = policy_loss - config.entropy_coef * entropy_loss + kl_beta * kl_ref

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                optimizer.step()

                # ---- logs ----
                with torch.no_grad():
                    logits_post = self.policy(obs_batch)
                    dist_post = Categorical(logits=logits_post)
                    new_log_probs_post = dist_post.log_prob(actions)
                    approx_kl_oldnew = (old_log_probs - new_log_probs_post).mean()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))
                kl_refs.append(float(kl_ref.detach().cpu().item()))
                approx_kls_oldnew.append(float(approx_kl_oldnew.detach().cpu().item()))

                # ---- early stop on old-vs-new KL (trust-region monitor) ----
                if target_kl is not None and float(approx_kl_oldnew.item()) > float(target_kl):
                    early_stop = True
                    early_stop_kl = float(approx_kl_oldnew.item())
                    break

            if early_stop:
                break

        # ---- update reference policy (EMA / lagged) ----
        self._update_calls += 1
        ref_mode = str(getattr(config, "grpo_ref_mode", "ema")).lower()

        if ref_mode == "ema":
            tau = float(getattr(config, "grpo_ref_tau", 0.01))
            # suggest tau to be little for stability：0.005~0.02
            self._ref_ema_update_(tau)

        elif ref_mode in ("hard", "lagged"):
            every = int(getattr(config, "grpo_ref_update_every", 10))
            if every > 0 and (self._update_calls % every == 0):
                self._ref_hard_sync_()

        return {
            "Loss_Policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "Loss_Entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "GRPO_KL_Ref": float(np.mean(kl_refs)) if kl_refs else 0.0,
            "GRPO_KL_OldNew": float(np.mean(approx_kls_oldnew)) if approx_kls_oldnew else 0.0,
            "GRPO_EarlyStop": 1.0 if early_stop else 0.0,
            "GRPO_EarlyStop_KL": float(early_stop_kl),
            "GRPO_MB_Ran": float(ran_mb),
        }

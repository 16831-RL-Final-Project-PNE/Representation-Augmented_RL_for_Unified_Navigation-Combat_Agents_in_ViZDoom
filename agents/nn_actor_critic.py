# agents/nn_actor_critic.py
# Minimal CNN encoder + actor-critic heads (discrete actions).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ConvEncoder(nn.Module):
    """DQN/Nature-CNN style encoder; works for (C,H,W) where C = 3*frame_stack (RGB stacks)."""
    def __init__(self, in_channels: int = 12, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
        )
        # Lazy mapping: infer conv output size at runtime
        self.head = nn.LazyLinear(feat_dim)

    def forward(self, x):
        # x: (B,C,H,W) uint8 or float; convert to float [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        z = self.conv(x)
        z = torch.flatten(z, 1)
        z = self.head(z)
        return z


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int = 12, n_actions: int = 12, feat_dim: int = 256):
        super().__init__()
        self.enc = ConvEncoder(in_channels, feat_dim)
        # IMPORTANT: do NOT use inplace=True here, since the same feature h is shared
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, n_actions))
        self.v  = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, 1))

    def forward(self, x):
        h = self.enc(x) #(B, feat_dim)
        logits = self.pi(h) #(B, n_actions)
        value  = self.v(h).squeeze(-1) #(B,)
        return logits, value

    @torch.no_grad()
    def act(self, x, deterministic: bool = False):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if deterministic:
            a = torch.argmax(dist.probs, dim=-1)
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value

if __name__ == "__main__":
    # shape check
    net = ActorCritic(in_channels=12, n_actions=12)
    x = torch.randint(0, 256, (2, 12, 84, 84), dtype=torch.uint8)  # RGB stack of 4 => 12 channels
    logits, value = net(x)
    print("logits:", logits.shape, "value:", value.shape)  # (2,12), (2,)

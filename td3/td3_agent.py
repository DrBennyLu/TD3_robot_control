from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from td3.networks import Actor, Critic
from td3.replay_buffer import ReplayBuffer


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.total_it = 0

    def select_action(self, state: np.ndarray, explore_noise: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if explore_noise > 0:
            a = a + np.random.normal(0, explore_noise, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action)

    def update(self, batch: tuple[np.ndarray, ...]) -> dict[str, float]:
        self.total_it += 1
        s, a, r, s2, d = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch]
        r = r.view(-1, 1)
        d = d.view(-1, 1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_a = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(s2, next_a)
            tq = torch.min(tq1, tq2)
            target_q = r + (1.0 - d) * self.gamma * tq

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_v = 0.0
        if self.total_it % self.policy_delay == 0:
            na = self.actor(s)
            actor_loss = -self.critic.q1_forward(s, na).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_v = float(actor_loss.detach().cpu())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_v,
        }

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.total_it = int(ckpt.get("total_it", 0))

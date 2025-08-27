from typing import List, Dict, Tuple
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, TanhTransform, TransformedDistribution

class PPO:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        device: str,
        epsilon_clipping: float,
        gamma: float,
        lam: float,
        max_kl: float,
        actor_train_iterations: int,
        critic_train_iterations: int,
        minibatch_size: int,
        gradient_clipping: float,
        log_path: str,
        checkpoints_path: str,
        load_epoch: int | None,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device
        self.epsilon_clipping = epsilon_clipping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.actor_train_iterations = actor_train_iterations
        self.critic_train_iterations = critic_train_iterations
        self.minibatch_size = minibatch_size
        self.gradient_clipping = gradient_clipping
        self.log_path = log_path
        self.checkpoints_path = checkpoints_path
        self.load_epoch = load_epoch
        self.log_file = open(self.log_path, "a")

        if load_epoch is not None:
            self.load_checkpoint()

    def collect_trajectories(self, env, steps_per_epoch: int) -> Dict[str, Tensor]:
        self.actor.eval()
        self.critic.eval()
        total_steps = 0
        obs_states: List[Tensor] = []
        actions: List[Tensor] = []
        actions_probabilities: List[Tensor] = []
        rewards: List[float] = []
        dones: List[float] = []
        critic_values: List[float] = []

        env.resume()
        obs, _ = env.reset()

        with torch.no_grad():
            while total_steps < steps_per_epoch:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                critic_value_tensor = self.critic(obs_tensor.unsqueeze(0)).squeeze(-1).item()
                critic_values.append(float(critic_value_tensor))

                actor_distribution = self.actor(obs_tensor.unsqueeze(0))
                action = actor_distribution.sample()
                action_probability = actor_distribution.log_prob(action).sum(dim=-1)

                obs_next, reward, done, _, _ = env.step(np.clip(action.squeeze(0).cpu().numpy(), env.action_space.low, env.action_space.high))
                obs_states.append(obs_tensor)
                actions.append(action.squeeze(0))
                rewards.append(float(reward))
                actions_probabilities.append(action_probability.squeeze(0))
                dones.append(float(done))

                obs = obs_next
                total_steps += 1

                if done:
                    obs, _ = env.reset()

            env.stop()

            last_critic_value = 0.0
            if len(dones) == 0 or bool(dones[-1]) is False:
                last_critic_value = self.critic(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze(-1).item()

            return {
                "obs_states": torch.stack(obs_states),
                "actions": torch.stack(actions),
                "actions_probabilities": torch.stack(actions_probabilities),
                "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                "dones": torch.as_tensor(dones, dtype=torch.float32, device=self.device),
                "critic_values": torch.as_tensor(critic_values + [float(last_critic_value)], dtype=torch.float32, device=self.device),
            }

    def gae(self, rewards: Tensor, critic_values: Tensor, dones: Tensor, gamma: float, lam: float) -> Tuple[Tensor, Tensor]:
        number_of_steps = rewards.shape[0]
        advantages = torch.zeros(number_of_steps, dtype=torch.float32, device=rewards.device)
        gae = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
        for t in reversed(range(number_of_steps)):
            no_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * critic_values[t + 1] * no_done - critic_values[t]
            gae = delta + gamma * lam * no_done * gae
            advantages[t] = gae
        returns = advantages + critic_values[:-1]
        return advantages, returns

    def update_policy_clip(self, trajectories: Dict[str, Tensor], advantages: Tensor) -> Dict[str, float]:
        self.actor.train()
        obs_states = trajectories["obs_states"]
        actions = trajectories["actions"]
        actions_probabilities = trajectories["actions_probabilities"].detach()
        advantages_detached = advantages.detach()
        number_of_steps = obs_states.shape[0]
        steps_ids = np.arange(number_of_steps)

        advantages_detached = (advantages_detached - advantages_detached.mean()) / (advantages_detached.std() + 1e-8)

        info = {"actor_loss": 0.0, "kl": 0.0, "clipfrac": 0.0}

        for _ in range(self.actor_train_iterations):
            np.random.shuffle(steps_ids)
            clipfracs = []

            for start in range(0, number_of_steps, self.minibatch_size):
                minibatch_steps_ids = steps_ids[start:start + self.minibatch_size]
                actor_distribution_new = self.actor(obs_states[minibatch_steps_ids])
                action_probability_new = actor_distribution_new.log_prob(actions[minibatch_steps_ids])
                if action_probability_new.ndim > 1:
                    action_probability_new = action_probability_new.sum(dim=-1)

                action_probability_difference = torch.exp(action_probability_new - actions_probabilities[minibatch_steps_ids])
                unclipped = action_probability_difference * advantages_detached[minibatch_steps_ids]
                clipped = torch.clamp(action_probability_difference, 1.0 - self.epsilon_clipping, 1.0 + self.epsilon_clipping) * advantages_detached[minibatch_steps_ids]
                actor_loss = -torch.mean(torch.min(unclipped, clipped))

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping)
                self.actor_optimizer.step()

                kl = torch.mean(actions_probabilities[minibatch_steps_ids] - action_probability_new).item()
                clipfrac = torch.mean((torch.abs(action_probability_difference - 1.0) > self.epsilon_clipping).float()).item()
                clipfracs.append(clipfrac)

                info["actor_loss"] = float(actor_loss.item())
                info["kl"] = float(kl)

            info["clipfrac"] = float(np.mean(clipfracs)) if clipfracs else 0.0
            if info["kl"] > 1.5 * self.max_kl:
                break

        return info

    def fit_value_function(self, obs_states: Tensor, returns: Tensor) -> Dict[str, float]:
        self.critic.train()
        number_of_steps = obs_states.shape[0]
        steps_ids = np.arange(number_of_steps)
        info = {"critic_loss": 0.0}

        for _ in range(self.critic_train_iterations):
            np.random.shuffle(steps_ids)
            for start in range(0, number_of_steps, self.minibatch_size):
                minibatch_steps_ids = steps_ids[start:start + self.minibatch_size]
                critic_values_new = self.critic(obs_states[minibatch_steps_ids]).squeeze(-1)
                mse = 0.5 * ((critic_values_new - returns[minibatch_steps_ids]) ** 2).mean()

                self.critic_optimizer.zero_grad()
                mse.backward()
                if self.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clipping)
                self.critic_optimizer.step()

                info["critic_loss"] = float(mse.item())
        return info

    def train(self, env, steps_per_epoch: int, epochs: int):
        start_epoch = 1

        for epoch in range(start_epoch, epochs):
            trajectories = self.collect_trajectories(env, steps_per_epoch=steps_per_epoch)

            advantages, returns = self.gae(
                rewards=trajectories["rewards"], critic_values=trajectories["critic_values"],
                dones=trajectories["dones"], gamma=self.gamma, lam=self.lam
            )

            actor_info = self.update_policy_clip(trajectories, advantages)
            critic_info = self.fit_value_function(trajectories["obs_states"], returns)
            average_return = trajectories["rewards"].sum().item()
            dones = trajectories["dones"].sum().item()

            info = (
                f"Epoch {epoch:03d} | average_return={average_return:.1f} | "
                f"actor_loss={actor_info['actor_loss']:.3f} | critic_loss={critic_info['critic_loss']:.3f} | "
                f"dones = {dones} | kl={actor_info['kl']:.4f} | clipfrac={actor_info['clipfrac']:.2f} \n"
            )
            print(info)
            self.log_file.write(info)
            torch.save({
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            }, f"{self.checkpoints_path}/epoch_{epoch:03d}.pt")

    def run(self, env):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(obs_tensor).mean.squeeze(0).cpu().numpy()
                obs, _, _, _, _ = env.step(action)

    def load_checkpoint(self) -> None:
        ckpt = torch.load(f"{self.checkpoints_path}/epoch_{self.load_epoch:03d}.pt", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor) -> Normal:
        mean = self.body(x)
        log_std = self.log_std.exp().expand_as(mean)
        return Normal(mean, log_std)

class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
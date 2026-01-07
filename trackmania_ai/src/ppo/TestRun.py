import gymnasium as gym
import torch
from PPO import PPO, Actor, Critic

env = gym.make("Pendulum-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim)

ppo = PPO(
    actor=actor,
    critic=critic,
    actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
    critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
    device="cpu",
    epsilon_clipping=0.2,
    gamma=0.99,
    lam=0.95,
    max_kl=0.01,
    actor_train_iterations=10,
    critic_train_iterations=10,
    minibatch_size=64,
    gradient_clipping=0.5,
    log_path="train.log",
    checkpoints_path="../../checkpoints-test",
    load_epoch=49,
)

ppo.run(env)

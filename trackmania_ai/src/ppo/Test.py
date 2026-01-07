import gymnasium as gym
import torch
import torch.optim as optim
from PPO import PPO, Actor, Critic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("Pendulum-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim)

actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

ppo = PPO(
    actor=actor,
    critic=critic,
    actor_optimizer=actor_opt,
    critic_optimizer=critic_opt,
    device=DEVICE,
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
    load_epoch=None,
)

ppo.train(env, steps_per_epoch=4000, epochs=50)
env = gym.make("Pendulum-v1")
ppo.run(env)

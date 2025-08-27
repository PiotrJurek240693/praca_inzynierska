import numpy as np
import torch
import torch.optim as optim
from code.environment.TrackmaniaEnv import TrackmaniaEnv
from code.game_communication.Controllers import GamepadController
from code.ppo.PPO import PPO, Actor, Critic

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_file = "maps/Test.Map.txt"
    env = TrackmaniaEnv(GamepadController(), checkpoint_file)

    obs_dimension = int(np.prod(env.observation_space.shape))
    actions_dimension = int(np.prod(env.action_space.shape))

    actor = Actor(obs_dimension, actions_dimension)
    critic = Critic(obs_dimension)

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    ppo = PPO(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        epsilon_clipping=0.2,
        device=device,
        gamma=0.99,
        lam=0.95,
        max_kl=0.01,
        actor_train_iterations=80,
        critic_train_iterations=80,
        minibatch_size=128,
        gradient_clipping=0.5,
        log_path="logs/log.txt",
        checkpoints_path="checkpoints/ppo6",
        load_epoch= 270,
    )

    ppo.run(env)
import argparse
import torch
import gym
import navigation_2d

import DDPG
from flclient import train, eval_policy
import utils


def main(device, args):
    env_name = f"Navi-Acc-Lidar-Obs-Task{args.env_id}_easy-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    training_iters = 0

    while training_iters < args.max_timesteps:
        train(policy, env, replay_buffer, int(args.eval_freq), args)
        avg_reward = eval_policy(policy, env_name, args.seed, eval_episodes=100)
        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}, avg_reward: {avg_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=argparse.SUPPRESS)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--gaussian_std", default=0.1, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device, args)

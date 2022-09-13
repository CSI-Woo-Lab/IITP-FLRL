import argparse
import gym
import navigation_2d
import numpy as np
import os
import torch

import BCQ
import DDPG
import utils
from flclient import eval_policy


def main(device, args):
    env_name = f"Navi-Acc-Lidar-Obs-Task{args.env_id}_easy-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    policy = train_BCQ(state_dim, action_dim, max_action, device, env_name, args)
    avg_reward = eval_policy(policy, env_name, args.seed, eval_episodes=100)
    print(avg_reward)


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, env_name, args):
    # For saving files
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
	
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        avg_reward = eval_policy(policy, env_name, args.seed, eval_episodes=100)
        evaluations.append(avg_reward)
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}, avg_reward: {avg_reward}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=argparse.SUPPRESS)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device, args)

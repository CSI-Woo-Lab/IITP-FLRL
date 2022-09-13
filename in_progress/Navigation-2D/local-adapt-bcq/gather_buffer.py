import argparse
from pathlib import Path
import torch
import numpy as np
import gym
import navigation_2d
import DDPG
import utils
from flclient import eval_policy


def main(device, args):
    env_name = f"Navi-Acc-Lidar-Obs-Task{args.env_id}_easy-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)
    policy.load(f"result-model/round-10-weights")  # FIXME

    # For saving files
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if np.random.uniform(0, 1) < args.rand_action_p:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

	# Save final buffer and performance
    evaluations.append(eval_policy(policy, env_name, args.seed))
    np.save(f"./results/buffer_performance_{setting}", evaluations)
    replay_buffer.save(f"./buffers/{buffer_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=argparse.SUPPRESS)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--rand_action_p", default=0.0, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.0, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("buffers").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device, args)

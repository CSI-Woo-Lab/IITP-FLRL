import argparse
from collections import OrderedDict

import flwr as fl
import torch
import numpy as np
import gym
import navigation_2d
import DDPG
import utils
from tqdm import tqdm


def train(policy, env, replay_buffer, iterations, args):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

	# Interact with the environment for iterations
    for t in tqdm(range(iterations)):
        if t < args.start_timesteps:
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

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main(env_id, args):

    env_name = f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # For saving files
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    tmp = 100

    class DDPGClient(fl.client.NumPyClient):
        def get_parameters(self):
            actor_params = [val.cpu().numpy() for _, val in policy.actor.state_dict().items()]
            critic_params = [val.cpu().numpy() for _, val in policy.critic.state_dict().items()]
            params = actor_params + critic_params
            return params

        def set_parameters(self, parameters):
            len_a = len(policy.actor.state_dict().items())
            len_c = len(policy.critic.state_dict().items())
            actor_params = parameters[0:len_a]
            critic_params = parameters[len_a:len_a+len_c]

            params_dict_a = zip(policy.actor.state_dict().keys(), actor_params)
            params_dict_c = zip(policy.critic.state_dict().keys(), critic_params)
            
            state_dict_a = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_a})
            state_dict_c = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_c})

            policy.actor.load_state_dict(state_dict_a, strict=True)
            policy.critic.load_state_dict(state_dict_c, strict=True)

        def fit(self, parameters, config):
            print("=============[fitting start]================")
            train(policy, env, replay_buffer, iterations=int(args.iterations), args=args)
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}

    # Start client
    fl.client.start_numpy_client(f"[::]:{args.port}", client=DDPGClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default=argparse.SUPPRESS)
    parser.add_argument("--port", default=8080)
    parser.add_argument("--iterations", default=50e3, type=int)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--start_timesteps", default=10e3, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--gaussian_std", default=0.1, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    args = parser.parse_args()
    main(env_id=args.env_id, args=args)

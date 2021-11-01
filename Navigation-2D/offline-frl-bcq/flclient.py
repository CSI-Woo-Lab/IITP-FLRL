import argparse
from collections import OrderedDict

import flwr as fl
from tqdm import tqdm
import torch
import numpy as np
import gym
import navigation_2d
import BCQ
import DDPG
import utils
from utils import eval_policy


def train(policy, replay_buffer, args):
    training_iters = 0

    while training_iters < args.max_timesteps:
        if args.client_name == "bcq" or "bcq-naive" or "bcq-critic":
            pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        elif args.client_name == "ddpg":
            for it in tqdm(range(args.eval_freq)):
                policy.train(replay_buffer, args.batch_size)

        training_iters += args.eval_freq


def train_online(policy, replay_buffer, env, args):
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    iterations = args.max_timesteps

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


def main(env_id, args):
    if args.client_name == "ddpg-online":
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
    if args.client_name == "bcq" or "bcq-naive" or "bcq-critic":
        policy = BCQ.BCQ(state_dim, action_dim, max_action, device)
    elif args.client_name == "ddpg" or "ddpg-online":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, device)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	# Load buffer
    if args.client_name != "ddpg-online":
        replay_buffer.load(f"./buffers/{buffer_name}")

    tmp = 100

    class BCQClient(fl.client.NumPyClient):
        def get_parameters(self):
            actor_params = [val.cpu().numpy() for _, val in policy.actor.state_dict().items()]
            critic_params = [val.cpu().numpy() for _, val in policy.critic.state_dict().items()]
            # vae_params = [val.cpu().numpy() for _, val in policy.vae.state_dict().items()]  ##
            params = actor_params + critic_params# + vae_params
            return params

        def set_parameters(self, parameters):
            len_a = len(policy.actor.state_dict().items())
            len_c = len(policy.critic.state_dict().items())
            # len_v = len(policy.vae.state_dict().items())  ##
            actor_params = parameters[0:len_a]
            critic_params = parameters[len_a:len_a+len_c]
            # vae_params = parameters[len_a+len_c:len_a+len_c+len_v]  ##

            params_dict_a = zip(policy.actor.state_dict().keys(), actor_params)
            params_dict_c = zip(policy.critic.state_dict().keys(), critic_params)
            # params_dict_v = zip(policy.vae.state_dict().keys(), vae_params)  ##
            
            state_dict_a = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_a})
            state_dict_c = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_c})
            # state_dict_v = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_v})  ##

            policy.actor.load_state_dict(state_dict_a, strict=True)
            policy.critic.load_state_dict(state_dict_c, strict=True)
            # policy.vae.load_state_dict(state_dict_v, strict=True)  ##

        def fit(self, parameters, config):
            print("=========[BCQ fitting start]============")
            train(policy, replay_buffer, args)
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}

    class BCQNaiveClient(fl.client.NumPyClient):
        def get_parameters(self):
            actor_params = [val.cpu().numpy() for _, val in policy.actor.state_dict().items()]
            critic_params = [val.cpu().numpy() for _, val in policy.critic.state_dict().items()]
            vae_params = [val.cpu().numpy() for _, val in policy.vae.state_dict().items()]  ##
            params = actor_params + critic_params + vae_params
            return params

        def set_parameters(self, parameters):
            len_a = len(policy.actor.state_dict().items())
            len_c = len(policy.critic.state_dict().items())
            len_v = len(policy.vae.state_dict().items())  ##
            actor_params = parameters[0:len_a]
            critic_params = parameters[len_a:len_a+len_c]
            vae_params = parameters[len_a+len_c:len_a+len_c+len_v]  ##

            params_dict_a = zip(policy.actor.state_dict().keys(), actor_params)
            params_dict_c = zip(policy.critic.state_dict().keys(), critic_params)
            params_dict_v = zip(policy.vae.state_dict().keys(), vae_params)  ##
            
            state_dict_a = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_a})
            state_dict_c = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_c})
            state_dict_v = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_v})  ##

            policy.actor.load_state_dict(state_dict_a, strict=True)
            policy.critic.load_state_dict(state_dict_c, strict=True)
            policy.vae.load_state_dict(state_dict_v, strict=True)  ##

        def fit(self, parameters, config):
            print("=========[BCQ-naive fitting start]============")
            train(policy, replay_buffer, args)
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}

    class BCQCriticClient(fl.client.NumPyClient):
        def get_parameters(self):
            # actor_params = [val.cpu().numpy() for _, val in policy.actor.state_dict().items()]
            critic_params = [val.cpu().numpy() for _, val in policy.critic.state_dict().items()]
            # vae_params = [val.cpu().numpy() for _, val in policy.vae.state_dict().items()]  ##
            # params = actor_params + critic_params# + vae_params
            params = critic_params
            return params

        def set_parameters(self, parameters):
            # len_a = len(policy.actor.state_dict().items())
            len_c = len(policy.critic.state_dict().items())
            # len_v = len(policy.vae.state_dict().items())  ##
            # actor_params = parameters[0:len_a]
            # critic_params = parameters[len_a:len_a+len_c]
            # vae_params = parameters[len_a+len_c:len_a+len_c+len_v]  ##
            critic_params = parameters
            # params_dict_a = zip(policy.actor.state_dict().keys(), actor_params)
            params_dict_c = zip(policy.critic.state_dict().keys(), critic_params)
            # params_dict_v = zip(policy.vae.state_dict().keys(), vae_params)  ##
            
            # state_dict_a = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_a})
            state_dict_c = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_c})
            # state_dict_v = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_v})  ##

            # policy.actor.load_state_dict(state_dict_a, strict=True)
            policy.critic.load_state_dict(state_dict_c, strict=True)
            # policy.vae.load_state_dict(state_dict_v, strict=True)  ##

        def fit(self, parameters, config):
            print("=========[BCQ fitting start]============")
            train(policy, replay_buffer, args)
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}


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
            print("=========[DDPG fitting start]============")
            train(policy, replay_buffer, args)
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}

    class DDPGOnlineClient(fl.client.NumPyClient):
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
            print("=========[DDPG-online fitting start]============")
            train_online(policy, replay_buffer, env, args)  ##
            return self.get_parameters(), tmp, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            avg_reward = eval_policy(policy, env_name, args.seed)
            return float(avg_reward), tmp, {"avg_reward":float(avg_reward)}

    client_dict = {
        "bcq": BCQClient,
        "bcq-naive": BCQNaiveClient,
        "bcq-critic": BCQCriticClient,
        "ddpg": DDPGClient,
        "ddpg-online": DDPGOnlineClient,
    }
    client_cls = client_dict[args.client_name]

    # Start client
    fl.client.start_numpy_client(f"[::]:{args.port}", client=client_cls())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_name", default=argparse.SUPPRESS)
    parser.add_argument("--env_id", default=argparse.SUPPRESS)
    parser.add_argument("--port", default=8080)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--eval_freq", default=1000, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--max_timesteps", default=1000, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=100, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--gaussian_std", default=0.1, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    args = parser.parse_args()
    main(env_id=args.env_id, args=args)

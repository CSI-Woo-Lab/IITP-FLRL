from gym.envs import register
import random
from collections import OrderedDict
import csv
from tqdm import tqdm
import numpy as np
import torch
import gym, navigation_2d
from d3rlpy.algos import CQL
from flwr.common.parameter import parameters_to_weights
import cv2
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
import os
import argparse
from config import *

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def scorer(algo, env, n_trials, env_id, epsilon=0):
    episode_rewards = []
    success_rate = 0.0

    # render image saving folder create
    render_f = 'render_image/task'+ str(env_id)
    render_first = True
    try:
        if not os.path.exists (render_f):
            os.makedirs (render_f)
    except OSError:
        print ("Error, failed to the render_image folder")
        exit(1)

    for n in range(n_trials):
        observation = env.reset()
        episode_reward = 0.0
        
        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict([observation])[0]

            observation, reward, done, info = env.step(action)
            episode_reward += reward

            if render_first:
                render_first=False
                image = env.render ('rgb_array')
                cv2.imwrite (render_f + '/start_scen.png', image)

            if done:
                break

        if info['is_success'] == True:
            success_rate += 1

        episode_rewards.append(episode_reward)
        
    return episode_rewards, success_rate

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
parser.add_argument("--model_name", type=str, default=argparse.SUPPRESS)
parser.add_argument("--start", type=int, default=300)
parser.add_argument("--end", type=int, default=400)
args=parser.parse_args()

env_name = f"Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0"
env = gym.make(env_name)

#env.seed(508)
#set_random_seed(508)

model_name = args.model_name
main_dir_name= '/root/offline_federated_rl/forl_logs/' + model_name
algo = "CQL-FL"
n_client = 4
n_epoch = 2

fname = f'task{args.env_id}.csv'
csv_file = open(fname, 'w')

writer = csv.writer(csv_file, delimiter=",")
writer.writerow(["Client Algo.", "Num. Client", "Local Epoch", "Round", "Test Score"])

round_success_rate = 0.0
for round in tqdm(range(args.start, args.end+1)):

    agent = CQL()
    agent.build_with_env(env)

    model_file = np.load(
        f"{main_dir_name}/round-{round}-weights.npz",
        allow_pickle=True,
    )

    weights = parameters_to_weights(model_file["arr_0"].item())
    model_file.close()

    policy_len = len(agent.impl.policy.state_dict())
    policy_param, q_param = weights[:policy_len], weights[policy_len:]

    policy_params_dict = zip(agent.impl.policy.state_dict().keys(), policy_param)
    policy_state_dict = OrderedDict({k: torch.tensor(v) for k, v in policy_params_dict})
    agent.impl.policy.load_state_dict(policy_state_dict, strict=True)

    qfunction_params_dict = zip(agent.impl.q_function.state_dict().keys(), q_param)
    qfunction_state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in qfunction_params_dict}
    )
    agent.impl.q_function.load_state_dict(qfunction_state_dict, strict=True)
    
    _, success_rate = scorer(agent, env, 100, args.env_id)
    round_success_rate += success_rate
    
    writer.writerow([algo, n_client, n_epoch, round, success_rate])

    #for score in score_list:
    #    writer.writerow([algo, n_client, n_epoch, score, round])

print(round_success_rate/round)

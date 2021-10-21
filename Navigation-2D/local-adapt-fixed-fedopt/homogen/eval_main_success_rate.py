import argparse
import gym
import navigation_2d
from stable_baselines3 import SAC
from tqdm import tqdm

import torch

from models.local_model_sac import LocalModelSAC
torch.manual_seed(2021)


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", "-a", type=float, default=argparse.SUPPRESS)
parser.add_argument("--env_id", "-e", default=argparse.SUPPRESS)
parser.add_argument("--optimizer", "-o", default=argparse.SUPPRESS)
args = parser.parse_args()

alpha = args.alpha  # local model coef
env_id = args.env_id


max_distance_threshold = 8.48528137423857
# FIXME: main
distance_threshold_rate_list = [0.2, 0.4, 0.6]
# env1: 0.1, env2: 0.5, env3: 0.25
beta = 1 - alpha  # global model coef

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
guide_model = SAC.load(f"model-results-{args.optimizer}/rl_model_main_guide_env_{env_id}_alpha_{alpha}")
model = LocalModelSAC.load(f"model-results-{args.optimizer}/rl_model_main_env_{env_id}_alpha_{alpha}", guide_model)

# evaluation
num_epi_eval = 100
num_success = [0 for _ in range(len(distance_threshold_rate_list))]
total_total_reward = 0
for epi in tqdm(range(num_epi_eval)):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # action_g, _states = guide_model.predict(obs, deterministic=True)
        # action_l, _states2 = local_model.predict(obs, deterministic=True)
        action, _ = model.predict(obs)
        # obs, reward, done, info = env.step(beta*action_g + alpha*action_l)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            for i, dist_theshold_rate in enumerate(distance_threshold_rate_list):
                dist_theshold = max_distance_threshold * dist_theshold_rate
                if dist_theshold_rate == 0 and info["is_success"]:
                    num_success[i] += 1
                elif env.distance < dist_theshold:
                    num_success[i] += 1
    total_total_reward += total_reward

for num in num_success:
    print(num / num_epi_eval)
print(total_total_reward/num_epi_eval)

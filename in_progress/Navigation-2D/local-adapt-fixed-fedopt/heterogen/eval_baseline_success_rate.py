import argparse
import gym
import navigation_2d
from stable_baselines3 import SAC
from tqdm import tqdm

import torch
torch.manual_seed(2021)


parser = argparse.ArgumentParser()
parser.add_argument("--env_id", "-e", default=argparse.SUPPRESS)
args = parser.parse_args()

env_id = args.env_id
env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

max_distance_threshold = 8.48528137423857

# FIXME: baseline
distance_threshold_rate_list = [0.2, 0.4, 0.6]

# no adaptation
# model = SAC.load("models/model_40_fedavg")
# model = SAC.load("models/model_40_fedadagrad")
# model = SAC.load("models/model_40_fedyogi")
model = SAC.load("models/model_40_fedadam")
# model = SAC.load("rl_model_scratch_3.2M")


# adaptation (fine tune)
model.set_env(env)
model.verbose = 1
model.learn(16000)  # 0.5%

# evaluation
num_epi_eval = 100
num_success = [0 for _ in range(len(distance_threshold_rate_list))]
total_total_reward = 0
for epi in tqdm(range(num_epi_eval)):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)

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

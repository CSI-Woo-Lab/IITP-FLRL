import gym
import navigation_2d
from stable_baselines3 import SAC
from tqdm import tqdm

import torch
torch.manual_seed(2021)

max_distance_threshold = 8.48528137423857
# FIXME: baseline
env_id = 3
distance_threshold_rate_list = [0.2, 0.4, 0.6]
# env1: 0.1, env2: 0.5, env3: 0.25
alpha = 0.1  # local model coef
beta = 1 - alpha  # global model coef

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

# adaptation
# model = SAC.load(f"rl_model_from_global_env_{env_id}")

# no adaptation
# model = SAC.load("models/model_40_FL_homogen.zip")
model = SAC.load("model_results/rl_model_scratch_3.2M")

# scratch adaptation
model.set_env(env)
model.learn(10000)

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

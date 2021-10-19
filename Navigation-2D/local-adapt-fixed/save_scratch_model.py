import gym
import navigation_2d
from stable_baselines3 import SAC

import torch
torch.manual_seed(2021)

NUM_ROUNDS = 40
TIME_STEPS = 20000
NUM_CLIENTS = 4

# 320만
timesteps = NUM_ROUNDS * TIME_STEPS * NUM_CLIENTS 

env_id = 0
env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
model = SAC("MlpPolicy", env, tensorboard_log="tensorboard_scratch")
model.learn(timesteps)
model.save(f"model_results/rl_model_scratch_3.2M")

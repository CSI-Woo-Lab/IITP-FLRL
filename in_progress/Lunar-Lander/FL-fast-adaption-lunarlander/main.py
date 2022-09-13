import argparse
from stable_baselines3 import SAC

import gym
import navigation_2d

from models.local_model_sac import LocalModelSAC

import torch
torch.manual_seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", "-a", type=float, default=argparse.SUPPRESS)
parser.add_argument("--env_id", "-e", default=argparse.SUPPRESS)
args = parser.parse_args()

alpha = args.alpha  # local model coef
env_id = args.env_id

# FIXME:
PHASE_1_TIME_STEP = 5000
PHASE_2_TIME_STEP = 5000
beta = 1 - alpha  # global model coef

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

# ===
guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=100)
guide_model = guide_model.load('models/model_40_FL_homogen')
guide_model.tensorboard_log = f"tensorboard/env_{env_id}/main_global_env_{env_id}_alpha_{alpha}"
guide_model.set_env(env)
guide_model.verbose = 1

# adaptation phase 1
guide_model.learn(PHASE_1_TIME_STEP)

# ===
local_model = LocalModelSAC("MlpPolicy", env, guide_model=guide_model, verbose=1, learning_starts=100, 
                            tensorboard_log=f"tensorboard/env_{env_id}/main_local_env_{env_id}_alpha_{alpha}")
local_model.replay_buffer = guide_model.replay_buffer

# adaptation phase 2
local_model.learn(PHASE_2_TIME_STEP)

local_model.save(f"model_results/rl_model_main_env_{env_id}_alpha_{alpha}.zip", exclude=["guide_model"])
guide_model.save(f"model_results/rl_model_main_guide_env_{env_id}_alpha_{alpha}.zip")

# evaluation
for epi in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # action_g, _states = guide_model.predict(obs, deterministic=True)
        # action_l, _states2 = local_model.predict(obs, deterministic=True)

        # obs, reward, done, info = env.step(beta*action_g + alpha*action_l)
        action, _ = local_model.predict(obs)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        # env.render()
    print(total_reward)

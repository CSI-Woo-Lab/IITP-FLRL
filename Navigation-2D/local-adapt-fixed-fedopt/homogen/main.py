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
parser.add_argument("--optimizer", "-o", default=argparse.SUPPRESS)
args = parser.parse_args()

alpha = args.alpha  # local model coef
env_id = args.env_id

# FIXME:
PHASE_1_TIME_STEP = 8000
PHASE_2_TIME_STEP = 8000
beta = 1 - alpha  # global model coef

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

# ===
guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=100)
guide_model = guide_model.load(f'models/model_40_{args.optimizer}')
guide_model.tensorboard_log = f"tensorboard-{args.optimizer}/env_{env_id}/main_global_env_{env_id}_alpha_{alpha}"
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

local_model.save(f"model-results-{args.optimizer}/rl_model_main_env_{env_id}_alpha_{alpha}.zip", exclude=["guide_model"])
guide_model.save(f"model-results-{args.optimizer}/rl_model_main_guide_env_{env_id}_alpha_{alpha}.zip")

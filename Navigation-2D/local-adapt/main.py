from stable_baselines3 import SAC

import gym
import navigation_2d

from models.local_model_sac import LocalModelSAC


# FIXME:
env_id = 7
PHASE_1_TIME_STEP = 5000
PHASE_2_TIME_STEP = 5000

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

# ===
guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=100)
guide_model = guide_model.load('models/model_30_FL')
guide_model.tensorboard_log = f"tensorboard/main_global_env_{env_id}"
guide_model.set_env(env)
guide_model.verbose = 1

# adaptation phase 1
guide_model.learn(PHASE_1_TIME_STEP)

# ===
local_model = LocalModelSAC("MlpPolicy", env, guide_model=guide_model, verbose=1, learning_starts=100, tensorboard_log=f"tensorboard/main_local_env_{env_id}")
local_model.replay_buffer = guide_model.replay_buffer

# adaptation phase 2
local_model.learn(PHASE_2_TIME_STEP)

# evaluation
for epi in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action_g, _states = guide_model.predict(obs, deterministic=True)
        action_l, _states2 = local_model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step((9*action_g+action_l) / 10)
        total_reward += reward
        # env.render()
    print(total_reward)

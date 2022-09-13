from stable_baselines3 import SAC

import gym
import envs

from models.local_model_sac import LocalModelSAC


env_id = 7
PHASE_1_TIMESTEP = 5000
PHASE_2_TIMESTEP = 5000

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-No-Obstacle-v0")  # FIXME

# ===
guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=0)  # FIXME
guide_model = guide_model.load('logs-model/model_40')  # FIXME
guide_model.tensorboard_log = f"tensorboard/main_global_env_{env_id}"
guide_model.set_env(env)
guide_model.verbose = 1

# adaptation phase 1
guide_model.learn(PHASE_1_TIMESTEP)

# ===
local_model = LocalModelSAC("MlpPolicy", env, guide_model=guide_model, verbose=1, learning_starts=100, tensorboard_log=f"tensorboard/main_local_env_{env_id}")
local_model.replay_buffer = guide_model.replay_buffer

# adaptation phase 2
local_model.learn(PHASE_2_TIMESTEP)

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

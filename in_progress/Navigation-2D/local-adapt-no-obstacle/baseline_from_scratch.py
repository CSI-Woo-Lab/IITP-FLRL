from stable_baselines3 import SAC

import gym
import envs

from models.local_model_sac import LocalModelSAC


env_id = 7
TIMESTEP = 10000

env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-No-Obstacle-v0")  # FIXME

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"tensorboard/base_scratch_env_{env_id}")
model.learn(TIMESTEP)

# ===
# guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=0)
# guide_model = guide_model.load('models/model_30')
# guide_model.tensorboard_log = f"tensorboard/main_global_env_{env_id}"
# guide_model.set_env(env)
# guide_model.verbose = 1

# # adaptation phase 1
# guide_model.learn(5000)

# # ===
# local_model = LocalModelSAC("MlpPolicy", env, guide_model=guide_model, verbose=1, learning_starts=100, tensorboard_log=f"tensorboard/main_local_env_{env_id}")
# local_model.replay_buffer = guide_model.replay_buffer

# # adaptation phase 2
# local_model.learn(5000)

# evaluation
for epi in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # action_g, _states = guide_model.predict(obs, deterministic=True)
        # action_l, _states2 = local_model.predict(obs, deterministic=True)
        action, _ = model.predict(obs)
        # obs, reward, done, info = env.step((9*action_g+action_l) / 10)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        # env.render()
    print(total_reward)
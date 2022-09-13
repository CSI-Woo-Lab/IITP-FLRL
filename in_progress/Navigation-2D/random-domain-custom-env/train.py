import gym
from stable_baselines3 import SAC
import envs


goal_reset_period = 10000

env = gym.make("Navi-Acc-Lidar-Obs-RandomDomain-easy-v0", goal_reset_period=goal_reset_period)
model = SAC("MlpPolicy", env, tensorboard_log="tensorboard")

model.learn(3000000)
model.save(f"model_test_3M_{goal_reset_period}")

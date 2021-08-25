import gym
from stable_baselines3 import PPO, SAC
import envs

env = gym.make("Navi-Acc-Lidar-Obs-RandomDomain-easy-v0")
model = SAC("MlpPolicy", env, tensorboard_log="tensorboard")

model.learn(3000000)
model.save("model_test_3M")

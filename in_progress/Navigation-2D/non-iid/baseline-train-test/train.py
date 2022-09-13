import gym
import navigation_2d
from stable_baselines3 import PPO


env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="tensorboard")
model.learn(3000000)

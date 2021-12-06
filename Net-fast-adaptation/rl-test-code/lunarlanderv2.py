import gym
import torch
import numpy as np

from stable_baselines3 import *
from stable_baselines3.sac.policies import *
from stable_baselines3.common.monitor import *

env = gym.make('LunarLanderContinuous-v2')

print(torch.cuda.is_available())

model = SAC('MlpPolicy', env, verbose=1, device='cuda')
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_lander")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_lander")

'''
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
'''

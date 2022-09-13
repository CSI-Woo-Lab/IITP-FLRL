import gym
import numpy as np

from stable_baselines3 import *
from stable_baselines3.sac.policies import *

env = gym.make('Pendulum-v0')

model = PPO(MlpPolicy, env, verbose=1, device='cuda')
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

'''
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''

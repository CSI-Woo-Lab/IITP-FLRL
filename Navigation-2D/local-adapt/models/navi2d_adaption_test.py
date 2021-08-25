import gym
from gym.core import Wrapper

from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
import navigation_2d

env_id = 5
env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")

guide_model = SAC("MlpPolicy", env, verbose=1, learning_starts=0)


guide_model = guide_model.load('./model_30')
guide_model.set_env(env)
guide_model.verbose=1
guide_model.learn(5000)

local_model = SAC("MlpPolicy", env, verbose=1, learning_starts=100, )
local_model.replay_buffer = guide_model.replay_buffer
local_model.learn(5000)

obs = env.reset()
end_time = 0
time = 0
tnum = 0 
for i in range(10000):
    action, _states = guide_model.predict(obs, deterministic=True)
    action2, _states2 = local_model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step((9*action+action2) / 10)
    env.render()
    if done:
      tnum += 1
      end_time += i - time
      time = i
      obs = env.reset()
      # break

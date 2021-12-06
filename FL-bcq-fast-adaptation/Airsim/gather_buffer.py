import time
import gym
import gym_airsim
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl

# FIXME
MODEL_TRAIN_STEP = int(5e5)
MODEL_NAME = "medium"
BUFFER_SIZE = int(5e5)  # 5만개
MAP_NAME = "LandscapeMountains"

save_path = f"buffers/replay_buffer_{MODEL_NAME}_{MAP_NAME}_{BUFFER_SIZE}_steps"

env = gym.make("AirSimDrone-v0")
agent = SAC.load(f"Airsim_model_{MAP_NAME}/rl_model_{MODEL_TRAIN_STEP}_steps")
buffer = ReplayBuffer(BUFFER_SIZE, env.observation_space, env.action_space)

start_time = time.time()

timestep = 0
while timestep <= BUFFER_SIZE:
    done = False
    obs = env.reset()
    while not done:
        action, _ = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)
        info = [info]
        buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs
        timestep += 1
        if timestep > BUFFER_SIZE:
            break
    save_to_pkl(save_path, buffer, verbose=1)

time_elapsed = time.time() - start_time
time_min, time_second = divmod(time_elapsed, 60)
time_hour, time_min = divmod(time_min, 60)
print(f"Time elapsed {int(time_hour)}:{int(time_min)}:{int(time_second)}")

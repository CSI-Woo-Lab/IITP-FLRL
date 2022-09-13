import torch
import gym, navigation_2d
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

env_name = f"Navi-Acc-Lidar-Obs-Task0_easy-v0"
env = gym.make(env_name)
agent = SAC.load(
    "models_env_id_0/model-Navi-Acc-Lidar-Obs-Task0_easy-v0_100000_steps.zip"
)

replay_buffer = ReplayBuffer(
    buffer_size=100000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

num_trajectories = 10
for i in range(4):
    seed = 2022 + i
    torch.manual_seed(seed)
    env.seed(seed)
    total_reward = 0
    for _ in tqdm(range(num_trajectories)):
        done = False
        obs = env.reset()
        while not done:
            action, _ = agent.predict(obs)
            next_obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, next_obs, action, reward, done, [info])
            obs = next_obs
            total_reward += reward

    print(replay_buffer.pos)
    print(total_reward / num_trajectories)

    save_to_pkl(
        f"buffers_fl/replay-buffer-Navi-Acc-Lidar-Obs-Task0_easy-v0-ntraj-{num_trajectories}_{seed}.pkl",
        replay_buffer,
    )

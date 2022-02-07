import argparse
import torch
import gym, navigation_2d
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--num_tarj", type=int, default=argparse.SUPPRESS)
parser.add_argument("--task_num", type=int, default=argparse.SUPPRESS)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

env_name = f"Navi-Acc-Lidar-Obs-Task{args.task_num}_easy-v0"
env = gym.make(env_name)
agent = SAC.load(
    f"models_env_id_{args.task_num}/model-Navi-Acc-Lidar-Obs-Task{args.task_num}_easy-v0_100000_steps.zip"
)

replay_buffer = ReplayBuffer(
    buffer_size=100000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

total_reward = 0
for _ in tqdm(range(args.num_traj)):
    done = False
    obs = env.reset()
    while not done:
        action, _ = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, next_obs, action, reward, done, [info])
        obs = next_obs
        total_reward += reward

print(replay_buffer.pos)
print(total_reward / args.num_traj)

save_to_pkl(
    f"buffers/replay-buffer-Navi-Acc-Lidar-Obs-Task{args.task_num}_easy-v0-ntraj-{args.num_traj}.pkl",
    replay_buffer,
)

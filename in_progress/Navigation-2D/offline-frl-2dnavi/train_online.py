import argparse
import pathlib
import gym, navigation_2d
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

learning_timesteps = 100_000

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
args = parser.parse_args()

env_name = f"Navi-Acc-Lidar-Obs-Task{args.env_id}_easy-v0"
env = gym.make(env_name)

checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./models_env_id_{args.env_id}/",
    name_prefix=f"model-{env_name}",
)
agent = SAC("MlpPolicy", env, tensorboard_log="tensorboard", verbose=1)
agent.learn(
    learning_timesteps, tb_log_name=f"env-{args.env_id}", callback=checkpoint_callback
)

agent.save_replay_buffer(
    f"buffers/replay-buffer-{env_name}-{learning_timesteps}-steps.pkl"
)

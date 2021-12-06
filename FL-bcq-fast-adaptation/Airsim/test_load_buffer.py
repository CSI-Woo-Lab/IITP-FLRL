from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl


# path = "Airsim_model_ZhangJiajie/replay_buffer_medium_13.pkl"
# path = "Airsim_model_ZhangJiajie/replay_buffer.pkl"
# path = "Airsim_model_ZhangJiajie/replay_buffer_medium_1000000.pkl"
path = "buffers/replay_buffer_medium_Forest_500000_steps.pkl"
replay_buffer = load_from_pkl(path, verbose=1)
print(len(replay_buffer.actions))
print(replay_buffer.pos)

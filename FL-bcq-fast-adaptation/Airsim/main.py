import time
from stable_baselines3.common import callbacks
from driving_env import MultiRotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# FIXME
map_name = "TrapCamera"

if __name__ == '__main__':
    # start_time = time.time()

    save_path = f"Airsim_model_{map_name}"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not os.path.isdir(save_path+'/log'):
        os.mkdir(save_path+'/log')

    env = MultiRotorEnv(drone_id="Drone1", speed=2.2)
    model = SAC('MlpPolicy', env, verbose=1, buffer_size=500000, tensorboard_log=save_path+"/tensorboard")

    callback = CheckpointCallback(save_freq=10000, save_path=save_path)
    model.learn(total_timesteps=1000000, log_interval=10, callback=callback)
    model.save(save_path+"/"+save_path)
    model.save_replay_buffer(save_path+f"/replay_buffer_train_{map_name}_1000000_steps")

    # time_elapsed = time.time() - start_time
    # time_min, time_second = divmod(time_elapsed, 60)
    # time_hour, time_min = divmod(time_min, 60)
    # print(f"Time elapsed {int(time_hour)}:{int(time_min)}:{int(time_second)}")

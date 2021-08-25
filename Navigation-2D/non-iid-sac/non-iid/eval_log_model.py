import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import navigation_2d

file = open("log_eval.txt", "w")

for env_id in range(4):
    env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
    file.write(f"mean_reward_env_{env_id} = [")
    for i in range(1, 41):
        model = SAC.load(f"logs-model/model_{i}")
        mm, ss = evaluate_policy(model, env)
        print(mm, ss)
        file.write(f"{mm} ")
    file.write("]\n")

file.close()

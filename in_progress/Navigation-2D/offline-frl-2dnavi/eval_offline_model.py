import csv
from tqdm import tqdm
import gym
import navigation_2d
from d3rlpy.algos import BCQ, CQL


algorithm_name = "cql"
num_trajectory = 1000
# bcq - 10:7, 1000:657, cql - 10:2, 1000:256
num_timestep = 256
main_dir_name = "d3rlpy_logs/cql_1000_traj_100_epochs_20220113194238"

# agent = BCQ()
agent = CQL()

env_name = f"Navi-Acc-Lidar-Obs-Task0_easy-v0"
env = gym.make(env_name)

agent.build_with_env(env)
agent.from_json(f"{main_dir_name}/params.json")


csv_file = open(f"{algorithm_name}_{num_trajectory}_traj_100_epochs.csv", "w")
writer = csv.writer(csv_file, delimiter=",")

for n_epoch in tqdm(range(1, 101)):
    agent.load_model(f"{main_dir_name}/model_{num_timestep*n_epoch}.pt")
    for episode in range(100):
        done = False
        obs = env.reset()
        cummulative_reward = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            cummulative_reward += reward
        writer.writerow(
            [algorithm_name, num_trajectory, n_epoch, episode, cummulative_reward]
        )

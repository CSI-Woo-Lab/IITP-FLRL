import argparse
import gym
import navigation_2d
from stable_baselines3 import SAC
from tqdm import tqdm
import BCQ
import DDPG

import torch

torch.manual_seed(2021)

max_distance_threshold = 8.48528137423857
# FIXME: main
distance_threshold_rate_list = [0.0, 0.2, 0.4, 0.6]

def evaluate_navi_2d(model, env):
    num_epi_eval = 100
    num_success = [0 for _ in range(len(distance_threshold_rate_list))]
    total_total_reward = 0
    for epi in tqdm(range(num_epi_eval)):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # action, _ = model.predict(obs)
            action = model.select_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                for i, dist_theshold_rate in enumerate(distance_threshold_rate_list):
                    dist_theshold = max_distance_threshold * dist_theshold_rate
                    if dist_theshold_rate == 0 and info["is_success"]:
                        num_success[i] += 1
                    elif env.distance < dist_theshold:
                        num_success[i] += 1
        total_total_reward += total_reward

    for num in num_success:
        print(num / num_epi_eval)
    print(total_total_reward/num_epi_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", "-e", default=argparse.SUPPRESS)
    parser.add_argument("--log_name", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--client_name", default=argparse.SUPPRESS, choices=["bcq", "bcq-naive", "bcq-critic", "ddpg-offline", "ddpg-online"])
    parser.add_argument("--train_seed", default=0, type=int)
    args = parser.parse_args()

    env_id = args.env_id

    env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.client_name in ["bcq", "bcq-naive", "bcq-critic"]:
        model = BCQ.BCQ(state_dim, action_dim, max_action, device)
        model.load(f"result-model-round-10/result-model-{args.client_name}-{args.log_name}/round-10-weights")
    elif args.client_name in ["ddpg-offline", "ddpg-online"]:
        model = DDPG.DDPG(state_dim, action_dim, max_action, device)
        model.load(f"result-model-round-10/result-model-{args.client_name}-{args.log_name}/round-10-weights")

    if args.client_name == "bcq":
        model.load_client_vae(f"result-model-round-10/result-model-{args.client_name}-{args.log_name}/weights", args.env_id, args.train_seed)

    # evaluation
    evaluate_navi_2d(model, env)

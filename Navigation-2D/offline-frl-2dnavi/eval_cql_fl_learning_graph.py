from collections import OrderedDict
import csv
from tqdm import tqdm
import numpy as np
import torch
import gym, navigation_2d
from d3rlpy.algos import CQL
from flwr.common.parameter import parameters_to_weights


torch.manual_seed(2022)


def scorer(algo, env, n_trials, epsilon=0):

    episode_rewards = []
    for _ in range(n_trials):
        observation = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict([observation])[0]

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)
    return episode_rewards


env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")


env.seed(2022)

main_dir_name = "forl_logs/model_20220119170336_cql_Navi-Acc-Lidar-Obs-Task0_easy-v0_50_round_models"

algo = "CQL-FL"
n_client = 5
n_epoch = 5
csv_file = open("forl_cql_num_client_5_epoch_5_round_50.csv", "w")
writer = csv.writer(csv_file, delimiter=",")
writer.writerow(["Client Algo.", "Num. Client", "Local Epoch", "Test Score", "Round"])
# print("Client Algo., Num. Client, Local Epoch, Test Score")

for round in tqdm(range(1, 51)):
    agent = CQL()
    agent.build_with_env(env)

    model_file = np.load(
        f"{main_dir_name}/round-{round}-weights.npz",
        allow_pickle=True,
    )
    weights = parameters_to_weights(model_file["arr_0"].item())

    model_file.close()

    policy_len = len(agent.impl.policy.state_dict())
    policy_param, q_param = weights[:policy_len], weights[policy_len:]

    policy_params_dict = zip(agent.impl.policy.state_dict().keys(), policy_param)
    policy_state_dict = OrderedDict({k: torch.tensor(v) for k, v in policy_params_dict})
    agent.impl.policy.load_state_dict(policy_state_dict, strict=True)

    qfunction_params_dict = zip(agent.impl.q_function.state_dict().keys(), q_param)
    qfunction_state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in qfunction_params_dict}
    )
    agent.impl.q_function.load_state_dict(qfunction_state_dict, strict=True)

    score_list = scorer(agent, env, 100)

    for score in score_list:
        writer.writerow([algo, n_client, n_epoch, score, round])
        # print(f"{algo}, {n_client}, {n_epoch}, {score}")

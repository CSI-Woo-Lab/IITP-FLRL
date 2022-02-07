from collections import OrderedDict
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
agent = CQL()
agent.build_with_env(env)

env.seed(2022)


algo = "cql"
n_client = 5
n_epoch = 100
file = np.load(
    "forl_logs/model_20220104190109_cql_Navi-Acc-Lidar-Obs-Task0_easy-v0_50_round_models/round-50-weights.npz",
    allow_pickle=True,
)
# print(file.files)
# print(file["arr_1"])
weights = parameters_to_weights(file["arr_0"].item())

file.close()

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

print("Client Algo., Num. Client, Local Epoch, Test Score")
for score in score_list:
    print(f"{algo}, {n_client}, {n_epoch}, {score}")

import numpy as np
import argparse
import fl_argparser
from collections import OrderedDict
from datetime import datetime
import torch
from random import random
import gym

import navigation_2d
#import env_wrapper
import flwr as fl
from d3rlpy.algos import CQL, BCQ
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym.envs import register
from config import *

# python fl_client.py --num_trajectories {num_trajectories} --dataset_name {dataset_name}


def main(args):
    """Create model, load data, define Flower client, start Flower client."""
    # Load data
    dataset_path=f"buffers_fl/mdp-dataset-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0-ntraj-{args.num_trajectories}_{args.dataset_name}.h5"
    dataset = MDPDataset.load(dataset_path)
    train_episodes = dataset
    
    env_name = f"Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0"
    env = gym.make(env_name)
    evaluate_scorer = evaluate_on_environment(env, 10)

    algos_dict = {
        "cql": CQL,
    }
    
    algo_cls = algos_dict[args.client_algo]
    rnd = args.fl_num_round
    
    lr = 0.0025
    agent = algo_cls(actor_learning_rate=lr, critic_learning_rate=lr, imitator_learning_rate=lr, actor_optim_factory=AdamFactory(), critic_optim_factory=AdamFactory(), imitator_optim_factory=AdamFactory(), use_gpu=True)

    agent.build_with_dataset(dataset)
    len_dataset = len(train_episodes)
    date = datetime.now().strftime("%Y%m%d%H%M%S")

    class D3rlpyBaseClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [
                val.cpu().numpy() for _, val in agent.impl.policy.state_dict().items()
            ] + [
                val.cpu().numpy()
                for _, val in agent.impl.q_function.state_dict().items()
            ]
           
        def set_parameters(self, parameters):
            policy_len = len(agent.impl.policy.state_dict())

            policy_param, q_param = parameters[:policy_len], parameters[policy_len:]

            policy_params_dict = zip(
                agent.impl.policy.state_dict().keys(), policy_param
            )
            policy_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in policy_params_dict}
            )
            agent.impl.policy.load_state_dict(policy_state_dict, strict=True)

            qfunc_params_dict = zip(agent.impl.q_function.state_dict().keys(), q_param)
            qfunc_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in qfunc_params_dict}
            )
            
            agent.impl.q_function.load_state_dict(qfunc_state_dict, strict=True)

            return self.get_parameters(), len_dataset, {}

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            agent.fit(
                train_episodes,
                n_epochs=args.num_local_epoch,
                verbose=True,
                experiment_name=self.client_setting,
                logdir="forl_logs/clients",
            )
            
            return self.get_parameters(), len_dataset, {}

        def evaluate(self, parameters, config):
            #self.set_parameters(parameters)
            score = evaluate_scorer(agent)
            return score, 100, {"Score": score}
    
    class CQLClient(D3rlpyBaseClient):
        client_setting = f"{date}_CQL_{dataset_path}"

    client_dict = {
        "cql": CQLClient,
    }
    client_cls = client_dict[args.client_algo]

    # Start client
    fl.client.start_numpy_client("0.0.0.0:8080", client=client_cls())


if __name__ == "__main__":
    args = fl_argparser.args
    main(args)

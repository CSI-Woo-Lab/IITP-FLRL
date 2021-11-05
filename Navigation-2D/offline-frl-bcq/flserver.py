from typing import List, Tuple, Optional
from collections import OrderedDict
import argparse
import numpy as np
from pathlib import Path
import os
import torch
import gym
import navigation_2d
import flwr as fl
from flwr.common.parameter import parameters_to_weights
import DDPG


def main(args):
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None and aggregated_weights[0] is not None:
                parameters = aggregated_weights[0]
                weight = parameters_to_weights(parameters)

                env_name = "Navi-Acc-Lidar-Obs-Task0_easy-v0"
                env = gym.make(env_name)
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0] 
                max_action = float(env.action_space.high[0])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # device = torch.device("cpu")
                policy = DDPG.DDPG(state_dim, action_dim, max_action, device)

                len_a = len(policy.actor.state_dict().items())
                len_c = len(policy.critic.state_dict().items())
                actor_params = weight[0:len_a]
                critic_params = weight[len_a:len_a+len_c]

                params_dict_a = zip(policy.actor.state_dict().keys(), actor_params)
                params_dict_c = zip(policy.critic.state_dict().keys(), critic_params)
                
                state_dict_a = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_a})
                state_dict_c = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_c})

                policy.actor.load_state_dict(state_dict_a, strict=True)
                policy.critic.load_state_dict(state_dict_c, strict=True)

                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                log_path = f"result-model-{args.log_name}"
                Path(log_path).mkdir(parents=True, exist_ok=True)
                policy.save(os.path.join(log_path, f"round-{rnd}-weights"))
            return aggregated_weights

    # Define strategy
    strategy = SaveModelStrategy(
        min_fit_clients=args.num_client,  # Minimum number of clients to be sampled for the next round
        min_available_clients=args.num_client,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients=args.num_client,
    )

    # Start server
    fl.server.start_server(
        server_address=f"[::]:{args.port}",
        config={"num_rounds": args.num_round},
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080)
    parser.add_argument("--num_client", default=4)
    parser.add_argument("--num_round", default=50)
    parser.add_argument("--log_name", default=argparse.SUPPRESS, type=str)
    args = parser.parse_args()
    main(args)

import torch
import random
import fl_argparser
import argparse
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import flwr as fl


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG)
    np.random.seed(seed)
    # Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = Flase

def main(args):
    set_random_seed(124)

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        env_name = f"Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0"

        server_setting = f"forl_logs/model_{date}_{args.client_algo}_{env_name}_{args.fl_num_round}_round_models"
    

        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                Path(f"./{self.server_setting}").mkdir(parents=True, exist_ok=True)
                np.savez(
                    f"./{self.server_setting}/round-{rnd}-weights.npz",
                    *aggregated_weights,
                )
            return aggregated_weights

    strategy = SaveModelStrategy(
        min_fit_clients=args.fl_num_client,
        min_available_clients=args.fl_num_client,
        min_eval_clients=args.fl_num_client,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": args.fl_num_round},
        strategy=strategy,
    )

if __name__ == "__main__":

    args = fl_argparser.args

    main(args)

import argparse
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import flwr as fl


def main(args):
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        env_name = f"Navi-Acc-Lidar-Obs-Task0_easy-v0"

        server_setting = f"forl_logs/model_{date}_{args.client_algo}_{env_name}_{args.num_round}_round_models"

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
        min_fit_clients=args.num_client,
        min_available_clients=args.num_client,
        min_eval_clients=args.num_client,
    )

    # Start server
    fl.server.start_server(
        server_address=f"[::]:8080",
        config={"num_rounds": args.num_round},
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--client_algo",
        type=str,
        default=argparse.SUPPRESS,
        choices=["cql", "bcq"],
    )
    parser.add_argument("--num_client", type=int, default=argparse.SUPPRESS)

    args = parser.parse_args()
    main(args)

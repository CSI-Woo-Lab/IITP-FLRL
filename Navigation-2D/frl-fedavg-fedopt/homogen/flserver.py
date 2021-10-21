from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import argparse

import flwr as fl
from flwr.common.typing import Parameters
from flwr.common.parameter import parameters_to_weights, ndarray_to_bytes
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
import torch
torch.manual_seed(2021)

import gym
import navigation_2d
from stable_baselines3 import SAC


# FIXME:
NUM_ROUNDS = 40
NUM_CLIENTS = 4

def set_parameters(model, parameters):
    params = model.get_parameters()
    params_dict = zip(params['policy'].keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    # remove useless parameters ( 오류에 대한 예외처리 )
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del( state_dict[d] )

    # stable-baselines3의 model의 parameters를 update하기 위한 형식 맞춤
    update_dict = model.get_parameters()
    update_dict['policy'] = state_dict

    # parameters update
    model.set_parameters(update_dict)
    return model


def main(cls_name):
    class SaveModelStrategyFedAdagrad(fl.server.strategy.FedAdagrad):
        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_res = super().aggregate_fit(rnd, results, failures)
            if aggregated_res is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                aggregated_weights = aggregated_res[0]

                eval_env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")
                model = SAC("MlpPolicy", eval_env, verbose=0)
                aggregated_parameter = parameters_to_weights(aggregated_weights)
                model = set_parameters(model, aggregated_parameter)
                model.save(f"./log-model-fedadagrad/model_{rnd}.zip")  # FIXME
                
            return aggregated_weights, aggregated_res[1]

    class SaveModelStrategyFedYogi(fl.server.strategy.FedYogi):
        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_res = super().aggregate_fit(rnd, results, failures)
            if aggregated_res is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                aggregated_weights = aggregated_res[0]

                eval_env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")
                model = SAC("MlpPolicy", eval_env, verbose=0)
                aggregated_parameter = parameters_to_weights(aggregated_weights)
                model = set_parameters(model, aggregated_parameter)
                model.save(f"./log-model-fedyogi/model_{rnd}.zip")  # FIXME
                
            return aggregated_weights, aggregated_res[1]

    class SaveModelStrategyFedAdam(fl.server.strategy.FedAdam):
        def __init__(
            self,
            *,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Parameters,
            eta: float = 1e-2,  # FIXME
            eta_l: float = 1e-1,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            tau: float = 1e-9,
        ) -> None:
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_eval=fraction_eval,
                min_fit_clients=min_fit_clients,
                min_eval_clients=min_eval_clients,
                min_available_clients=min_available_clients,
                eval_fn=eval_fn,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                accept_failures=accept_failures,
                initial_parameters=initial_parameters,
                eta=eta,
                eta_l=eta_l,
                beta_1=beta_1,
                beta_2=beta_2,
                tau=tau,
            )

        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Optional[fl.common.Weights]:
            aggregated_res = super().aggregate_fit(rnd, results, failures)
            if aggregated_res[0] is None:
                return aggregated_res
            if aggregated_res is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                aggregated_weights = aggregated_res[0]

                eval_env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")
                model = SAC("MlpPolicy", eval_env, verbose=0)
                aggregated_parameter = parameters_to_weights(aggregated_weights)
                model = set_parameters(model, aggregated_parameter)
                model.save(f"./log-model-fedadam/model_{rnd}.zip")  # FIXME
                
            return aggregated_weights, aggregated_res[1]

    env_id = 0
    env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
    model = SAC("MlpPolicy", env)
    parameters = model.get_parameters()
    parameters = [ndarray_to_bytes(val.cpu().numpy()) for _, val in parameters['policy'].items()]
    params = Parameters(parameters, "")

    strategy_dict = {
        "fedadagrad": SaveModelStrategyFedAdagrad,
        "fedyogi": SaveModelStrategyFedYogi,
        "fedadam": SaveModelStrategyFedAdam,
    }
    strategy_class = strategy_dict[cls_name]
    strategy = strategy_class(
        # FIXME:
        min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=NUM_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients=NUM_CLIENTS,
        initial_parameters=params,

    )

    fl.server.start_server(config={"num_rounds": NUM_ROUNDS}, strategy=strategy, server_address=f"[::]:8001")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", "-o", default=None, type=str)
    args = parser.parse_args()
    cls_name = args.optimizer
    main(cls_name)

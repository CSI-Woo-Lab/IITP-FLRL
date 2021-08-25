import flwr as fl
from flwr.common.parameter import parameters_to_weights
import torch
from collections import OrderedDict
from typing import List, Tuple, Optional

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

import sys
import datetime
from torch.utils.tensorboard import SummaryWriter

# 참조자료
# https://flower.dev/docs/strategies.html
# https://flower.dev/docs/apiref-flwr.html#server-strategy-strategy

# eval_fn():
# https://github.com/adap/flower/blob/3db5d103b03d4fb4e1ee1579bac31e385c864d0c/src/py/flwr/server/strategy/fedavg.py#L156


# FIXME:
NUM_ROUNDS = 30


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

log_path = "./path"
rounds = 0

def main():
    eval_env = gym.make('LunarLanderContinuous-v2')
    
    
    def eval_fn(weights):
        model = SAC("MlpPolicy", eval_env, verbose=1)
        model = set_parameters(model, weights)
        mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                  n_eval_episodes=10,
                                                 )

        loss = 1  # temp
        other = {"reward": mean_reward}
        # print("***DEBUG", "mean_reward", mean_reward)
        global rounds
        summary.add_scalar("result_centralized", mean_reward, rounds)
        rounds += 1

        return [loss, other]


    def on_fit_config_fn(i) :
        print(i)
        return { "round" : i }

    clients = 2

    class SaveModelStrategy(fl.server.strategy.FedAvg):
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

                eval_env = gym.make("LunarLanderContinuous-v2")
                model = SAC("MlpPolicy", eval_env, verbose=0)
                aggregated_parameter = parameters_to_weights(aggregated_weights)
                model = set_parameters(model, aggregated_parameter)
                model.save(f"./logs-model/model_{rnd}.zip")
                
            return aggregated_weights, aggregated_res[1]

    strategy = SaveModelStrategy(
        # FIXME:
        min_fit_clients=clients,  # Minimum number of clients to be sampled for the next round
        min_available_clients=clients,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients=clients,
        eval_fn=eval_fn,
        on_fit_config_fn=on_fit_config_fn,
    )


    fl.server.start_server(server_address="[::]:13080", config={"num_rounds": NUM_ROUNDS}, strategy=strategy)

if __name__ == "__main__":
    # now = datetime.datetime.now()
    # cur_time = now.strftime("%H:%M:%S")
    if len(sys.argv)-1 != 1 :
        print("usage : flserver.py [log folder name] ")
        exit()
    log_path = "./{}/server_result".format(sys.argv[1])
    print("log path : " ,log_path)
    summary = SummaryWriter(log_path)
    main()
    summary.close()

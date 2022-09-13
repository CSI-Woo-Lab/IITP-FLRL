import flwr as fl
from flwr.common.parameter import parameters_to_weights
import torch
from collections import OrderedDict
from typing import List, Tuple, Optional

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# import navigation_2d
import envs

# 참조자료
# https://flower.dev/docs/strategies.html
# https://flower.dev/docs/apiref-flwr.html#server-strategy-strategy

# eval_fn():
# https://github.com/adap/flower/blob/3db5d103b03d4fb4e1ee1579bac31e385c864d0c/src/py/flwr/server/strategy/fedavg.py#L156


# FIXME:
NUM_ROUNDS = 40
NUM_ADAPT_TIMESTEP = 3  # 3, 25, 50


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


def main():
    # FIXME:

    def eval_fn(weights):
        total_mean_reward = 0
        total_adapt_reward = 0
        for env_id in range(4):
            # FIXME:
            eval_env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-No-Obstacle-v0")

            model = SAC("MlpPolicy", eval_env, verbose=0)
            model = set_parameters(model, weights)
            mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                    n_eval_episodes=25,  # FIXME
                                                    )
            total_mean_reward += mean_reward
            
            # adaptation phase
            model.learn(NUM_ADAPT_TIMESTEP)
            a_rwd, std = evaluate_policy(model, eval_env,
                                       n_eval_episodes=25,  # FIXME
                                      )
            total_adapt_reward += a_rwd

        total_mean_reward /= 4
        total_adapt_reward /= 4

        loss = 1  # temp
        # TODO: more safe logger...
        other = {"mean_reward": total_mean_reward, "adapt_reward": total_adapt_reward}
        # print("***DEBUG", "mean_reward", mean_reward)
        return [loss, other]


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

                eval_env = gym.make("Navi-Acc-Lidar-Obs-Task0_easy-v0")
                model = SAC("MlpPolicy", eval_env, verbose=0)
                aggregated_parameter = parameters_to_weights(aggregated_weights)
                model = set_parameters(model, aggregated_parameter)
                model.save(f"./logs-model/model_{rnd}.zip")
                
            return aggregated_weights, aggregated_res[1]


    strategy = SaveModelStrategy(
        # FIXME:
        min_fit_clients=4,  # Minimum number of clients to be sampled for the next round
        min_available_clients=4,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients=4,
        eval_fn=eval_fn,
    )

    fl.server.start_server(config={"num_rounds": NUM_ROUNDS}, strategy=strategy, server_address="[::]:8081")


if __name__ == "__main__":
    main()

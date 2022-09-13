from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device

import sys
import datetime
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import navigation_2d

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print( "Device = ", DEVICE)


# FIXME:
TIME_STEPS = 20000


def main(log_path, env_id):
    """Create model, Create env, define Flower client, start Flower client."""
    
    # FIXME: 0 1 2 3, non-iid
    env = gym.make(f"Navi-Acc-Lidar-Obs-Task{env_id}_easy-v0")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    
    tmp = 100 # RL에서 사용하지 않는 num_examples를 대신하기위한 dummy value

    # Flower client
    class PPOClient(fl.client.NumPyClient):
        def get_parameters(self):
            params = model.get_parameters()
            return [val.cpu().numpy() for _, val in params['policy'].items()]

        def set_parameters(self, parameters):
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

        def fit(self, parameters, config):
            print("=============[fitting start]================") # 각 round를 구분하기위한 출력
            self.set_parameters(parameters)
            train(model, TIME_STEPS)
            return self.get_parameters(), tmp , {'round' : 1}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, reward = test(model, env)
            return float(loss), tmp , {"Reward": float(reward)}

    # FIXME: Start client ( flserver를 돌리는 주소로 변경 후 사용 )
    fl.client.start_numpy_client("[::]:8081", client=PPOClient())


def train(model, time_steps):
    model.learn(total_timesteps=time_steps, log_interval=4, reset_num_timesteps=False)


def test(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return 1, mean_reward


if __name__ == "__main__":
    if len(sys.argv)-1 != 2 :
        print("usage : flclient.py [log folder name] [client id (env id)]")
        print("client id - integer for identify client log")
        exit()
    log_path = "./{}/client_{}".format(sys.argv[1], sys.argv[2])
    print("log path : " ,log_path)
    main( log_path, sys.argv[2])
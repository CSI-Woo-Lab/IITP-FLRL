from collections import OrderedDict

import flwr as fl
import torch
from torch.cuda import init
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.tensorboard import SummaryWriter

import sys
import datetime
import numpy as np

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gym_LLCC


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print( "Device = ", DEVICE)

time_steps = 20000

def main(log_path):
    """Create model, Create env, define Flower client, start Flower client."""

    env = gym.make('LLCC_wind_right-v0')
    model_init = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    tmp = 100 # RL에서 사용하지 않는 num_examples를 대신하기위한 dummy value

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        model = model_init
        def get_parameters(self):
            print("> get params")
            params = self.model.get_parameters()
            return [val.cpu().numpy() for _, val in params['policy'].items()]

        def set_parameters(self, parameters):
            print("> set params")
            params = self.model.get_parameters()
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
            update_dict = self.model.get_parameters()
            update_dict['policy'] = state_dict

            # parameters update
            self.model.set_parameters(update_dict)
            

        def fit(self, parameters, config):
            print(config)
            print("=============[fitting start]================") # 각 round를 구분하기위한 출력
            self.set_parameters(parameters)
            train(self.model, time_steps, config["round"])
            
            # reset ep_info_buffer for logging each round's mean reward
            self.model.ep_info_buffer = None

            return self.get_parameters(), tmp , config

        def evaluate(self, parameters, config):
            print("=============[eval start]================") # 각 round를 구분하기위한 출력
            self.set_parameters(parameters)
            loss, reward = test(self.model, env)
            return float(loss), tmp , {"Reward": float(reward)}

    # Start client ( flserver를 돌리는 주소로 변경 후 사용 )
    fl.client.start_numpy_client("[::]:13080", client=CifarClient())


def train(model, time_steps, round):
    round_str = "round_"+str(round)
    return model.learn(total_timesteps=time_steps, log_interval=1, reset_num_timesteps=False)
    
def test(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10,)

    loss = 1  # temp
    return 1, mean_reward


if __name__ == "__main__":
    if len(sys.argv)-1 != 2 :
        print("usage : flclient.py [log folder name] [client id]")
        print("client id - integer for identify client log")
        exit()
    log_path = "./{}/client_{}".format(sys.argv[1], sys.argv[2])
    print("log path : " ,log_path)
    main( log_path)
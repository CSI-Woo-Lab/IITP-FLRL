import argparse
import torch
import gym, navigation_2d
import d3rlpy
from d3rlpy.algos import BCQ, CQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.dataset import MDPDataset


# FIXME
N_EPOCHS = 100

use_gpu = torch.cuda.is_available()
d3rlpy.seed(2022)

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
parser.add_argument("--num_traj", type=int, default=argparse.SUPPRESS)
parser.add_argument(
    "--algo", type=str, choices=["bcq", "cql"], default=argparse.SUPPRESS
)
args = parser.parse_args()

dataset = MDPDataset.load(
    f"buffers/mdp-dataset-Navi-Acc-Lidar-Obs-Task0_easy-v0-ntraj-{args.num_traj}.h5"
)

env_name = f"Navi-Acc-Lidar-Obs-Task{args.env_id}_easy-v0"
env = gym.make(env_name)

if args.algo == "cql":
    model = CQL(use_gpu=use_gpu)
elif args.algo == "bcq":
    model = BCQ(use_gpu=use_gpu)
model.build_with_dataset(dataset)

# set environment in scorer function
evaluate_scorer = evaluate_on_environment(env)
log_name = f"{args.algo}_{args.num_traj}_traj_{N_EPOCHS}_epochs"
model.fit(
    dataset,
    eval_episodes=dataset,
    n_epochs=N_EPOCHS,
    experiment_name=log_name,
    tensorboard_dir=f"tensorboard/{log_name}",
    scorers={
        "environment": evaluate_scorer,
    },
)

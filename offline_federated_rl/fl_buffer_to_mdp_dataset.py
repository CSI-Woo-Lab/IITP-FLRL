import pickle
import argparse
import numpy as np
from d3rlpy.datasets import MDPDataset
from d3rlpy.wrappers.sb3 import to_mdp_dataset

# python fl_buffer_to_mdp.py --env_id {env_id} --num_trajectories {num_trajectoreis} --num_clinets {num_clients}

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
parser.add_argument("--num_trajectories", type=int, default=argparse.SUPPRESS)
parser.add_argument("--num_clients", type=int, default=argparse.SUPPRESS)
parser.add_argument("--dataset_name", type=int, default=argparse.SUPPRESS)
args = parser.parse_args()

def main(seed):
    file = open(
        f"buffers_fl/replay-buffer-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0-ntraj-{args.num_trajectories}_{count}.pkl",
        "rb",
    )
    replay_buffer = pickle.load(file)
    file.close()

    mdp_dataset = to_mdp_dataset(replay_buffer)
    print(len(mdp_dataset.observations))

    mdp_dataset.dump(
        f"buffers_fl/mdp-dataset-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0-ntraj-{args.num_trajectories}_{count}.h5"
    )

if __name__ == "__main__":
    for i in range(args.num_clients):
        dataset_name = args.dataset_name
        count = dataset_name + i
        main(505)

import pickle
import numpy as np
from d3rlpy.datasets import MDPDataset
from d3rlpy.wrappers.sb3 import to_mdp_dataset


ENV_ID = 0
NUM_TRAJ = 10


def main(seed):
    file = open(
        f"buffers_fl/replay-buffer-Navi-Acc-Lidar-Obs-Task{ENV_ID}_easy-v0-ntraj-{NUM_TRAJ}_{seed}.pkl",
        "rb",
    )
    replay_buffer = pickle.load(file)
    file.close()

    mdp_dataset = to_mdp_dataset(replay_buffer)
    print(len(mdp_dataset.observations))

    mdp_dataset.dump(
        f"buffers_fl/mdp-dataset-Navi-Acc-Lidar-Obs-Task{ENV_ID}_easy-v0-ntraj-{NUM_TRAJ}_{seed}.h5"
    )


if __name__ == "__main__":
    for i in range(4):
        seed = 2022 + i
        main(seed)

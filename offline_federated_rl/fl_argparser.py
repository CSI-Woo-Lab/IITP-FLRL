import argparse

parser = argparse.ArgumentParser(description="Argument for FRL")

parser.add_argument("--num_trajectories", type=int, default=argparse.SUPPRESS)
parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
parser.add_argument("--env_id", type=int, default=0)
parser.add_argument("--fl_num_round", type=int, default=300)
parser.add_argument("--client_algo", type=str, default="cql")
parser.add_argument("--fl_num_client", type=int, default=4)
parser.add_argument("--num_local_epoch", type=int, default=2)
parser.add_argument("--dataset_path", type=str, default=argparse.SUPPRESS)
parser.add_argument("--dataset_name", type=str)

args=parser.parse_args()

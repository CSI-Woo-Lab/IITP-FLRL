import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", "-o", default=None, type=str)
args = parser.parse_args()
if not args.optimizer:
    print("set optimizer")
    exit()

for i in range(4):
    os.system(f"python flclient.py log-client-{args.optimizer} {i} 8001 &")

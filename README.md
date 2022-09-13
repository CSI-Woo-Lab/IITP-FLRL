# Offline Federated Reinforcement Learning Simulator in 2D-Navigation

### This project was supported in part by the Institute for Information and Communications Technology Planning and Evaluation (IITP) under Grant 2021-0-00900

## Introduction
#### (i) Executes offline federated reinforcement learning in a 2D-Navigation environment with two heterogeneous tasks (task classification according to destination location), and (ii) measures the success rate of destination arrival of object navigation according to the global policy created by federated learning.
* #### Environments: Navigation-2D (Components: a start-point, a end-point, a navigator, moving obstacles)
* #### Learning Methods: Federated Learning (Server) + Offline Reinforcement Learning (Client)
* #### Evaluation Metric: Arrival Success Rate - Goal Score: 95% ↑
* #### Settings: 1 Server / 4 Clients (based on d3rl and flower frameworks)

## System Architecture 
![navigation_2d](.jpg)

## Install

### 1. Docker
``` bash
docker pull mkris0714/flrl:base
```

### 2. Git

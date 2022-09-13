# Offline Federated Reinforcement Learning Simulator in 2D-Navigation

### This project was supported in part by the Institute for Information and Communications Technology Planning and Evaluation (IITP) under Grant 2021-0-00900

## Introduction
#### (i) Executes offline federated reinforcement learning in a 2D-Navigation environment with two heterogeneous tasks (task classification according to destination location), and (ii) measures the success rate of destination arrival of object navigation according to the global policy created by federated learning.
* #### Environments: Navigation-2D (Components: a start-point, a end-point, a navigator, moving obstacles)
* #### Learning Methods: Federated Learning (Server) + Offline Reinforcement Learning (Client)
* #### Evaluation Metric: Arrival Success Rate - Goal Score: 95% ↑
* #### Settings: 1 Server, 4 Clients (based on d3rl and flower frameworks)

## System Architecture 
![architecture](/asset/architecture.png)

## Environments
Red: Navigator, Blue: Obstacles, Green: Destination
![navigation_2d](/asset/navigation_2d.gif)

## Install

### 1. Docker
``` bash
docker pull mkris0714/flrl:base
docker run --gpus all -e LC_ALL=C.UTF-8 -p 8080:8080 -it mkris0714/flrl:base /bin/bash
```

### 2. Git
#### pip Requirements (pip dependency resolving)
``` bash
pip install -U pip==20.3
pip install -r requirements.txt --use-deprecated=legacy-resolver
```
#### apt Requirements
``` bash
apt install freeglut3-dev
```

#### Environments Install (based on python3.8)
``` bash
cp -r navigation_2d/ /opt/conda/lib/python3.8/site-packages/
```

#### GPU Error Resolving (based on NVIDIA GTX3090, CUDNN11)
``` bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```



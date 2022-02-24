# offline-rl-2dnavi
## Usage

1. train online `train_online.py`
---
2. gather buffer `gather_buffer.py`
3. convert buffer to mdp dataset `buffer_to_mdp_dataset.py`
4. train offline `train_offline.py`
---
5. gather fl buffer `fl_gather_buffer.py`
6. convert fl buffer to mdp dataet `fl_buffer_to_mdp_dataset.py`
7. run flserver and flclient `flserver.py`, `flclient.py`

## Requirements

`python3.7.11`

```
torch==1.10.0
```

```
d3rlpy==0.91
flwr==0.17.0
gym==0.18.0
navigation-2d==1.4.0
stable-baselines3==1.3.0
```

* [navigation-2d github repo](https://github.com/mjyoo2/navigation_2d)
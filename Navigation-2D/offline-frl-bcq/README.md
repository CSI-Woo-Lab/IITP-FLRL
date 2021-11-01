# offline-frl-bcq

## From
[continuous_BCQ](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ)

## Usage
```
python main.py --train_behavioral --gaussian_std 0.1 --env Navi-Acc-Lidar-Obs-Task<num>_easy-v0
```

```
python main.py --generate_buffer --max_timesteps 100000 --env Navi-Acc-Lidar-Obs-Task<num>_easy-v0
```

```
python flserver.py
python flclient.py --client_name <name> --env_id <num>
```

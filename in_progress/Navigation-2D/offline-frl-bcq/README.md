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

### Train BCQ
```
python main.py --env Navi-Acc-Lidar-Obs-Task<num>_easy-v0
```

### Train CQL
```
python main.py --env Navi-Acc-Lidar-Obs-Task<num>_easy-v0 --cql
```


### Offline FRL
Use `homogen` or `heterogen`.
```
python flserver.py --log_name heterogen --client_name <name> 2>&1 | tee log-heterogen-<name>.txt
python flclient.py --client_name <name> --log_name heterogen --env_id <num>
```

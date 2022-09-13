from .navigation_env import NavigationEnvDefault, NavigationEnvAcc, NavigationEnvAccLidarObs, NonStationaryNavigation
from gym.envs import register
from .config import *

mode = ['easy', 'normal', 'hard', 'very_hard']

custom_envs = {}
for idx, obs_conf in enumerate(config_set):
    custom_envs['Navi-Vel-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[idx//8])] = dict(
                 path='navigation_2d:NavigationEnvDefault',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[idx//8])] = dict(
                 path='navigation_2d:NavigationEnvAcc',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Lidar-Obs-Task{}_{}-v0'.format(idx%8, mode[idx//8])] = dict(
                 path='navigation_2d:NavigationEnvAccLidarObs',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))

for i in range(5):
    for j in range(5):
        custom_envs['Non-Stationary-Navigation_dyn_{}_unc_{}-v0'.format(i, j)] = dict(
                     path='navigation_2d:NonStationaryNavigation',
                     max_episode_steps=200,
                     kwargs=dict(task_args=non_sta_config_set[5 * i + j]))

# register each env into
def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value['path'],
                        max_episode_steps=value['max_episode_steps'],
                        kwargs=value['kwargs'])
        if 'reward_threshold' in value.keys():
            arg_dict['reward_threshold'] = value['reward_threshold']
        register(**arg_dict)

register_custom_envs()

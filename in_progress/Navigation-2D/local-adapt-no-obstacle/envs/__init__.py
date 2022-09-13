from gym.envs import register
from navigation_2d.config import *


mode = ['easy', 'normal', 'hard', 'very_hard']

custom_envs = {}
for idx, obs_conf in enumerate(config_set):
    custom_envs['Navi-Acc-Lidar-Obs-Task{}_{}-No-Obstacle-v0'.format(idx%8, mode[idx//8])] = dict(
                 path='envs.no_obs_navi_env:NoObstacleNavigationEnvAccLidarObs',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))

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

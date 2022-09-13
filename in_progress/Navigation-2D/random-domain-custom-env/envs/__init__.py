from gym.envs import register
from navigation_2d import config


task_dict = {}
task_dict['OBSTACLE_POSITIONS'] = config.obs_set[0]
task_dict['Goal'] = config.goal_set[0]
kwargs = dict(task_args=task_dict)

register(
    id="Navi-Acc-Lidar-Obs-RandomDomain-easy-v0",
    entry_point="envs.random_domain_navi_2d_env:RandomDomainNavigationEnvAccLidarObs",
    max_episode_steps=200,
    kwargs=kwargs,
)

import random
from navigation_2d import NavigationEnvAccLidarObs, config


class RandomDomainNavigationEnvAccLidarObs(NavigationEnvAccLidarObs):
    def __init__(self, task_args, max_obs_range=3, max_speed=2, initial_speed=1, goal_reset_period=1, **kwargs):
        super().__init__(task_args, max_obs_range=max_obs_range, max_speed=max_speed, initial_speed=initial_speed, **kwargs)
        self.goal_reset_period = goal_reset_period
        self.cur_num_reset = 0
        self.cur_env_num = 0

    def reset(self):
        self.cur_num_reset += 1
        if self.cur_num_reset >= self.goal_reset_period:
            env_num = self.cur_env_num % 8
            self.task_args["Goal"] = config.goal_set[env_num]
            self.cur_num_reset = 0
        self.cur_env_num += 1
        return super().reset()
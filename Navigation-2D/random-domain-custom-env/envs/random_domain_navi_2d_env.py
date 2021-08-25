import random
from navigation_2d import NavigationEnvAccLidarObs, config


class RandomDomainNavigationEnvAccLidarObs(NavigationEnvAccLidarObs):
    def __init__(self, task_args, max_obs_range=3, max_speed=2, initial_speed=1, **kwargs):
        super().__init__(task_args, max_obs_range=max_obs_range, max_speed=max_speed, initial_speed=initial_speed, **kwargs)

    def reset(self):
        random_num = random.randint(0, 7)
        self.task_args["Goal"] = config.goal_set[random_num]
        return super().reset()

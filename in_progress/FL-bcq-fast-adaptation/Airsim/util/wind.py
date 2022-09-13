import numpy as np

from util.util import random, angle_add


class Wind(object):
    def __init__(self, max_speed, dynamics, smooth):
        self.max_speed = max_speed
        self.smooth = smooth
        self.dynamics = dynamics
        self.direction = np.array([np.random.random() * np.pi, np.random.random() * 2 * np.pi])
        self.tendency_speed = 0
        self.tendency_direction = np.zeros(2, )
        self.temp_speed = np.random.random() * max_speed
        return

    def change_wind(self):
        change_speed = random((1, ))[0] * self.max_speed * self.dynamics * (1 - self.smooth) + self.tendency_speed * self.smooth
        self.tendency_speed = change_speed
        self.temp_speed = np.clip(change_speed + self.temp_speed, a_max=self.max_speed, a_min=0)
        change_direction = random((2, )) * self.dynamics * np.pi * (1 - self.smooth) + self.tendency_direction * self.smooth
        self.tendency_direction = change_direction
        self.direction[0] = angle_add(np.pi, self.direction[0], change_direction[0], mode='theta')
        self.direction[1] = angle_add(2 * np.pi, self.direction[1], change_direction[1], mode='phi')
        return

    def get_wind(self):
        return self.wind

    def step(self):
        self.change_wind()
        return self.wind

    @property
    def wind(self):
        x = self.temp_speed * np.sin(self.direction[0]) * np.cos(self.direction[1])
        y = self.temp_speed * np.sin(self.direction[0]) * np.sin(self.direction[1])
        z = self.temp_speed * np.cos(self.direction[1])
        return np.array([x, y, z])
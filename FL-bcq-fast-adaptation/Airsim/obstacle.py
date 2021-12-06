from util.spot import spot_list
from numpy.linalg import norm

import numpy as np

dyn_list = [4.0, 2.0, 1.0, 0.5, 0.25]

class Obstacle_set(object):
    def __init__(self, num_of_object, dyn):
        self.num_of_object = num_of_object
        self.obstacles = []
        self.dyn = dyn_list[dyn]
        self.init_obstacle()

    def init_obstacle(self):
        for i in range(self.num_of_object):
            self.obstacles.append(Obstacle(self.dyn))

    def step(self, position):
        state = []
        violation = False
        for obstacle in self.obstacles:
            state.append(obstacle.step())
            violation = violation or obstacle.violation(position)
        return np.array(state), violation

    def reset(self):
        state = []
        for obstacle in self.obstacles:
            state.append(obstacle.reset())
        return np.array(state)

    def position(self):
        state = []
        for obstacle in self.obstacles:
            state.append(obstacle.position)
        return np.array(state)


class Obstacle(object):
    def __init__(self, dyn):
        self.position = np.array([self.set_pos(), self.set_pos(), -15])
        self.goal = np.concatenate([np.array(spot_list[np.random.randint(43)]), np.array([-15])])
        self.dyn = dyn

    def step(self):
        move_direction = self.goal - self.position
        self.position += move_direction / (norm(move_direction) * self.dyn)
        self.success()
        return self.position

    def success(self):
        distance = norm(self.position - self.goal)
        if distance < 10:
            self.goal = np.concatenate([np.array(spot_list[np.random.randint(43)]), np.array([-15])])

    def violation(self, drone_position):
        if norm(self.position - drone_position) < 5:
            return True
        else:
            return False

    def reset(self):
        self.position = np.array([self.set_pos(), self.set_pos(), -15])
        return self.position

    def set_pos(self):
        if np.random.random() > 0.5:
            val = 20 + abs(np.random.random() * 180)
        else:
            val = -20 - abs(np.random.random() * 180)
        return val

if __name__ == '__main__':
    obstacle = Obstacle()
    for i in range(100):
        print(obstacle.step())
import numpy as np
import copy
from .util import normalize_position
from .config import W, H
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)

class Obstacles(object):
    def __init__(self, world, max_speed, args):
        self.args = args
        self.world = world
        self.speed_table = np.zeros(len(self.args['OBSTACLE_POSITIONS']))
        self.obstacles = []
        self.max_speed = max_speed
        self.min_speed = 0.5

    def build_obstacles(self):
        for i in range(len(self.args['OBSTACLE_POSITIONS'])):
            obstacle = Obstacle(self.world, self.min_speed, self.max_speed, self.args['OBSTACLE_POSITIONS'][i][:2],
                                mode=self.args['OBSTACLE_POSITIONS'][i][2])
            obstacle.build_obstacle()
            self.speed_table[i] = obstacle.speed / 5
            self.obstacles.append(obstacle)

    def clean_obstacles(self):
        while self.obstacles:
            obstacle = self.obstacles.pop(0)
            self.world.DestroyBody(obstacle.dynamic_body)

    def step(self):
        for i, obs in enumerate(self.obstacles):
            obs.step()
            self.speed_table[i] = obs.speed / 5

    def reset(self):
        self.speed_table = np.zeros(len(self.args['OBSTACLE_POSITIONS']))


    def positions(self, drone_position):
        position = normalize_position(drone_position, W, H)
        obstacle_position = [position - normalize_position(o.position, W, H) for o in self.obstacles]
        return obstacle_position

    @property
    def speeds(self):
        return np.copy(self.speed_table)

    @property
    def dynamic_bodies(self):
        ret_list = []
        for obs in self.obstacles:
            ret_list.append(obs.dynamic_body)
        return ret_list


class Obstacle(object):
    def __init__(self, world, min_speed, max_speed, move_range, mode=0):
        self.mode = mode
        self.world = world
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.move_range = move_range
        self.direction = 0
        self.distance = np.linalg.norm(np.array(move_range[0]) - np.array(move_range[1]))
        self.dynamic_body = None

    def build_obstacle(self):
        pos = copy.deepcopy(self.move_range[0])
        if self.mode == 2:
            theta = np.random.random() * np.pi * 2
            pos = np.array([np.cos(theta), np.sin(theta)]) * self.distance + np.array(self.move_range[1])
        else:
            pos[self.mode] = np.random.uniform(low=self.move_range[0][self.mode], high=self.move_range[1][self.mode])
        self.dynamic_body = self.world.CreateDynamicBody(position=(pos[0], pos[1]), angle=0.0,
                                                         fixtures=fixtureDef(
                                                             shape=circleShape(radius=0.3, pos=(0, 0)),
                                                             density=5.0, friction=0, categoryBits=0x001,
                                                             maskBits=0x0010, restitution=1.0))
        self.dynamic_body.color1 = (0.7, 0.2, 0.2)
        self.dynamic_body.color2 = (0.7, 0.2, 0.2)
        self.step(init=True)

    def step(self, init=False):
        if self.mode == 2:
            self.set_circular_vel(init)
        else:
            self.set_linear_vel(init)

    def set_linear_vel(self, init=False):
        self.direction = self.check_range()
        if init:
            self.direction = np.random.choice([-1, 1])
        if self.direction == 0:
            return
        speed = np.random.uniform(low=self.min_speed, high=self.max_speed)
        mode_direction = np.array([0, 0])
        mode_direction[self.mode] = 1
        next_velocity = (speed * self.direction) * mode_direction
        self.dynamic_body.linearVelocity.Set(next_velocity[0], next_velocity[1])

    def set_circular_vel(self, init=False):
        tuning_vec = np.array([self.move_range[1][0] - self.position[0], self.move_range[1][1] - self.position[1]]) * 1.5
        moving_vec = np.array([(self.move_range[1][1] - self.position[1])/5, -(self.move_range[1][0] - self.position[0])/5]) * 1.5

        if self.distance < np.linalg.norm(tuning_vec):
            self.dynamic_body.linearVelocity.Set(moving_vec[0]+tuning_vec[0]/10, moving_vec[1]+tuning_vec[1]/10)
        elif self.distance < np.linalg.norm(tuning_vec):
            self.dynamic_body.linearVelocity.Set(moving_vec[0]-tuning_vec[0]/10, moving_vec[1]-tuning_vec[1]/10)
        else:
            self.dynamic_body.linearVelocity.Set(moving_vec[0], moving_vec[1])

    def check_range(self):
        if self.dynamic_body.position[self.mode] >= self.move_range[1][self.mode]:
            return -1
        elif self.dynamic_body.position[self.mode] <= self.move_range[0][self.mode]:
            return 1
        else:
            return 0

    @property
    def speed(self):
        return np.linalg.norm(self.dynamic_body.linearVelocity) * self.direction

    @property
    def position(self):
        return self.dynamic_body.position
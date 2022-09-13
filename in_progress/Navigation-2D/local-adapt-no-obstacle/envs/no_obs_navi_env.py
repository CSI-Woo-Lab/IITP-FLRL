from navigation_2d import NavigationEnvAccLidarObs
import numpy as np

from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)
from collections import deque

from navigation_2d.navigation_env import LidarCallback, ContactDetector
from navigation_2d.config import *


class NoObstacleNavigationEnvAccLidarObs(NavigationEnvAccLidarObs):
    def __init__(self, task_args, max_obs_range=3, max_speed=2, initial_speed=1, **kwargs):
        super().__init__(task_args, max_obs_range=max_obs_range, max_speed=max_speed, initial_speed=initial_speed, **kwargs)

    def reset(self):
        if self.scores is None:
            self.scores = deque(maxlen=10000)
        else:
            if self.achieve_goal:
                self.scores.append(1)
            else:
                self.scores.append(0)


        self.game_over = False
        self.prev_shaping = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        # timer
        self.energy = 0.2
        # clean up objects in the Box 2D world
        self._destroy()
        # create lidar objects
        self.lidar = [LidarCallback() for _ in range(self.num_beams)]
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # create new world
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        p1 = (1, 1)
        p2 = (W - 1, 1)
        self.moon.CreateEdgeFixture(vertices=[p1, p2], density=100, friction=0, restitution=1.0)
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        self._build_wall()

        drone_pos = self._build_drone()
        # create goal
        np.random.seed(np.random.randint(low=0, high=100000))
        self._build_goal()
        self._build_obs_range()

        self.drawlist = [self.obs_range_plt, self.drone, self.goal] + self.walls + self.obstacles.dynamic_bodies
        self._observe_lidar(drone_pos)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        return np.copy(self.array_observation())

    def step(self, action: np.iterable):
        action = np.asarray(action, dtype=np.float64)
        before_pos = np.asarray(self.drone.position).copy()
        self.timesteps += 1
        self.action = action
        # mass == 1 impulse == action
        self.drone.ApplyForce((action[0], action[1]), self.drone.position, wake=True)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        v = self.drone.linearVelocity
        v = np.clip(v, a_min=-1, a_max=1)
        self.drone.linearVelocity.Set(v[0], v[1]) # clip velocity

        self.energy -= 1e-3
        pos = np.array(self.drone.position)
        self._observe_lidar(pos)
        reward = self.prev_distance - self.distance
        self.prev_distance = self.distance
        done = False
        if self.collision_done:
            done = self.game_over

        info = {}
        if self.energy <= 0:
            done = True
        if done or self.achieve_goal:
            if self.achieve_goal:
                reward = 1
                done = True
            info['is_success'] = self.achieve_goal
            info['energy'] = self.energy
            info['episode'] = {'r': reward, 'l': (0.2 - self.energy) * 1000}

        obs = np.copy(self.array_observation())
        self.obs_queue.append(obs)
        return obs, reward, done, info

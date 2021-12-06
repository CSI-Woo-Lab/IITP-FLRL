import time
import numpy as np
from numpy.linalg import norm
from util.spot import spot_list
from util.util import vector_process

class EnvInfo(object):
    def __init__(self):
        self.episode_length = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.start_time = time.time()

    def __repr__(self):
        print("episode length: ", self.episode_length)
        print("episode reward: ", self.episode_reward)
        print("total timesteps: ", self.total_steps)
        print("FPS: ", self.fps)
        return " "

    def step(self, reward):
        self.episode_length += 1
        self.episode_reward += reward
        self.total_steps += 1

    def reset(self):
        self.episode_length = 0
        self.episode_reward = 0
        self.start_time = time.time()

    @property
    def fps(self):
        return self.episode_length / ((time.time() -self.start_time) + 1e-10)


class MultiRotorEnvInfo(object):
    def __init__(self):
        self.goal_reward = 0
        self.prev_distance = None
        self.last_action = None
        self.goal = None
        self.pos_info = None
        self.lidar_data = None

    def __repr__(self):
        print("success: ", self.goal_reward)
        return " "

    def step(self, pos_info, lidar_data, last_action):
        self.last_action = last_action
        self.pos_info = pos_info
        self.lidar_data = lidar_data
        reward = self.get_reward()
        self.prev_distance = self.goal_distance
        return reward

    def success(self):
        print ("목표 지점에 도달하였습니다.")
        goal = spot_list[np.random.randint(0, 43)]
        self.goal = np.array([goal[0], goal[1], -10])
        self.goal_reward += 1
        self.prev_distance = self.goal_distance
        return

    def reset(self, pos_info, lidar_data):
        goal = spot_list[np.random.randint(0, 43)]
        self.goal = np.array([goal[0], goal[1], -10])
        self.pos_info = pos_info
        self.prev_distance = self.goal_distance
        self.goal_reward = 0
        self.lidar_data = lidar_data
        self.last_action = np.zeros((3,))
        return

    def make_state(self):
        lidar_data = self.get_lidar_data()
        pos, vel, ori = self.get_sensor_data()
        goal_data = self.goal
        state = np.concatenate([goal_data, pos, vel, ori, lidar_data])
        return state

    @property
    def goal_distance(self):
        return np.array([self.pos_info.position.x_val - self.goal[0], self.pos_info.position.y_val - self.goal[1],
                  self.pos_info.position.z_val - self.goal[2]])

    def get_reward(self):
        return norm(self.prev_distance, 2) - norm(self.goal_distance, 2)

    def get_sensor_data(self):
        position = np.array([self.pos_info.position.x_val,
                             self.pos_info.position.y_val,
                             self.pos_info.position.z_val])
        velocity = np.array([self.pos_info.linear_velocity.x_val,
                             self.pos_info.linear_velocity.y_val,
                             self.pos_info.linear_velocity.z_val])
        orientation = np.array([self.pos_info.orientation.w_val,
                                self.pos_info.orientation.x_val,
                                self.pos_info.orientation.y_val,
                                self.pos_info.orientation.z_val])
        return position, velocity, orientation

    def get_lidar_data(self):
        result = np.ones(shape=(16, 8)) * 30
        points = self.lidar_data.point_cloud
        if len(points) == 1:
            return result
        points = np.array(points).reshape([-1, 3])
        for vector in points:
            distance, theta, phi = vector_process(vector, self.pos_info)
            y_axis = int(theta/(np.pi/8))
            if y_axis == 8:
                y_axis = 7
            x_axis = int(phi/(np.pi/8))
            result[x_axis][y_axis] = min(distance, result[x_axis][y_axis])
        return result.flatten()

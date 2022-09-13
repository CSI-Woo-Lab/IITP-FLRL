import gym
from airsim import *

import numpy as np
import time
from util import EnvInfo, MultiRotorEnvInfo
from numpy.linalg import norm

class MultiRotorEnv(gym.Env):
    def __init__(self, drone_id, speed, 
                # unc, dyn,
                verbose=1):
        self.drone_id = drone_id
        self.verbose = verbose
        if self.verbose:
            print('Drone ID: ', self.drone_id)
        self.client = MultirotorClient(ip="127.0.0.1")
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name = self.drone_id)
        self.client.armDisarm(True, vehicle_name = self.drone_id)
        self.client.takeoffAsync(vehicle_name = self.drone_id).join()

        self.multi_rotor_info = MultiRotorEnvInfo()
        self.info = EnvInfo()
        # self.wind = np.zeros(2)
        self.substep = 0
        self.timesteps = 0
        # self.uncertainty = unc
        # self.dynamics = dyn
        self.stationary = False

        self.speed = speed
        self.temp_time = time.time()
        self.time_per_step = 0.1 / self.speed
        self.theta = np.random.random() * np.pi * 2
        # self.state_stack = StateStack(shape=(self.sensor_shape[0] + 2, self.sensor_shape[1]), max_len=3)
        self.action_space = gym.spaces.Box(low=np.full((3,), -10), high=np.full((3,), 10), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(141, -1), high=np.full(141, -1), dtype=np.float32)
        print("action space:", self.action_space.shape)
        print("obs space:", self.observation_space.shape)

    def step(self, action : np.ndarray):
        already_col = False
        done = False
        self.timesteps += 1
        while time.time() - self.temp_time < self.time_per_step:
            time.sleep(min(max(self.time_per_step - (time.time() - self.temp_time), 0), 0.01))
            col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
            if col_info.has_collided:
                already_col = True
                break
        self.temp_time = time.time()

        self.client.moveByVelocityAsync(float(action[0]), float(action[1]), float(action[2]), 1, vehicle_name=self.drone_id)
        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)
        lidar_data = self.client.getLidarData(lidar_name='lidar', vehicle_name=self.drone_id)
        # self.wind_step()
        reward = self.multi_rotor_info.step(pos_info, lidar_data, action)
        self.info.step(reward)

        info = {'episode': None, 'success': False}

        if norm(self.multi_rotor_info.goal_distance) < 5:
            self.multi_rotor_info.success()
            info['success'] = True

        if pos_info.position.z_val > -1 or pos_info.position.z_val < -20 or abs(pos_info.position.y_val) > 200 or \
                abs(pos_info.position.x_val) > 200 or col_info.has_collided or already_col or self.substep > 1000:
            done = True

        state = self.multi_rotor_info.make_state()
        # print(state)

        return state, reward, done, info

    def reset(self):
        if self.verbose:
            print(self.info)
            print(self.multi_rotor_info)

        self.timesteps = 0
        self.substep = 0
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.drone_id)
        self.client.armDisarm(True, vehicle_name=self.drone_id)
        self.client.takeoffAsync(vehicle_name=self.drone_id).join()
        self.client.moveToPositionAsync(0, 0, -15, 10).join()

        # self.wind = np.zeros(2)
        time.sleep(3)

        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name = self.drone_id)
        lidar_data = self.client.getLidarData(lidar_name='lidar', vehicle_name=self.drone_id)

        self.multi_rotor_info.reset(pos_info, lidar_data)

        state = self.multi_rotor_info.make_state()
        # self.state_stack.add(state)

        self.temp_time = time.time()
        self.info.reset()
        return state

    # def wind_step(self):
    #     if np.random.random() > self.dynamics * 0.01 and not self.stationary:
    #         self.wind = np.array([np.cos(self.theta), np.sin(self.theta)]) * np.random.random() * 20
    #         if np.random.random() > self.uncertainty * 0.25:
    #             self.theta += np.pi / 180 * (np.random.normal(0, 1) * 10)
    #         else:
    #             self.theta = np.random.random() * np.pi * 2
    #     wind = Vector3r(self.wind[0], self.wind[1], 0)
    #     self.client.simSetWind(wind)
    #     return

    # def set_stationary(self, wind_dir):
    #     self.wind = np.array([np.cos(wind_dir), np.sin(wind_dir)]) * np.random.random() * 20
    #     self.stationary = True

    def render(self, mode='human'):
        pass

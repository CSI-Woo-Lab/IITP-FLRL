import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)
# from Integrated_policy_learning.network import Network
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from collections import deque
from .config import *
from .util import *
from .objects import Obstacles

def to_rect(obstacle_pos):
    axis = obstacle_pos[2]
    if axis == HORIZONTAL:
        y_range = 0.6
        x_range = np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6
        position = [(obstacle_pos[0][0] + obstacle_pos[1][0])/2, (obstacle_pos[0][1] + obstacle_pos[1][1])/2]
        poly = rotation_4([x_range/2, y_range/2])
    else:
        y_range = np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6
        x_range = 0.6
        position = [(obstacle_pos[0][0] + obstacle_pos[1][0])/2, (obstacle_pos[0][1] + obstacle_pos[1][1])/2]
        poly = rotation_4([x_range/2, y_range/2])
    return position, poly


class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.drone == contact.fixtureA.body or self.env.drone == contact.fixtureB.body:
            # if the drone is collide to something, set game over true
            self.env.game_over = True
            # if the drone collide with the goal, success
            if self.env.goal == contact.fixtureA.body or self.env.goal == contact.fixtureB.body:
                self.env.achieve_goal = True

def MapGoal(goalpos):
    goal_0 = (denormalize_position([0.2, 0.2], W, H))
    goal_1 = (denormalize_position([0.2, 0.8], W, H))
    goal_2 = (denormalize_position([0.8, 0.8], W, H))
    goal_3 = (denormalize_position([0.8, 0.2], W, H))
    goal_4 = (denormalize_position([0.0756, 0.5], W, H))
    goal_5 = (denormalize_position([0.5, 0.0756], W, H))
    goal_6 = (denormalize_position([0.924, 0.5], W, H))
    goal_7 = (denormalize_position([0.5, 0.924], W, H))

    if (goalpos[0] == goal_0[0]) & (goalpos[1] == goal_0[1]):
        return [1,0,0,0,0,0,0,0]
    elif (goalpos[0] == goal_1[0]) & (goalpos[1] == goal_1[1]):
        return [0,1,0,0,0,0,0,0]
    elif (goalpos[0] == goal_2[0]) & (goalpos[1] == goal_2[1]):
        return [0,0,1,0,0,0,0,0]
    elif (goalpos[0] == goal_3[0]) & (goalpos[1] == goal_3[1]):
        return [0,0,0,1,0,0,0,0]
    elif (goalpos[0] == goal_4[0]) & (goalpos[1] == goal_4[1]):
        return [0,0,0,0,1,0,0,0]
    elif (goalpos[0] == goal_5[0]) & (goalpos[1] == goal_5[1]):
        return [0,0,0,0,0,1,0,0]
    elif (goalpos[0] == goal_6[0]) & (goalpos[1] == goal_6[1]):
        return [0,0,0,0,0,0,1,0]
    elif (goalpos[0] == goal_7[0]) & (goalpos[1] == goal_7[1]):
        return [0,0,0,0,0,0,0,1]
    else:
        print('Mapping Goal Error!')

def Goal_change(goalpos):
    # if goal == 0
    goal_0 = (denormalize_position([0.2, 0.2], W, H))
    goal_5 = (denormalize_position([0.5, 0.0756], W, H))
    if (goalpos[0] == goal_0[0]) & (goalpos[1] == goal_0[1]):
        return goal_5
    # if goal == 5
    elif (goalpos[0] == goal_5[0]) & (goalpos[1] == goal_5[1]):
        return goal_0
    else:
        print('goal error!')

class NavigationEnvDefault(gym.Env, EzPickle):
    def __init__(self, task_args, max_obs_range=3,  max_speed=2, initial_speed=2, **kwargs):
        super(EzPickle, self).__init__()
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': FPS
        }
        ID = MapGoal(task_args['Goal'])
        self.ID = ID
        
        '''
        dictionary representation of observation 
        it is useful handling dict space observation, 
        classifying local observation and global observation, 
        lazy evaluation of observation space; whenever we add or delete some observation information   
        '''
        self.observation_meta_data = {
            'position': gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
            'distance': gym.spaces.Box(np.array([0]), np.array([np.sqrt(W ** 2 + H ** 2)]), dtype=np.float32),
            'lidar': gym.spaces.Box(low=0, high=1, shape=(8,)),
            'energy': gym.spaces.Box(low=0, high=1, shape=(1,)),
            'obstacle_speed': gym.spaces.Box(low=-1, high=1, shape=(len(task_args['OBSTACLE_POSITIONS']),)),
            'obstacle_position': gym.spaces.Box(low=0, high=1, shape=(2 * len(task_args['OBSTACLE_POSITIONS']),)),
            'ID' : gym.spaces.Box(low=0, high=1, shape=(8,))
        }

        # meta data keys. It is used to force order to get observation.
        # We may use ordered dict, but saving key with list is more economic
        # rather than create ordered dict whenever steps proceed
        self.observation_meta_data_keys = ['position', 'distance', 'lidar', 'energy', 'obstacle_speed',
                                           'obstacle_position', 'ID']
        self._ezpickle_args = ( )
        self._ezpickle_kwargs = {}
        self.np_random = 7777
        self.verbose = False
        self.scores = None

        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, 0))
        self.moon = None
        self.drone = None
        self.walls = []
        self.goal = None
        self.obs_range_plt = None
        self.max_obs_range = max_obs_range
        self.num_beams = 8
        self.lidar = None
        self.drawlist = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.dynamics = initial_speed
        self.energy = 0.2
        self.random_init = False
        self.collision_done = True

        p1 = (0.75, 0.5)
        p2 = (W - 0.5, 0.5)
        self.sky_polys = [[p1, p2, (p2[0], H-0.5), (p1[0], H-0.5)]]
        self.position_intrinsic_reward = None
        self.obstacles = Obstacles(self.world, max_speed, task_args)
        self.task_args = task_args
        self.game_over = False
        self.prev_shaping = None
        self.flag = True
        self.count = 0


        # debug
        self.action = None
        self.obs_queue = deque(maxlen=10)
    
    def set_random_init(self, random_init: bool):
        self.random_init = random_init

    def set_collision_done(self, collision_done: bool):
        self.collision_done = collision_done

    @property
    def drone_start_pos(self):
        if self.random_init:
            return denormalize_position(np.random.random(2) * 0.8 + 0.1, W, H)
        else:
            return denormalize_position([0.5, 0.924], W ,H)

    @property
    def observation_space(self):
        size = 0
        for k in self.observation_meta_data:
            val = self.observation_meta_data[k]
            size += val.shape[0]

        return spaces.Box(-np.inf, np.inf, shape=(size, ), dtype=np.float32)

    @property
    def action_space(self):
        # Action is two floats [vertical speed, horizontal speed].
        return spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    
    @property
    def distance(self):
        return np.linalg.norm(self.goal.position - self.drone.position)

    def seed(self, seed=7777):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def to_polar(x):
        r = np.linalg.norm(x)
        x_pos = x[0]
        y_pos = x[1]
        theta = np.arctan(y_pos/x_pos)
        return np.asarray([r, theta])
    
    def dict_observation(self):
        position = normalize_position(self.drone.position, W, H)
        distance = np.linalg.norm(normalize_position(self.drone.position, W, H) - (normalize_position(self.goal.position, W, H)))
        lidar = [l.fraction for l in self.lidar]
        obstacle_speed = self.obstacles.speeds
        obstacle_position = self.obstacles.positions(self.drone.position)
        ID = self.ID
        
        dict_obs = {
            'position':position,
            'distance': distance,
            'lidar': lidar,
            'energy': self.energy,
            'obstacle_speed': obstacle_speed,
            'obstacle_position':obstacle_position,
            'ID' : ID,
        }

        return dict_obs
    
    def array_observation(self, dict_obs=None):
        if dict_obs is None:
            dict_obs = self.dict_observation()

        obs = []

        for k in self.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        
        return np.concatenate(obs)

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.drone)
        self.drone = None
        self._clean_walls()
        self.world.DestroyBody(self.goal)
        self.goal = None
        self.world.DestroyBody(self.obs_range_plt)
        self.obs_range_plt = None
        self.obstacles.clean_obstacles()

    def _observe_lidar(self, pos):
        for i in range(self.num_beams):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(i * 2 * np.pi / self.num_beams) * self.max_obs_range,
                pos[1] + math.cos(i * 2 * np.pi / self.num_beams) * self.max_obs_range)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    def _build_wall(self):
        wall_pos = WALL_INFOS['pos']
        wall_ver = WALL_INFOS['vertices']

        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(position=p, angle=0.0,
                                                fixtures=fixtureDef(shape=polygonShape(vertices=v), density=100.0,
                                                friction=0.0, categoryBits=0x001, restitution=1.0,))
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)

    def _build_drone(self):
        # create controller object
        while True:
            self.drone = self.world.CreateDynamicBody(position=self.drone_start_pos, angle=0.0,
                                                      fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                                                                          density=5.0, friction=0.1, categoryBits=0x0010,
                                                                          maskBits=0x003, restitution=0.0))
            self.drone.color1 = (0.5, 0.4, 0.9)
            self.drone.color2 = (0.3, 0.3, 0.5)
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            if self.game_over:
                self.world.DestroyBody(self.drone)
                self.game_over = False
            else:
                break
        return self.drone.position
    

    def _build_goal(self):
         
        #ok edit
        ''' 
        import random
        random_task = random.randint(0, 1)
        print('random_task' , random_task)
        # if odd
        if random_task == 0:
            #self.flag = True
            goal_pos = self.task_args['Goal']
        else:
            #self.flag = False
            goal_pos = Goal_change(self.task_args['Goal'])
         
        ##divide here##        
        if self.flag = True:
            goal_pos = self.task_args['Goal']
            self.flag = False
        else:
            goal_pos = Goal_change(self.task_args['Goal'])
            self.flag = True
        print('after_flag : ', self.flag)
        '''
        goal_pos = self.task_args['Goal']
        #print('goal_pos', goal_pos)
    
        self.goal = self.world.CreateDynamicBody(position=goal_pos, angle=0.0,
                                                fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                                                                    density=5.0, friction=0.1, categoryBits=0x002,
                                                                    maskBits=0x0010, restitution=0.0))
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)

    def _build_obs_range(self):
        self.obs_range_plt = self.world.CreateKinematicBody(position=(self.drone.position[0], self.drone.position[1]), angle=0.0,
                                                            fixtures=fixtureDef(shape=circleShape(radius=np.float64(self.max_obs_range), pos=(0, 0)),
                                                                                density=0, friction=0, categoryBits=0x0100,
                                                                                maskBits=0x000, restitution=0.3))
        self.obs_range_plt.color1 = (0.2, 0.2, 0.4)
        self.obs_range_plt.color2 = (0.6, 0.6, 0.6)

    def _clean_walls(self):
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _get_observation(self, position):
        delta_angle = 2* np.pi/self.num_beams
        ranges = [self.world.raytrace(position, i * delta_angle, self.max_obs_range) for i in range(self.num_beams)]
        ranges = np.array(ranges)
        return ranges

    @property
    def last_score(self):
        print(len(self.scores))
        return np.mean(self.scores)

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
        # create obstacles
        self.obstacles.build_obstacles()
        drone_pos = self._build_drone()
        
        # ok edit
        # goal change flag
        #task_name = str(self)
        #task_id_loc = task_name.rfind('Task')
        #task_id_loc = task_id_loc + len('Task')
        #task_id = task_name[task_id_loc]
        
        # create goal
        np.random.seed(np.random.randint(low=0, high=100000))
        self._build_goal()
        self._build_obs_range()

        self.drawlist = [self.obs_range_plt, self.drone, self.goal] + self.walls + self.obstacles.dynamic_bodies
        self._observe_lidar(drone_pos)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        
        self.prev_distance = self.distance
        return np.copy(self.array_observation())

    def step(self, action: np.iterable):
        action = np.asarray(action, dtype=np.float64)
        self.action = action
        self.drone.linearVelocity.Set(action[0], action[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.obstacles.step()

        self.energy -= 1e-3
        pos = np.array(self.drone.position)
        self._observe_lidar(pos)
        done = False
        if self.collision_done:
            done = self.game_over
        reward = self.prev_distance - self.distance
        self.prev_distance = self.distance 


        info = {}
        if self.energy <= 0:
            done = True
        if done or self.achieve_goal:
            if self.achieve_goal:
                reward = 1
                done = True
            info['is_success'] = self.achieve_goal
            info['energy'] = self.energy
            info['episode'] = {'r': reward, 'l': (0.15 - self.energy) * 1000}
        obs = np.copy(self.array_observation())
        self.obs_queue.append(obs)
        return obs, reward, done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def init_position_set(self, x, y):
        self.drone.position.Set(x, y)

class NavigationEnvAcc(NavigationEnvDefault):
    def __init__(self, obstacles_args, max_obs_range=3, max_speed=2, initial_speed=1, **kwargs):
        super().__init__(obstacles_args, max_obs_range, max_speed, initial_speed, )
        self.observation_meta_data['velocity'] = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.observation_meta_data_keys.append('velocity')
        self.timesteps = 0
        self.prev_distance = 0
        self.randomize_goal = False
        
    def dict_observation(self):
        dict_obs = super().dict_observation()
        velocity = self.drone.linearVelocity
        dict_obs['velocity'] = velocity
        return dict_obs

    @property
    def distance(self):
        return np.linalg.norm(self.goal.position - self.drone.position)

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
        self.obstacles.step()
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

    def reset(self):
        state = super().reset()
        self.prev_distance = self.distance
        self.timesteps = 0
        return state

    def randomize_goal(self):
        theta = np.pi * 2 * np.random.random()
        goal = denormalize_position(np.array([np.cos(theta), np.sin(theta)]) * 0.424 + np.array([0.5, 0.5]),
                                    W, H)
        self.task_args['Goal'] = goal
        self.goal.position.Set(goal[0], goal[1])

class NavigationEnvAccLidarObs(NavigationEnvAcc):
    def __init__(self, task_args, max_obs_range=3, max_speed=2, initial_speed=1, **kwargs):
        super().__init__(task_args, max_obs_range, max_speed, initial_speed, **kwargs)
        goalpos = MapGoal(task_args['Goal'])
        self.ID = goalpos

        self.observation_meta_data = {
            'position': gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
            'distance': gym.spaces.Box(np.array([0]), np.array([np.sqrt(W ** 2 + H ** 2)]), dtype=np.float32),
            'lidar': gym.spaces.Box(low=0, high=1, shape=(8,)),
            'energy': gym.spaces.Box(low=0, high=1, shape=(1,)),
            'velocity': gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32),
            'ID' : gym.spaces.Box(low=0, high=1, shape=(8,)),
        }

        self.observation_meta_data_keys = ['position', 'lidar', 'energy', 'distance', 'velocity', 'ID']

    def dict_observation(self):
        position = normalize_position(self.drone.position, W, H)
        distance = np.linalg.norm(normalize_position(self.drone.position, W, H) - (normalize_position(self.goal.position, W, H)))
        lidar = [l.fraction for l in self.lidar]
        velocity = self.drone.linearVelocity
        ID = self.ID
        
        dict_obs = {
            'position': position,
            'distance': distance,
            'lidar': lidar,
            'energy': self.energy,
            'velocity': velocity,
            'ID': ID,
        }
        return dict_obs

class NonStationaryNavigation(NavigationEnvAccLidarObs):
    def __init__(self, task_args, max_obs_range=3, max_speed=2, initial_speed=1, **kwargs):
        super().__init__(task_args, max_obs_range, max_speed, initial_speed, **kwargs)
        self.theta = 0
        self.dynamics = task_args['dynamics']
        self.uncertainty = task_args['uncertainty']

    def _build_goal(self):
        goal_pos = self.task_args['Goal']
        self.goal = self.world.CreateDynamicBody(position=goal_pos, angle=0.0,
                                                 fixtures=fixtureDef(shape=polygonShape(
                                                     vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                                                                     density=5.0, friction=0.1, categoryBits=0x002,
                                                                     maskBits=0x0010, restitution=0.0))
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)
        self.theta += np.pi * 2 * np.random.random()
        goal = denormalize_position(np.array([np.cos(self.theta), np.sin(self.theta)]) * 0.424 + np.array([0.5, 0.5]), W, H)
        self.task_args['Goal'] = goal
        self.goal.position.Set(goal[0], goal[1])

    def step(self, action):
        state, reward, done, info = super().step(action)
        if np.random.random() < self.dynamics:
            self.moving_goal()
        return state, reward, done, info

    def moving_goal(self):
        if np.random.random() > self.uncertainty:
            self.theta += np.pi / 180 * (np.random.normal(0, 1) * 10)
        else:
            self.theta = np.random.random() * np.pi * 2
        goal = denormalize_position((np.array([np.cos(self.theta), np.sin(self.theta)]) * (0.424)) + np.array([0.5, 0.5]), W, H)
        self.task_args['Goal'] = goal
        self.goal.position.Set(goal[0], goal[1])
        self.prev_distance = self.distance
        return

    @property
    def drone_start_pos(self):
        return denormalize_position([0.5, 0.5], W ,H)

import numpy as np
import copy

from .util import denormalize_position, normalize_position

# how many action do in 1 second
FPS = 5
# affects how fast-paced the game is, forces should be adjusted as well
SCALE = 30.0
# Drone's shape
DRONE_POLY = [
    (-11, +14), (-14, 0), (-14, -7),
    (+14, -7), (14, 0), (+11, +14)]
# obstacle initial velocity
OBSTACLE_INIT_VEL = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
                     (-1/np.sqrt(2), -1/np.sqrt(2))]
# map size
VIEWPORT_W = 600
VIEWPORT_H = 600

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

# Shape of Walls
WALL_POLY = [
    (-50, +20), (50, 20),
    (-50, -20), (50, -20)
]

HORIZON_LONG = [(W, -0.3), (W, 0.3), (-W, -0.3), (-W, 0.3)]
VERTICAL_LONG = [(-0.3, H), (0.3, H), (-0.3, -H), (0.3, -H)]
HORIZON_SHORT = [(W/3, -0.5), (W/3, 0.5), (-W/3, -0.5), (-W/3, 0.5)]
WALL_INFOS = {"pos": [(int(W /2), int(H)), (int(W), int(H/2)), (int(W / 2), 0), (0, int(H/2))],
              "vertices": [HORIZON_LONG, VERTICAL_LONG, HORIZON_LONG, VERTICAL_LONG]}

GOAL_POS = [[0.1, 0.1]]
CIRCULAR = 2
VERTICAL = 1
HORIZONTAL = 0

# x[0], x[1] is moving range for VERTICAL, HORIZONTAL
# distance with x[0] and x[1] is radius, and x[1] is center of circle for CIRCULAR

transform = lambda x: [denormalize_position(x[0], W, H), denormalize_position(x[1], W, H), x[2]]

obs_set = [
    [transform([[0.3, 0.3], [0.5, 0.5], CIRCULAR]),
     transform([[0.4, 0.4], [0.5, 0.5], CIRCULAR]),
     transform([[0.2, 0.2], [0.5, 0.5], CIRCULAR])],

    [transform([[0.2, 0.2], [0.8, 0.2], HORIZONTAL]),
     transform([[0.2, 0.8], [0.8, 0.8], HORIZONTAL]),
     transform([[0.4, 0.4], [0.5, 0.5], CIRCULAR]),
     transform([[0.2, 0.2], [0.5, 0.5], CIRCULAR])],

    [transform([[0.2, 0.2], [0.2, 0.8], VERTICAL]),
     transform([[0.8, 0.2], [0.8, 0.8], VERTICAL]),
     transform([[0.2, 0.2], [0.5, 0.5], CIRCULAR]),
     transform([[0.3, 0.3], [0.5, 0.5], CIRCULAR]),
     transform([[0.4, 0.4], [0.5, 0.5], CIRCULAR])],

    [transform([[0.2, 0.2], [0.8, 0.2], HORIZONTAL]),
     transform([[0.2, 0.8], [0.8, 0.8], HORIZONTAL]),
     transform([[0.2, 0.2], [0.2, 0.8], VERTICAL]),
     transform([[0.8, 0.2], [0.8, 0.8], VERTICAL]),
     transform([[0.2, 0.2], [0.5, 0.5], CIRCULAR]),
     transform([[0.3, 0.3], [0.5, 0.5], CIRCULAR]),
     transform([[0.4, 0.4], [0.5, 0.5], CIRCULAR])]
]

goal_set = [denormalize_position([0.2, 0.2], W, H),
            denormalize_position([0.2, 0.8], W, H),
            denormalize_position([0.8, 0.8], W, H),
            denormalize_position([0.8, 0.2], W, H),
            denormalize_position([0.0756, 0.5], W, H),
            denormalize_position([0.5, 0.0756], W, H),
            denormalize_position([0.924, 0.5], W, H),
            denormalize_position([0.5, 0.924], W, H)]

config_set = []

for obs in obs_set:
    for goal in goal_set:
        task_dict = {}
        task_dict['OBSTACLE_POSITIONS'] = obs
        task_dict['Goal'] = goal
        config_set.append(task_dict)

non_sta_config_set = []
for i in range(5):
    for j in range(5):
        task_dict = {}
        task_dict['OBSTACLE_POSITIONS'] = obs_set[0]
        task_dict['Goal'] = goal_set[0]
        task_dict['uncertainty'] = j * 0.25
        task_dict['dynamics'] = 0.01 * i
        non_sta_config_set.append(task_dict)
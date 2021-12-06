import numpy as np
import math as m


EPSILON = 1e-10


def random(size=None):
    return 2 * (np.random.random_sample(size) - 0.5)


def vector_process(vector, pos_info):
    vector = vector - np.array([pos_info.position.x_val, pos_info.position.y_val, pos_info.position.z_val])
    distance = np.linalg.norm(vector)
    if distance == 0:
        distance += 1e-10
    theta = m.acos(vector[2] / distance)
    if vector[0] == 0:
        vector[0] += 1e-10
    phi = m.atan(vector[1] / vector[0]) + np.pi/2
    if vector[0] < 0:
        phi = np.pi + phi
    return distance, theta, phi


def angle_add(a_max, angle1, angle2, mode='theta'):
    angle = angle1 + angle2
    if angle < 0 :
        if mode == 'theta':
            angle = 2 * a_max - angle
        else:
            angle = angle + a_max
    elif angle > a_max:
        if mode == 'theta':
            angle = 2 * a_max - angle
        else:
            angle = angle - a_max
    return angle

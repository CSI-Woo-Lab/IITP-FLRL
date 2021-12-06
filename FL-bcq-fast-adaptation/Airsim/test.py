import numpy as np
import math as m

def vector_process(vector, pos_info):
    vector = vector - pos_info
    distance = np.linalg.norm(vector)
    theta = m.acos(vector[2] / distance)
    if vector[0] == 0:
        vector[0] += 1e-10
    phi = m.atan(vector[1] / vector[0]) + np.pi/2
    if vector[0] < 0:
        phi = np.pi + phi
    return distance, theta, phi

if __name__ == '__main__':
    for i in range(18):
        np1 = np.array([m.cos(i*np.pi/9), m.sin(i*np.pi/9), -100], dtype=np.float32)
        np2 = np.array([0, 0, 0], dtype=np.float32)

        r, theta, phi = vector_process(np1, np2)

        print(r, int(theta/(np.pi/8)), int(phi/(np.pi/8)))

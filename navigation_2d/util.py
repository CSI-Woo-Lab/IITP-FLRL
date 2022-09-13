import numpy as np

def normalize_position(x, width, height):
    y = np.copy(x)
    y[0] = x[0] / width
    y[1] = x[1] / height
    return y

def denormalize_position(x, width, height):
    y = np.copy(x)
    y[0] = x[0] * width
    y[1] = x[1] * height
    return y

def rotation_4(z):
    x = z[0]
    y = z[1]
    rot = [[x, y], [-x, y], [-x, -y], [x, -y]]
    return rot

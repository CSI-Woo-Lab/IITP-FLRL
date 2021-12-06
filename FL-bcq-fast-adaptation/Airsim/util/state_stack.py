import numpy as np
import copy


class StateStack(object):
    def __init__(self, shape, max_len):
        self.shape = (shape[0], shape[1], max_len)
        self.data = np.zeros((shape[0], shape[1], max_len))
        self.max_len = max_len

    def add(self, new_data):
        self.data = self.data[:, :, 0: self.max_len - 1]
        self.data = np.concatenate([new_data, self.data], axis=2)

    def show_data(self):
        return copy.deepcopy(self.data).flatten()

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]
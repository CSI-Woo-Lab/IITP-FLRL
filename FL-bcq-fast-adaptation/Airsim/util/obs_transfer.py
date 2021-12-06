import numpy as np

class SensorNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, state):
        distrib = np.random.normal(self.mean, self.std, state.shape)
        observation = state + distrib
        return observation


class SensorOff(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, state):
        off = np.random.random(state.shape)
        observation = np.where(off < self.prob, 0, state)
        return observation


class SensorDelay(object):
    def __init__(self, update_prob):
        self.update_prob = update_prob

    def __call__(self, prev_state, state):
        update = np.random.random()
        observation = np.where(update < self.update_prob, state, prev_state)
        return observation


class SensorBias(object):
    def __init__(self, ratio, prob):
        self.ratio = ratio
        self.prob = prob

    def __call__(self, state):
        bias = np.random.random(state.shape)
        bias = np.where(bias < self.prob, state * self.ratio, 0)
        observation = state + bias
        return observation


class obs_transfer(object):
    def __init__(self, noise_mean, noise_std, off_prob, update_prob, bias_ratio, bias_prob, state_shape):
        self.noise = SensorNoise(noise_mean, noise_std)
        self.off = SensorOff(off_prob)
        self.delay = SensorDelay(update_prob)
        self.bias = SensorBias(bias_ratio, bias_prob)
        self.prev_state = np.zeros(state_shape)

    def __call__(self, state):
        obs = self.delay(state, self.prev_state)
        obs = self.noise(obs)
        obs = self.bias(obs)
        obs = self.off(obs)
        return obs

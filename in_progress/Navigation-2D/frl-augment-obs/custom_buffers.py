import numpy as np
from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class AmplitudeSampleReplayBuffer(ReplayBuffer):
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)

        # amplitude
        z = np.random.uniform(0.6, 1.2, obs.shape)
        obs *= z
        next_obs *= z

        data = (
            obs,
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class NoiseSampleReplayBuffer(ReplayBuffer):
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)

        # noise
        z = np.random.normal(0, 1, obs.shape)
        obs += z
        next_obs += z

        data = (
            obs,
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

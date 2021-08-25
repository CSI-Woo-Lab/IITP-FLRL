import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LLCC-v0',
    entry_point='gym_LLCC.envs:LLCC',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LLCC_left-v0',
    entry_point='gym_LLCC.envs:LLCC_left',
    max_episode_steps=1000,
    reward_threshold=200,
)
register(
    id='LLCC_right-v0',
    entry_point='gym_LLCC.envs:LLCC_right',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LLCC_wind_left-v0',
    entry_point='gym_LLCC.envs:LLCC_wl',
    max_episode_steps=1000,
    reward_threshold=200,
)
register(
    id='LLCC_wind_right-v0',
    entry_point='gym_LLCC.envs:LLCC_wr',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LLCC_chaos-v0',
    entry_point='gym_LLCC.envs:LLCC_chaos',
    max_episode_steps=1000,
    reward_threshold=200,
)
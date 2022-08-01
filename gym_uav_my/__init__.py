from gym.envs.registration import register
from gym_uav_my.env.utils import GameStateType
from gym_uav_my.env.config import NamedDict
from gym_uav_my.env.utils import gen_obs_map


register(
    id='Multi-Objective-Uav-v0',
    entry_point='gym_uav_my.env.gym_uav:GymUav'
)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='to_origin-v0',
    entry_point='gym_to_origin.envs:ToOriginEnv',
)

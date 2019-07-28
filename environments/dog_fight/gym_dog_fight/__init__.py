import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='dog_fight-v1',
    entry_point='gym_dog_fight.envs:DogFightEnv',
)

register(
    id='dog_fight-v2',
    entry_point='gym_dog_fight.envs:DogFightEnv2',
)

register(
    id='dog_fight-v3',
    entry_point='gym_dog_fight.envs:DogFightEnv3',
)
register(
    id='dog_fight-v4',
    entry_point='gym_dog_fight.envs:DogFightEnv4',
)

import gym
import gym_dog_fight
import time

env = gym.make('dog_fight-v3')
env.seed(0)

env.reset()

while(True):
    env.step(0)
    env.step(180)

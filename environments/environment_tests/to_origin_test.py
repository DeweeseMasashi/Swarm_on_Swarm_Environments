import gym
import gym_to_origin
import time

env = gym.make('to_origin-v0')
env.seed(0)
print(env.reset())

total_reward = 0
while(not env.is_done()):
    env.step(225)
    total_reward += env._get_reward()
    print(total_reward)
    env._render()
print("total reward: ", total_reward)

env.reset()
total_reward = 0
while(not env.is_done()):
    env.step(45)
    total_reward += env._get_reward()
    print(total_reward)
    env._render()
print("total reward: ", total_reward)

env.step(0)
env._render()
print(env._get_state())
print(env._get_reward())
print(env.reset())


env.reset()
print("prev_state", env.prev_state)
print("current state", env._get_state())
print("transition details: ",env.step(0))
print("prev_state", env.prev_state)
print("current state", env._get_state())

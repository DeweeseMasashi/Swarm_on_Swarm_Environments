import gym
import gym_dog_fight
import time

env = gym.make('dog_fight-v3')
env.seed(0)

env.reset()
total_reward = 0

for i in range(50):
    env.step(270)
    total_reward += env._get_reward()
    env._render()
    print(env._get_state())

while(not env.is_done()):
    env.step(180)
    total_reward += env._get_reward()
    env._render()
    print(env._get_state())
print("total reward: ", total_reward)


env.reset()
total_reward = 0
for i in range(50):
    env.step(180)
    total_reward += env._get_reward()
    env._render()
    print(env._get_state())

while(not env.is_done()):
    env.step(270)
    total_reward += env._get_reward()
    env._render()
    print(env._get_state())
print("total reward: ", total_reward)


env.reset()
total_reward = 0
for i in range(100):
    env.step(180)
    total_reward += env._get_reward()
    env._render()

for i in range(50):
    env.step(270)
    total_reward += env._get_reward()
    env._render()

while(not env.is_done()):
    env.step(0)
    total_reward += env._get_reward()
    env._render()
print("total reward: ", total_reward)

env.reset()
total_reward = 0
for i in range(100):
    env.step(270)
    total_reward += env._get_reward()
    env._render()

for i in range(50):
    env.step(180)
    total_reward += env._get_reward()
    env._render()

while(not env.is_done()):
    env.step(90)
    total_reward += env._get_reward()
    env._render()
print("total reward: ", total_reward)


print(env.reset())

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

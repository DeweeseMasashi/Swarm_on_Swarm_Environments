#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.

Each episode is selling a single banana.
"""

# core modules
import logging
import logging.config
import math
import pkg_resources
import random


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

# 3rd party modules
from gym import spaces
import cfg_load
import gym
import numpy as np


path = 'config.yaml'  # always use slash in packages
filepath = pkg_resources.resource_filename('gym_to_origin', path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config['LOGGING'])

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1 - y2)*(y1 - y2))
def in_radius(x1, y1, x2, y2, r):
    return distance(x1, y1, x2, y2) < r

class ToOriginEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        '''
        self.__version__ = "0.1.0"
        logging.info("BananaEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_PRICE = 2.0
        self.TOTAL_TIME_STEPS = 2

        self.curr_step = -1
        self.is_banana_sold = False

        # Define what the agent can do
        # Sell at 0.00 EUR, 0.10 Euro, ..., 2.00 Euro
        self.action_space = spaces.Discrete(21)

        # Observation is the remaining time
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([self.TOTAL_TIME_STEPS,  # remaining_tries
                         ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        '''
        self.max_distance = 6
        self.start_x = 5
        self.start_y = 5
        self.position = [self.start_x, self.start_y]

        self.min_angle = 0
        self.max_angle = 360

        self.action_space = spaces.Box(low=self.min_angle, high=self.max_angle, shape=(1,),dtype=np.float32)
        low = np.array([-self.max_distance, -self.max_distance])
        high = np.array([self.max_distance, self.max_distance])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.movement_length = 0.1
        self.out_of_bounds = False
        self.curr_step = -1
        self.prev_state = self._get_state()

        self.curr_episode = -1
        self.action_episode_memory = []


        self.init_pyplot()
        self.L = 0.3


    def init_pyplot(self):

        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-self.max_distance, self.max_distance])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-self.max_distance, self.max_distance])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-self.max_distance, self.max_distance])
        self.ax.set_zlabel('Z')
        #self.ax.set_title('Quadcopter Simulation')

        l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)
        self.vehicle = [l1, l2, hub]

    def update_graphics(self):
        L = self.L
        points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
        points[0,:] += self.position[0]
        points[1,:] += self.position[1]
        points[2,:] += 0
        self.vehicle[0].set_data(points[0,0:2],points[1,0:2])
        self.vehicle[0].set_3d_properties(points[2,0:2])
        self.vehicle[1].set_data(points[0,2:4],points[1,2:4])
        self.vehicle[1].set_3d_properties(points[2,2:4])
        self.vehicle[2].set_data(points[0,5],points[1,5])
        self.vehicle[2].set_3d_properties(points[2,5])
        plt.pause(0.000000000000001)

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        '''
        if self.is_banana_sold:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_banana_sold, {}
        '''
        if self.is_done():
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_done(), {}

    def is_done(self):
        return self.out_of_bounds or in_radius(self.position[0], self.position[1], 0, 0, self.movement_length)

    def _take_action(self, action):
        """
        self.action_episode_memory[self.curr_episode].append(action)
        self.price = ((float(self.MAX_PRICE) /
                      (self.action_space.n - 1)) * action)

        chance_to_take = get_chance(self.price)
        banana_is_sold = (random.random() < chance_to_take)

        if banana_is_sold:
            self.is_banana_sold = True

        remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = (remaining_steps <= 0)
        throw_away = time_is_over and not self.is_banana_sold
        if throw_away:
            self.is_banana_sold = True  # abuse this a bit
            self.price = 0.0
        """
        self.action_episode_memory[self.curr_episode].append(action)

        self.prev_state = self._get_state()[:]
        x_adjustment = math.cos(math.radians(action)) * self.movement_length
        y_adjustment = math.sin(math.radians(action)) * self.movement_length

        self.position[0] += x_adjustment
        self.position[1] += y_adjustment

        if distance(self.position[0], self.position[1], 0, 0) > distance(self.max_distance, self.max_distance, 0, 0):
            self.out_of_bounds = True

    def _get_reward(self):
        """Reward is given for a sold banana."""
        if self.out_of_bounds:
            return -10
        elif in_radius(self.position[0], self.position[1], 0, 0, self.movement_length):
            return 10
        else:
            curr_state = self._get_state()
            return ((distance(self.prev_state[0], self.prev_state[1], 0, 0) - distance(curr_state[0], curr_state[1], 0, 0))/(distance(self.start_x, self.start_y, 0, 0))) * 10

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.out_of_bounds = False
        self.position[0] = self.start_x
        self.position[1] = self.start_y
        return self._get_state()

    def _render(self, mode='human', close=False):
        self.update_graphics()

    def _get_state(self):
        """Get the observation."""
        return self.position

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

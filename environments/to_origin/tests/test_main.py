#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_to_origin


class Environments(unittest.TestCase):

    def test_env(self):
        env = gym.make('to_origin-v0')
        env.seed(0)
        print(env.reset())
        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)
        env.step(5)
        print(env.reset())

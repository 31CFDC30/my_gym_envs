# -*- coding: utf-8 -*-
"""
Created on 2021/9/27 15:40

@author: qk
"""

import dynamics

import numpy as np
from scipy import integrate

import gym
from gym import spaces
import logging
# from gym.utils import seeding, EzPickle

logger = logging.getLogger(__name__)


class ParafoilEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.state = np.zeros(21)

        self.done = 0

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.ode = integrate.ode(dynamics.dynamics).set_integrator("none")

        self.reset()

    def step(self, action):
        pass

    def reset(self):
        self.state = np.array([0, 0, 0, 0, -0.0293, 0, 0, 0.2462, 0, 10.623,  0, 3.6301, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

    def storage(self):
        pass


if __name__ == '__main__':
    env = gym.make('Parafoil-v1')

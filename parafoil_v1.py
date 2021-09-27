# -*- coding: utf-8 -*-
"""
Created on 2021/9/27 15:40

@author: qk
"""
import logging
import random
import gym

logger = logging.getLogger(__name__)


class ParafoilEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action):
        pass

    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    pass

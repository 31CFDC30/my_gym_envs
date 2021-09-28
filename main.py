# -*- coding: utf-8 -*-
"""
Created on 2021/9/27 15:37

@author: qk
"""
from tqdm import tqdm
import gym


def main():
    env = gym.make('Parafoil-v1')
    env.reset()
    for i in range(100000):
        if 5000 < i < 50000:
            env.step((0.1, 0.2))
        else:
            env.step((0.0, 0.0))
        env.render()


if __name__ == '__main__':
    main()

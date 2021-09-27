# -*- coding: utf-8 -*-
"""
Created on 2021/9/27 15:37

@author: qk
"""


def main():
    import gym

    env = gym.make('GridWorld-v1')
    env.reset()
    env.render()
    env.close()


if __name__ == '__main__':
    main()

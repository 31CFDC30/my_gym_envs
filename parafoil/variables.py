# -*- coding: utf-8 -*-
"""
Created on 2021/6/9 9:28

@author: qk
"""
import numpy as np
t_now = 0   # 由于不直接调用dynamics，而计算CVR、CVL需要知道当前时刻，因此需要在调用r.integrate(t[i]) 前更新
""" 控制量 """
CV_target = np.array([0.0, 0.0])
CV_now = np.array([0.0, 0.0])
CV_next = np.array([0.0, 0.0])    # 下一时刻t[n+1]的Control Vault，在ode外更新
CVL, CVR = (0.0, 0.0)   # 由于ode45会在t[n]和t[n+1]之间不同时刻t下多次调用dynamics函数，CVR和CVL用于计算这些时刻实际Control Vault值，在ode内（dynamics）更新

""" 位置 """
# Vcg = np.array(y[9:12]).reshape(3, 1)  # 连接点C的速度
# euler_b = np.array(y[3:6]).reshape(3, 1)  # 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）
# euler_p = np.array(y[6:9]).reshape(3, 1)  # 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）



""" 速度 """

""" 加速度 """



def main():
    pass


if __name__ == '__main__':
    main()

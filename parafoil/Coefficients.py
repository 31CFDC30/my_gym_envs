# -*- coding: utf-8 -*-
"""
Created on 2021/6/8 16:41

@author: qk
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d

data_CD = np.array([[-20, 0.1, -20, 0.08, -20, 0.05],
                    [-10.269, 0.12112, -10.398, 0.13572, -10.398, 0.15065],
                    [-4.4237, 0.13804, -4.8818, 0.16127, -5.12, 0.20806],
                    [-0.08063, 0.14866, -0.31886, 0.1845, -0.08063, 0.26513],
                    [5.7651, 0.1845, 6.1132, 0.27177, 5.7651, 0.35871],
                    [10.456, 0.21237, 10.804, 0.33316, 9.9798, 0.4264],
                    [15.257, 0.27376, 15.019, 0.40318, 14.909, 0.51135],
                    [19.692, 0.32685, 19.692, 0.48779, 19.82, 0.60028],
                    [25.794, 0.42209, 24.86, 0.57905, 24.86, 0.68722],
                    [30.009, 0.49841, 29.185, 0.65769, 30.009, 0.77416],
                    [34.92, 0.58137, 34.81, 0.75094, 34.92, 0.83589],
                    [39.96, 0.66201, 39.831, 0.82527, 39.96, 0.90159],
                    [45.457, 0.74662, 46.044, 0.90358, 46.172, 0.94804],
                    [50.258, 0.81896, 50.497, 0.94605, 51.541, 0.98222],
                    [55.536, 0.83356, 54.711, 0.98421, 56.232, 1.0035],
                    [59.971, 0.84418, 59.861, 1.0141, 59.751, 1.0141],
                    [65.725, 0.8465, 65.138, 1.0204, 65.945, 0.98853],
                    [70.159, 0.85049, 69.94, 1.0161, 70.288, 0.9716],
                    [75.089, 0.82726, 74.612, 0.98421, 75.089, 0.93112],
                    [80.11, 0.81233, 79.89, 0.94804, 80, 0.88666]])

data_CL = np.array([[-20, 0.3, -20, 0.2, -20, 0.1],
                    [-10.167, 0.3498, -10.167, 0.35167, -10.034, 0.42483],
                    [-5.0514, 0.38238, -3.7201, 0.4761, -5.0514, 0.54953],
                    [0.04564, 0.45234, -0.06847, 0.57677, -0.06847, 0.77303],
                    [3.9445, 0.55113, 4.7813, 0.7749, 6.9684, 0.9968],
                    [9.7642, 0.67931, 10.011, 0.97784, 12.56, 0.91482],
                    [14.88, 0.64166, 14.88, 0.86515, 15.108, 0.87717],
                    [20.091, 0.67583, 19.977, 0.9028, 20.091, 0.91642],
                    [24.713, 0.73378, 25.93, 0.80561, 24.96, 0.79199],
                    [29.334, 0.60587, 30.799, 0.7474, 29.829, 0.72016],
                    [34.812, 0.55113, 36.257, 0.71162, 34.564, 0.66035],
                    [39.909, 0.55808, 39.909, 0.67423, 39.661, 0.62296],
                    [47.193, 0.54099, 44.892, 0.65367, 45.976, 0.55808],
                    [51.567, 0.51375, 50.73, 0.57356, 49.989, 0.49826],
                    [55.941, 0.48304, 55.333, 0.51188, 53.887, 0.43178],
                    [59.84, 0.45581, 60.087, 0.45394, 59.84, 0.33458],
                    [64.937, 0.40107, 66.649, 0.40801, 64.823, 0.25421],
                    [69.92, 0.3498, 70.053, 0.37196, 70.167, 0.1741],
                    [75.017, 0.27984, 74.903, 0.24913, 75.15, 0.08705],
                    [80, 0.21335, 79.886, 0.1263, 79.772, 0.00347]])

data_Cmy = np.array([[-20, -0.01, -20, -0.011, -20, -0.015],
                     [-10.079, -0.06667, -10.079, -0.06866, -10.183, -0.10787],
                     [-5.048, -0.10671, -4.9432, -0.11846, -4.9432, -0.14907],
                     [-0.13974, -0.1173, -0.03493, -0.1671, -0.13974, -0.20929],
                     [5.5546, -0.13848, 4.7686, -0.21789, 4.8734, -0.25909],
                     [10.131, -0.14692, 9.9039, -0.27596, 9.0131, -0.31617],
                     [15.266, -0.18182, 14.026, -0.32361, 14.026, -0.36481],
                     [19.843, -0.21144, 19.843, -0.37755, 19.616, -0.41775],
                     [25.415, -0.25164, 25.869, -0.42404, 25.869, -0.45464],
                     [30.218, -0.28026, 29.991, -0.45365, 29.886, -0.47896],
                     [35.022, -0.3203, 34.9, -0.47681, 34.009, -0.50113],
                     [39.93, -0.35951, 39.93, -0.49799, 39.808, -0.5329],
                     [45.17, -0.40071, 45.834, -0.52347, 46.061, -0.55193],
                     [49.974, -0.43876, 49.852, -0.54035, 49.974, -0.56467],
                     [54.76, -0.47053, 54.987, -0.55722, 55.319, -0.56996],
                     [59.895, -0.50329, 59.79, -0.5741, 59.895, -0.57525],
                     [65.712, -0.47896, 65.258, -0.58055, 65.817, -0.55408],
                     [70.061, -0.45895, 69.834, -0.5827, 69.939, -0.53935],
                     [74.41, -0.43347, 74.638, -0.55722, 73.852, -0.52545],
                     [80, -0.39856, 80, -0.52876, 80, -0.50544]])

CL_zero_rfa0 = data_CL[:, 0].reshape(-1)
CL_zero_zero = data_CL[:, 1].reshape(-1)
CL_half_rfa0 = data_CL[:, 2].reshape(-1)
CL_half_half = data_CL[:, 3].reshape(-1)
CL_full_rfa0 = data_CL[:, 4].reshape(-1)
CL_full_full = data_CL[:, 5].reshape(-1)
fun_CL_zero = interp1d(CL_zero_rfa0, CL_zero_zero, kind='cubic')
fun_CL_half = interp1d(CL_half_rfa0, CL_half_half, kind='cubic')
fun_CL_full = interp1d(CL_full_rfa0, CL_full_full, kind='cubic')

CD_zero_rfa0 = data_CD[:, 0].reshape(-1)
CD_zero_zero = data_CD[:, 1].reshape(-1)
CD_half_rfa0 = data_CD[:, 2].reshape(-1)
CD_half_half = data_CD[:, 3].reshape(-1)
CD_full_rfa0 = data_CD[:, 4].reshape(-1)
CD_full_full = data_CD[:, 5].reshape(-1)
fun_CD_zero = interp1d(CD_zero_rfa0, CD_zero_zero, kind='cubic')
fun_CD_half = interp1d(CD_half_rfa0, CD_half_half, kind='cubic')
fun_CD_full = interp1d(CD_full_rfa0, CD_full_full, kind='cubic')

Cmy_zero_rfa0 = data_Cmy[:, 0].reshape(-1)
Cmy_zero_zero = data_Cmy[:, 1].reshape(-1)
Cmy_half_rfa0 = data_Cmy[:, 2].reshape(-1)
Cmy_half_half = data_Cmy[:, 3].reshape(-1)
Cmy_full_rfa0 = data_Cmy[:, 4].reshape(-1)
Cmy_full_full = data_Cmy[:, 5].reshape(-1)
fun_Cmy_zero = interp1d(Cmy_zero_rfa0, Cmy_zero_zero, kind='cubic')
fun_Cmy_half = interp1d(Cmy_half_rfa0, Cmy_half_half, kind='cubic')
fun_Cmy_full = interp1d(Cmy_full_rfa0, Cmy_full_full, kind='cubic')


def cal(rfa):
    """升力系数"""
    CL_zero_rfa = fun_CL_zero(rfa)
    CL_half_rfa = fun_CL_half(rfa)
    CL_full_rfa = fun_CL_full(rfa)
    """阻力系数"""
    CD_zero_rfa = fun_CD_zero(rfa)
    CD_half_rfa = fun_CD_half(rfa)
    CD_full_rfa = fun_CD_full(rfa)
    """力矩系数"""
    Cmy_zero_rfa = fun_Cmy_zero(rfa)
    Cmy_half_rfa = fun_Cmy_half(rfa)
    Cmy_full_rfa = fun_Cmy_full(rfa)
    return CL_zero_rfa, CL_half_rfa, CL_full_rfa, CD_zero_rfa, CD_half_rfa, CD_full_rfa, Cmy_zero_rfa, Cmy_half_rfa, Cmy_full_rfa


def plot():
    x_new = np.linspace(-10, 70, 10000)
    # kind_lst = ['nearest', 'zero', 'slinear', 'cubic', 'previous', 'next']

    y_new = fun_CL_half(x_new)
    pl.plot(x_new, y_new)

    pl.legend(loc="lower right")
    pl.show()


def main():
    plot()
    pass


if __name__ == '__main__':
    main()
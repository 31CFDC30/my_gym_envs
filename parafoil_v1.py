# -*- coding: utf-8 -*-
"""
Created on 2021/9/27 15:40

@author: qk
"""


import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import integrate
import copy
import matplotlib.pyplot as plt
import time

import gym
from gym import spaces
import logging
# from gym.utils import seeding, EzPickle

logger = logging.getLogger(__name__)


def anti_symmetric_matrix(vector):

    matrix = np.array([[0, -vector[2][0], vector[1][0]],
                       [vector[2][0], 0, -vector[0][0]],
                       [-vector[1][0], vector[0][0], 0]])
    return matrix


def transform_matrix_omega2euler(euler):
    """从角速度变换欧拉角变换率的矩阵"""
    sin_phi, sin_theta = np.sin(euler[0][0]), np.sin(euler[1][0])
    cos_phi, cos_theta = np.cos(euler[0][0]), np.cos(euler[1][0])
    tan_phi, tan_theta = np.tan(euler[0][0]), np.tan(euler[1][0])
    matrix = np.array([[1, sin_phi * tan_theta, cos_phi * tan_theta],
                       [0, cos_phi, -sin_phi],
                       [0, sin_phi / cos_theta, cos_phi / cos_theta]])
    return matrix


def transform_matrix_g2b(euler):
    """ 也可以计算T_g2p """
    sin_phi, sin_theta, sin_psi = np.sin(euler[0][0]), np.sin(euler[1][0]), np.sin(euler[2][0])
    cos_phi, cos_theta, cos_psi = np.cos(euler[0][0]), np.cos(euler[1][0]), np.cos(euler[2][0])
    tan_phi, tan_theta, tan_psi = np.tan(euler[0][0]), np.tan(euler[1][0]), np.tan(euler[2][0])
    matrix = np.array([[cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
                       [sin_theta * sin_phi * cos_psi - cos_phi * sin_psi,
                        sin_theta * sin_phi * sin_psi + cos_phi * cos_psi, sin_phi * cos_theta],
                       [sin_theta * cos_phi * cos_psi + sin_phi * sin_psi,
                        sin_theta * cos_phi * sin_psi - sin_phi * cos_psi, cos_phi * cos_theta]])
    return matrix


""" xc yc zc/ funb sitab ksib /funp sitap ksip /uc vc wc /pb qb rb /pp qp rp /Fcjx Fcjy Fcjz """
""" C点位置，载荷欧拉角，翼伞欧拉角，C点线速度，载荷角速度，翼伞角速度，？"""
""" 第二组 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z） """


class ParafoilEnv(gym.Env):
    def __init__(self):
        """ 动力学初始化 """
        """ 常量：翼伞、载荷基本参数 """
        mb0, mp0 = 135, 6     # 质量参数
        rou, a, b, c, th = 1.225, 1, 7.5, 3.75, 0.6750  # 附加质量模型参数
        lx, ly, lz = 0.6, 0.5, 0.4  #
        rcbx, rcby, rcbz, rcpx, rcpy, rcpz = 0, 0, 1.1206, -1.0126, 0, -7.0046

        """ 常量：翼伞、载荷质量矩阵 """
        a1 = a / b
        t1 = th / b
        t2 = th / c
        A1 = b / c
        S = b * c
        A = rou * 0.666 * (1 + 8 / 3 * a1 ** 2) * th ** 2 * b
        B = rou * 0.267 * (th ** 2 + 2 * a ** 2 * (1 - t1 ** 2)) * c
        C = rou * 0.785 * (1 + 2 * a1 ** 2 * (1 - t1 ** 2)) ** 0.5 * A1 / (1 + A1) * c ** 2 * b
        mp1 = np.array([[A, 0, 0],
                        [0, B, 0],
                        [0, 0, C]])     # 翼伞附加质量矩阵
        mp01 = np.identity(3) * mp0     # 翼伞真实质量矩阵（推断）
        self.mp = mp01 + mp1  # 翼伞总质量矩阵
        self.mb = np.identity(3) * mb0  # 载荷总质量

        """ 常量：翼伞、载荷转动惯量 """
        Ibx = mb0 * (ly ** 2 + lz ** 2) / 12
        Iby = mb0 * (lx ** 2 + lz ** 2) / 12
        Ibz = mb0 * (lx ** 2 + ly ** 2) / 12
        self.Ib = np.array([[Ibx, 0, 0],  # 载荷转动惯量
                       [0, Iby, 0],
                       [0, 0, Ibz]])
        Ipx = mp0 * ((a + th) ** 2 + b ** 2) / 12
        Ipy = mp0 * ((a + th) ** 2 + c ** 2) / 12
        Ipz = mp0 * (b ** 2 + c ** 2) / 12
        Ip0 = np.array([[Ipx, 0, 0],
                        [0, Ipy, 0],
                        [0, 0, Ipz]])   # 翼伞真实转动惯量（推断）
        IA = rou * 0.055 * A1 / (1 + A1) * b * S ** 2
        IB = rou * 0.0308 * A1 / (1 + A1) * (1 + np.pi / 6 * (1 + A1) * A1 * a1 ** 2 * t2 ** 2) * c ** 3 * S
        IC = rou * 0.0555 * (1 + 8 * a1 ** 2) * th ** 2 * b ** 3
        Ip1 = np.array([[IA, 0, 0],
                        [0, IB, 0],
                        [0, 0, IC]])    # 翼伞附加转动惯量
        self.Ip = Ip0 + Ip1  # 翼伞转动惯量 = 翼伞真实转动惯量 + 翼伞附加转动惯量




        """ 变量 """
        self.Vcg = np.zeros((3, 1))     # 状态：地速：连接点c点相对于地面g的速度速度
        self.euler_p = np.zeros((3, 1))     # 状态：状态中的翼伞欧拉角 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）
        self.euler_b = np.zeros((3, 1))     # 状态：状态中的载荷欧拉角 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）
        self.xsjzp = np.zeros((3, 1))       # 翼伞的xz矩阵
        self.xsjzb = np.zeros((3, 1))       # 载荷的xz矩阵
        self.omiga_p = np.zeros((3, 1))     # 状态：翼伞三个轴的角度变化率（角速度）
        self.rate_p = np.zeros((3, 1))      # 翼伞三个轴的欧拉角变化率
        self.omiga_b = np.zeros((3, 1))     # 状态：载荷三个轴的角度变化率（角速度）
        self.rate_b = np.zeros((3, 1))      # 载荷三个轴的欧拉角变化率
        self.omigb1 = np.identity(3)        #
        self.omigp1 = np.identity(3)        #

        self.T_g2b = np.identity(3)     # 从大地坐标系g到载荷体坐标系b的转换矩阵
        self.T_g2p = np.identity(3)     # 从大地坐标系g到翼伞体坐标系p的转换矩阵

        """ 控制量 """
        self.CV_target = np.array([0.0, 0.0])
        self.CV = np.array([0.0, 0.0])
        self.CV_old = np.array([0.0, 0.0])

        """ 位置矢量 """
        self.rcb = np.zeros((3, 1))
        self.rcb1 = np.identity(3)
        self.rcp = np.zeros((3, 1))
        self.rcp1 = np.identity(3)

        """ 力、力矩 """
        self.Fgb = np.zeros((3, 1))
        self.Fgp = np.zeros((3, 1))
        self.Fap = np.zeros((3, 1))
        self.Mab = np.zeros((3, 1))
        self.Mcb = np.zeros((3, 1))
        self.Mcp = np.zeros((3, 1))
        self.Map = np.zeros((3, 1))
        self.Vb = np.zeros((3, 1))
        self.Vb_norm = 0
        self.Fab = np.zeros((3, 1))
        self.Vp = np.zeros((3, 1))
        self.Vp_norm = 0
        self.q = 0  # 气动压力，与翼伞速度（标量）相关

        self.B = np.zeros([12, 12])  # 力和力矩向量
        self.A_matrix = np.zeros([12, 12])  # 质量矩阵
        self.acc = np.zeros((12, 1))    # 加速度向量 = 力和力矩向量 / 质量矩阵
        self.y_dot = np.zeros(21)    # 状态y的增量

        self.coefficients = Coefficients()

        """ 环境参数 """
        self.state = [0.0, 0.0, -3000, 0.0, -0.0293, 0.0, 0.0, 0.2462, 0.0, 10.623, 0.0, 3.6301, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ctrl_step = 0.01
        self.step_count = 0
        self.t = 0.0
        self.t_step = 0.01

        self.ode = integrate.ode(self.dynamics).set_integrator("none")
        self.ode.set_initial_value(self.state, 0)

        self.states_log = copy.deepcopy([self.state])
        self.display = Display()
        self.done = 0

    def dynamics(self, t, y):
        """ dynamics """
        """ 常量：翼伞、载荷基本参数 """
        mb0, mp0 = 135, 6  # 质量参数
        rou, a, b, c, th = 1.225, 1, 7.5, 3.75, 0.6750  # 附加质量模型参数
        lx, ly, lz = 0.6, 0.5, 0.4  #
        rcbx, rcby, rcbz, rcpx, rcpy, rcpz = 0, 0, 1.1206, -1.0126, 0, -7.0046
        Ab = 0.5
        CDb = 1.05
        Ap = b * c  # 迎风面积

        """气动力方程调用参数"""
        CL_derta = 0.2350
        CD_derta = 0.0957
        CY_beita = -0.0095
        CY_r = -0.0060
        CY_derta = 0.1368
        Cmx_beita = -0.0014
        Cmx_p = -0.1330
        Cmx_r = 0.0100
        Cmx_derta = -0.0063
        Cmy_q = -1.8640
        Cmy_derta = 0.2940
        Cmz_beita = 0.0005
        Cmz_p = -0.0130
        Cmz_r = -0.0350
        Cmz_derta = 0.0155

        """ """
        self.Vcg = np.array(y[9:12]).reshape(3, 1)  # 读取c点地速
        self.euler_b = np.array(y[3:6]).reshape(3, 1)  # 读取载荷欧拉角 分别为phi theta psi（滚转x 俯仰y 偏航z）
        self.euler_p = np.array(y[6:9]).reshape(3, 1)  # 读取翼伞欧拉角 分别为phi theta psi（滚转x 俯仰y 偏航z）
        self.xsjzb = transform_matrix_omega2euler(self.euler_b)
        self.xsjzp = transform_matrix_omega2euler(self.euler_p)
        self.omiga_b = np.array(y[12:15]).reshape(3, 1)  # 载荷角速度
        self.rate_b = self.xsjzb @ self.omiga_b  # 载荷欧拉角变化率
        self.omiga_p = np.array(y[15:18]).reshape(3, 1)  # 翼伞角速度
        self.rate_p = self.xsjzp @ self.omiga_p  # 翼伞欧拉角变化率

        self.omigb1 = anti_symmetric_matrix(self.omiga_b)
        self.omigp1 = anti_symmetric_matrix(self.omiga_p)
        self.T_g2b = transform_matrix_g2b(self.euler_b)
        self.T_g2p = transform_matrix_g2b(self.euler_p)

        self.T_g2b = transform_matrix_g2b(self.euler_b)
        self.T_g2p = transform_matrix_g2b(self.euler_p)

        """ 计算位置矢量及其anti_symmetric_matrix（是否需要不停更新？） """
        self.rcb = np.array([rcbx, rcby, rcbz]).reshape(3, 1)  # 点c到载荷b的位置矢量r
        self.rcb1 = anti_symmetric_matrix(self.rcb)
        self.rcp = np.array([rcpx, rcpy, rcpz]).reshape(3, 1)  # 点c到翼伞p的位置矢量r
        self.rcp1 = anti_symmetric_matrix(self.rcp)

        """ 计算控制（下拉）量 """
        # self.CVL, self.CVR = self.CV_now + (t - self.t_now) * (self.CV_next - self.CV_now)
        CVL, CVR = self.CV_old + (t - self.t) * (self.CV - self.CV_old)
        CVL = np.clip(CVL, 0.0, 1.0)
        CVR = np.clip(CVR, 0.0, 1.0)

        """ 计算各外力、外力矩 """
        self.Fgb = self.T_g2b @ np.array([0, 0, mb0 * 9.8]).reshape(3, 1)
        self.Fgp = self.T_g2p @ np.array([0, 0, mp0 * 9.8]).reshape(3, 1)
        self.Mab = np.zeros(3).reshape(3, 1)
        self.Mcb = np.zeros(3).reshape(3, 1)
        self.Mcp = np.zeros(3).reshape(3, 1)
        self.Vb = self.T_g2b @ self.Vcg + self.omigb1 @ self.rcb
        self.Vb_norm = np.linalg.norm(self.Vb)
        self.Fab = -0.5 * rou * self.Vb_norm * Ab * CDb * self.Vb
        self.Vp = self.T_g2p @ self.Vcg + self.omigp1 @ self.rcp
        self.Vp_norm = np.linalg.norm(self.Vp)

        # acoef
        derts = min(CVL, CVR)  # δs对称下偏量(百分比),取左、右下偏量中小的值√√√
        rfa = np.arctan(self.Vp[2] / self.Vp[0]) * 180 / np.pi
        CL_zero_rfa, CL_half_rfa, CL_full_rfa, CD_zero_rfa, CD_half_rfa, CD_full_rfa, Cmy_zero_rfa, Cmy_half_rfa, Cmy_full_rfa = self.coefficients.cal(rfa)

        if derts <= 0.5:
            CL_min = CL_zero_rfa
            CL_max = CL_half_rfa
            CL_rds = CL_min + (CL_max - CL_min) * (derts - 0.0) / (0.5 - 0.0)
            CD_min = CD_zero_rfa
            CD_max = CD_half_rfa
            CD_rds = CD_min + (CD_max - CD_min) * (derts - 0.0) / (0.5 - 0.0)
            Cmy_min = Cmy_zero_rfa
            Cmy_max = Cmy_half_rfa
            Cmy_rds = Cmy_min + (Cmy_max - Cmy_min) * (derts - 0.0) / (0.5 - 0.0)
        else:
            CL_min = CL_half_rfa
            CL_max = CL_full_rfa
            CL_rds = CL_min + (CL_max - CL_min) * (derts - 0.5) / (1.0 - 0.5)
            CD_min = CD_half_rfa
            CD_max = CD_full_rfa
            CD_rds = CD_min + (CD_max - CD_min) * (derts - 0.5) / (1.0 - 0.5)
            Cmy_min = Cmy_half_rfa
            Cmy_max = Cmy_full_rfa
            Cmy_rds = Cmy_min + (Cmy_max - Cmy_min) * (derts - 0.5) / (1.0 - 0.5)

        derta = (CVR - CVL) * 0.24  # δa非对称下偏量,左、右下偏量的差值（为一个百分数）×0.24，转换为非对称下偏量的实际值√√√
        beita = np.arcsin(self.Vp[1] / self.Vp_norm) * 180 / np.pi  # 侧滑角，单位为（度，即°）

        # 升力系数(z)
        CL = CL_rds + CL_derta * abs(derta)  # CL_rds曲线拟合公式求解
        # 阻力系数(x)
        CD = CD_rds + CD_derta * abs(derta)  # CD_rds曲线拟合公式求解
        # y方向气动力系数(y)
        CY = CY_beita * beita + CY_r * self.omiga_p[2][0] * b * 0.5 / self.Vp_norm + CY_derta * derta
        # x方向气动力系数(x)
        CX = (-CD * self.Vp[0] + CL * self.Vp[2]) / self.Vp_norm
        # z方向气动力系数(z)
        CZ = (-CD * self.Vp[2] - CL * self.Vp[0]) / self.Vp_norm

        self.q = 0.5 * rou * self.Vp_norm ** 2  # 气动压力
        Fapx = CX * Ap * self.q  # 机体坐标系下翼伞x方向所受气动力
        Fapy = CY * Ap * self.q  # 机体坐标系下翼伞y方向所受气动力
        Fapz = CZ * Ap * self.q  # 机体坐标系下翼伞z方向所受气动力
        """修改"""
        self.Fap = np.array([Fapx, Fapy, Fapz]).reshape(3, 1)
        # 气动力矩（Map）
        # 滚转力矩系数(Cl,x)
        Cmx = Cmx_beita * beita + Cmx_p * self.omiga_p[0][0] * b * 0.5 / self.Vp_norm + Cmx_r * self.omiga_p[2][
            0] * b * 0.5 / self.Vp_norm + Cmx_derta * derta
        # 俯仰力矩系数(Cm,y)
        Cmy = Cmy_rds + Cmy_q * self.omiga_p[1][0] * c * 0.5 / self.Vp_norm + Cmy_derta * abs(derta)  # Cmy_rds曲线拟合公式求解
        # 偏航力矩系数(Cn,z)
        Cmz = Cmz_beita * beita + Cmz_p * self.omiga_p[0][0] * b * 0.5 / self.Vp_norm + Cmz_r * self.omiga_p[2][
            0] * b * 0.5 / self.Vp_norm + Cmz_derta * derta
        Map_x = Cmx * Ap * b * self.q  # 滚转力矩，乘以b！！！√√？？√√
        # Mapy=Cmy*Ap*c*q   # 俯仰力矩
        Map_y = (Cmy - 0.25 * CZ) * Ap * c * self.q
        # Mapz=(Cmz*b+CY*0.12*c)*Ap*q   #偏航力矩，乘以b！！！√√？？√√
        Map_z = Cmz * Ap * b * self.q
        self.Map = np.array([Map_x, Map_y, Map_z]).reshape(3, 1)  # 机体坐标系下翼伞所受气动力矩
        """ 计算合力、力矩向量 """
        B1 = self.Fab + self.Fgb - self.mb @ self.omigb1 * self.omigb1 @ self.rcb
        B2 = self.Fap + self.Fgp - self.mp @ self.omigp1 * self.omigp1 @ self.rcp
        B3 = self.Mab + self.Mcb - self.omigb1 @ self.Ib @ self.omiga_b
        B4 = self.Map + self.Mcp - self.omigp1 @ self.Ip @ self.omiga_p
        self.B = np.array([B1, B2, B3, B4]).reshape(-1, 1)

        """ 计算总质量、惯量矩阵 """

        self.A_matrix[0:3, 0:3], self.A_matrix[0:3, 3:6], self.A_matrix[0:3, 6:9], self.A_matrix[0:3, 9:12] = self.mb @ self.T_g2b, -self.mb @ self.rcb1, np.zeros([3, 3]), -self.T_g2b
        self.A_matrix[3:6, 0:3], self.A_matrix[3:6, 3:6], self.A_matrix[3:6, 6:9], self.A_matrix[3:6, 9:12] = self.mp @ self.T_g2p, np.zeros([3, 3]), -self.mp @ self.rcp1, self.T_g2p
        self.A_matrix[6:9, 0:3], self.A_matrix[6:9, 3:6], self.A_matrix[6:9, 6:9], self.A_matrix[6:9, 9:12] = np.zeros([3, 3]), self.Ib, np.zeros([3, 3]), self.rcb1 @ self.T_g2b
        self.A_matrix[9:12, 0:3], self.A_matrix[9:12, 3:6], self.A_matrix[9:12, 6:9], self.A_matrix[9:12, 9:12] = np.zeros([3, 3]), np.zeros([3, 3]), self.Ip, -self.rcp1 @ self.T_g2p
        self.acc = np.linalg.inv(self.A_matrix) @ self.B

        """ 计算状态y增量 """
        self.y_dot[0:3], self.y_dot[3:6], self.y_dot[6:9], self.y_dot[9:21] = self.Vcg.reshape(1, -1), self.rate_b.reshape(1, -1), self.rate_p.reshape(1, -1), self.acc.reshape(1, -1)
        return self.y_dot

    def step(self, control_vaults):
        if len(control_vaults) != 2:
            print('bad action')
        self.ctrl(control_vaults)
        self.t += self.t_step
        self.step_count += 1
        self.state = self.ode.integrate(self.t)
        self.storage(self.state)

        reward = 0
        print(self.state)
        return self.state, reward, self.done, None

    def reset(self):
        self.state = np.array([0, 0, 0, 0, -0.0293, 0, 0, 0.2462, 0, 10.623,  0, 3.6301, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return self.state

    def ctrl(self, control_vaults):
        for i, target in enumerate(control_vaults):
            if self.CV[i] < target - self.ctrl_step:
                self.CV[i] += self.ctrl_step
            elif self.CV[i] > target + self.ctrl_step:
                self.CV[i] -= self.ctrl_step
            else:
                self.CV[i] = target

    def render(self, mode='human'):
        if self.step_count % 100 == 0:
            self.display.plot(self.state[0:3])

    def storage(self, data):
        self.states_log.append(data)


class Coefficients(object):
    def __init__(self):
        self.data_CD = np.array([[-20, 0.1, -20, 0.08, -20, 0.05],
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
        self.data_CL = np.array([[-20, 0.3, -20, 0.2, -20, 0.1],
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
        self.data_Cmy = np.array([[-20, -0.01, -20, -0.011, -20, -0.015],
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
        fun_CL, fun_CD, fun_Cmy = self.interpolate()
        self.fun_CL_zero, self.fun_CL_half, self.fun_CL_full = fun_CL
        self.fun_CD_zero, self.fun_CD_half, self.fun_CD_full = fun_CD
        self.fun_Cmy_zero, self.fun_Cmy_half, self.fun_Cmy_full = fun_Cmy

    def interpolate(self):
        CL_zero_rfa0 = self.data_CL[:, 0].reshape(-1)
        CL_zero_zero = self.data_CL[:, 1].reshape(-1)
        CL_half_rfa0 = self.data_CL[:, 2].reshape(-1)
        CL_half_half = self.data_CL[:, 3].reshape(-1)
        CL_full_rfa0 = self.data_CL[:, 4].reshape(-1)
        CL_full_full = self.data_CL[:, 5].reshape(-1)
        fun_CL_zero = interp1d(CL_zero_rfa0, CL_zero_zero, kind='cubic')
        fun_CL_half = interp1d(CL_half_rfa0, CL_half_half, kind='cubic')
        fun_CL_full = interp1d(CL_full_rfa0, CL_full_full, kind='cubic')
        fun_CL = (fun_CL_zero, fun_CL_half, fun_CL_full)

        CD_zero_rfa0 = self.data_CD[:, 0].reshape(-1)
        CD_zero_zero = self.data_CD[:, 1].reshape(-1)
        CD_half_rfa0 = self.data_CD[:, 2].reshape(-1)
        CD_half_half = self.data_CD[:, 3].reshape(-1)
        CD_full_rfa0 = self.data_CD[:, 4].reshape(-1)
        CD_full_full = self.data_CD[:, 5].reshape(-1)
        fun_CD_zero = interp1d(CD_zero_rfa0, CD_zero_zero, kind='cubic')
        fun_CD_half = interp1d(CD_half_rfa0, CD_half_half, kind='cubic')
        fun_CD_full = interp1d(CD_full_rfa0, CD_full_full, kind='cubic')
        fun_CD = (fun_CD_zero, fun_CD_half, fun_CD_full)

        Cmy_zero_rfa0 = self.data_Cmy[:, 0].reshape(-1)
        Cmy_zero_zero = self.data_Cmy[:, 1].reshape(-1)
        Cmy_half_rfa0 = self.data_Cmy[:, 2].reshape(-1)
        Cmy_half_half = self.data_Cmy[:, 3].reshape(-1)
        Cmy_full_rfa0 = self.data_Cmy[:, 4].reshape(-1)
        Cmy_full_full = self.data_Cmy[:, 5].reshape(-1)
        fun_Cmy_zero = interp1d(Cmy_zero_rfa0, Cmy_zero_zero, kind='cubic')
        fun_Cmy_half = interp1d(Cmy_half_rfa0, Cmy_half_half, kind='cubic')
        fun_Cmy_full = interp1d(Cmy_full_rfa0, Cmy_full_full, kind='cubic')
        fun_Cmy = (fun_Cmy_zero, fun_Cmy_half, fun_Cmy_full)

        return fun_CL, fun_CD, fun_Cmy

    def cal(self, rfa):
        """升力系数"""
        CL_zero_rfa = self.fun_CL_zero(rfa)
        CL_half_rfa = self.fun_CL_half(rfa)
        CL_full_rfa = self.fun_CL_full(rfa)
        """阻力系数"""
        CD_zero_rfa = self.fun_CD_zero(rfa)
        CD_half_rfa = self.fun_CD_half(rfa)
        CD_full_rfa = self.fun_CD_full(rfa)
        """力矩系数"""
        Cmy_zero_rfa = self.fun_Cmy_zero(rfa)
        Cmy_half_rfa = self.fun_Cmy_half(rfa)
        Cmy_full_rfa = self.fun_Cmy_full(rfa)
        return CL_zero_rfa, CL_half_rfa, CL_full_rfa, CD_zero_rfa, CD_half_rfa, CD_full_rfa, Cmy_zero_rfa, Cmy_half_rfa, Cmy_full_rfa

class Display:
    def __init__(self):
        self.ax = plt.axes(projection='3d')
        plt.cla()  # 此命令是每次清空画布，所以就不会有前序的效果
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('3d_mobile_obs')
        self.ax.set_xlim([-1500, 1500])
        self.ax.set_ylim([-1500, 1500])
        self.ax.set_zlim([0, -3000])
        plt.grid(True)
        plt.ion()  # interactive mode on!!!! 很重要，有了他就不需要plt.show()了

        self.pos_track = np.array([0.0, 0.0, -3000.0]).reshape((1, 3))

    def plot(self, pos):
        self.pos_track = np.append(self.pos_track, pos.reshape(1, 3), axis=0)
        self.ax.plot3D(self.pos_track[:, 0], self.pos_track[:, 1], self.pos_track[:, 2], 'blue')
        plt.pause(0.0001)


class ParafoilEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.state = np.zeros(21)

        self.done = 0

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        # self.ode = integrate.ode(dynamics.dynamics).set_integrator("none")

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


if __name__ == '__main__':
    env = gym.make('Parafoil-v1')

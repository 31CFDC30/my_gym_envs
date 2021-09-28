# -*- coding: utf-8 -*-
"""
Created on 2021/6/5 10:54

@author: qk
"""
import numpy as np
from Coefficients import cal

import variables as var
import constants as con


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
    sin_phi, sin_theta, sin_psi = np.sin(euler[0][0]), np.sin(euler[1][0]), np.sin(euler[2][0])
    cos_phi, cos_theta, cos_psi = np.cos(euler[0][0]), np.cos(euler[1][0]), np.cos(euler[2][0])
    tan_phi, tan_theta, tan_psi = np.tan(euler[0][0]), np.tan(euler[1][0]), np.tan(euler[2][0])
    matrix = np.array([[cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
                       [sin_theta * sin_phi * cos_psi - cos_phi * sin_psi,
                        sin_theta * sin_phi * sin_psi + cos_phi * cos_psi, sin_phi * cos_theta],
                       [sin_theta * cos_phi * cos_psi + sin_phi * sin_psi,
                        sin_theta * cos_phi * sin_psi - sin_phi * cos_psi, cos_phi * cos_theta]])
    return matrix





def dynamics(t, y):
    mb0, mp0 = 135, 6
    rou, a, b, c, th = 1.225, 1, 7.5, 3.75, 0.6750
    lx, ly, lz = 0.6, 0.5, 0.4
    rcbx, rcby, rcbz, rcpx, rcpy, rcpz = 0, 0, 1.1206, -1.0126, 0, -7.0046
    # pp, qp, rp, ub, vb, wb, up, vp, wp = -0.0228, 0.0239, 3.1175, 2.6348, -9.3567, 4.4098, 10.0214, -0.2099, 3.1175   # 变量
    # derts_left, derts_right = 0.5, 0
    d_left, d_right = 0.1, 0.0
    d_left_t, d_right_t = 200, 200

    Vcg = np.array(y[9:12]).reshape(3, 1)  # 连接点C的速度
    euler_b = np.array(y[3:6]).reshape(3, 1)  # 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）
    euler_p = np.array(y[6:9]).reshape(3, 1)  # 欧拉角分别为phi theta psi（滚转x 俯仰y 偏航z）
    xsjzb = transform_matrix_omega2euler(euler_b)
    xsjzp = transform_matrix_omega2euler(euler_p)
    omiga_b = np.array(y[12:15]).reshape(3, 1)  # 载荷角速度
    rate_b = xsjzb @ omiga_b  # 载荷欧拉角变化率
    omiga_p = np.array(y[15:18]).reshape(3, 1)  # 翼伞角速度
    rate_p = xsjzp @ omiga_p  # 翼伞欧拉角变化率

    omigb1 = anti_symmetric_matrix(omiga_b)
    omigp1 = anti_symmetric_matrix(omiga_p)
    T_g2b = transform_matrix_g2b(euler_b)
    T_g2p = transform_matrix_g2b(euler_p)

    """常数(向量)"""
    """质量"""
    mb = np.identity(3) * mb0  # 载荷总质量
    mp01 = np.identity(3) * mp0

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
                    [0, 0, C]])
    mp = mp01 + mp1  # 翼伞总质量
    """转动惯量"""
    Ibx = mb0 * (ly ** 2 + lz ** 2) / 12
    Iby = mb0 * (lx ** 2 + lz ** 2) / 12
    Ibz = mb0 * (lx ** 2 + ly ** 2) / 12
    Ib = np.array([[Ibx, 0, 0],  # 载荷转动惯量
                   [0, Iby, 0],
                   [0, 0, Ibz]])
    Ipx = mp0 * ((a + th) ** 2 + b ** 2) / 12
    Ipy = mp0 * ((a + th) ** 2 + c ** 2) / 12
    Ipz = mp0 * (b ** 2 + c ** 2) / 12
    Ip0 = np.array([[Ipx, 0, 0],
                    [0, Ipy, 0],
                    [0, 0, Ipz]])
    IA = rou * 0.055 * A1 / (1 + A1) * b * S ** 2
    IB = rou * 0.0308 * A1 / (1 + A1) * (1 + np.pi / 6 * (1 + A1) * A1 * a1 ** 2 * t2 ** 2) * c ** 3 * S
    IC = rou * 0.0555 * (1 + 8 * a1 ** 2) * th ** 2 * b ** 3
    Ip1 = np.array([[IA, 0, 0],
                    [0, IB, 0],
                    [0, 0, IC]])
    Ip = Ip0 + Ip1  # 翼伞转动惯量 = 真是质量转动惯量 + 附加质量转动惯量
    """位置矢量"""
    rcb = np.array([rcbx, rcby, rcbz]).reshape(3, 1)
    rcb1 = anti_symmetric_matrix(rcb)
    rcp = np.array([rcpx, rcpy, rcpz]).reshape(3, 1)
    rcp1 = anti_symmetric_matrix(rcp)
    """外力: 重力Fg、气动力Fa、气动力矩Ma,Mc"""
    Fgb = T_g2b @ np.array([0, 0, mb0 * 9.8]).reshape(3, 1)
    Fgp = T_g2p @ np.array([0, 0, mp0 * 9.8]).reshape(3, 1)
    Mab = np.zeros(3).reshape(3, 1)
    Mcb = np.zeros(3).reshape(3, 1)
    Mcp = np.zeros(3).reshape(3, 1)

    Vb = T_g2b @ Vcg + omigb1 @ rcb
    Vb0 = np.linalg.norm(Vb)

    Ab = 0.5
    CDb = 1.05
    Fab = -0.5 * rou * Vb0 * Ab * CDb * Vb
    Vp = T_g2p @ Vcg + omigp1 @ rcp
    """控制量"""
    # if 50 < t < 500:
    #     ctrl(var.CV_target)
    # else:
    #     ctrl((0.0, 0.0))
    # print('t:{} \tCV:{} {}'.format(t, CVL, CVR))
    # var.CVL = var.CVL_now + (t - var.t_now) * (var.CVL_next - var.CVL_now)
    # var.CVR = var.CVR_now + (t - var.t_now) * (var.CVR_next - var.CVR_now)
    var.CVL, var.CVR = var.CV_now + (t - var.t_now) * (var.CV_next - var.CV_now)
    # print('t: {} \t CV_Target: {} \t CV_now: {} \t CV_next: {} \t CV: {},{}'.format(t, var.CV_target, var.CV_now, var.CV_next, var.CVL, var.CVR))

    # if t < 50:
    #     derts_left = 0
    # elif 50 < t <= 51:
    #     derts_left = d_left * (t - 50) / (51 - 50)
    # elif 51 < t <= 50 + d_left_t:
    #     derts_left = d_left
    # elif 50 + d_left_t < t <= 51 + d_left_t:
    #     derts_left = d_left * (1 - (t - (50 + d_left_t)) / ((51 + d_left_t) - (50 + d_left_t)))
    # else:
    #     derts_left = 0
    #
    # if t < 50:
    #     derts_right = 0
    # elif 50 < t <= 51:
    #     derts_right = d_right * (t - 50) / (51 - 50)
    # elif 51 < t <= 50 + d_right_t:
    #     derts_right = d_right
    # elif 50 + d_right_t < t <= 51 + d_right_t:
    #     derts_right = d_right * (1 - (t - (50 + d_right_t)) / ((51 + d_right_t) - (50 + d_right_t)))
    # else:
    #     derts_right = 0
    """调用气动力方程"""
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

    # acoef
    derts = min(var.CVL, var.CVR)  # δs对称下偏量(百分比),取左、右下偏量中小的值√√√
    # dertm = max(derts_left, derts_right)  # 取左、右下偏量中大的值
    rfa = np.arctan(Vp[2] / Vp[0]) * 180 / np.pi
    CL_zero_rfa, CL_half_rfa, CL_full_rfa, CD_zero_rfa, CD_half_rfa, CD_full_rfa, Cmy_zero_rfa, Cmy_half_rfa, Cmy_full_rfa = cal(
        rfa)
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

    derta = (var.CVR - var.CVL) * 0.24  # δa非对称下偏量,左、右下偏量的差值（为一个百分数）×0.24，转换为非对称下偏量的实际值√√√
    Vp0 = (Vp[0] ** 2 + Vp[1] ** 2 + Vp[2] ** 2) ** 0.5
    beita = np.arcsin(Vp[1] / Vp0) * 180 / np.pi  # 侧滑角，单位为（度，即°）
    # 升力系数(z)
    CL = CL_rds + CL_derta * abs(derta)  # CL_rds曲线拟合公式求解
    # 阻力系数(x)
    CD = CD_rds + CD_derta * abs(derta)  # CD_rds曲线拟合公式求解
    # y方向气动力系数(y)
    CY = CY_beita * beita + CY_r * omiga_p[2][0] * b * 0.5 / Vp0 + CY_derta * derta
    # x方向气动力系数(x)
    CX = (-CD * Vp[0] + CL * Vp[2]) / Vp0
    # z方向气动力系数(z)
    CZ = (-CD * Vp[2] - CL * Vp[0]) / Vp0

    Ap = b * c  # 迎风面积
    q = 0.5 * rou * Vp0 ** 2  # 气动压力

    Fapx = CX * Ap * q  # 机体坐标系下翼伞x方向所受气动力
    Fapy = CY * Ap * q  # 机体坐标系下翼伞y方向所受气动力
    Fapz = CZ * Ap * q  # 机体坐标系下翼伞z方向所受气动力
    """修改"""
    Fap = np.array([Fapx, Fapy, Fapz]).reshape(3, 1)
    # 气动力矩（Map）
    # 滚转力矩系数(Cl,x)
    Cmx = Cmx_beita * beita + Cmx_p * omiga_p[0][0] * b * 0.5 / Vp0 + Cmx_r * omiga_p[2][0] * b * 0.5 / Vp0 + Cmx_derta * derta
    # 俯仰力矩系数(Cm,y)
    Cmy = Cmy_rds + Cmy_q * omiga_p[1][0] * c * 0.5 / Vp0 + Cmy_derta * abs(derta)  # Cmy_rds曲线拟合公式求解
    # 偏航力矩系数(Cn,z)
    Cmz = Cmz_beita * beita + Cmz_p * omiga_p[0][0] * b * 0.5 / Vp0 + Cmz_r * omiga_p[2][0] * b * 0.5 / Vp0 + Cmz_derta * derta
    Mapx = Cmx * Ap * b * q  # 滚转力矩，乘以b！！！√√？？√√
    # Mapy=Cmy*Ap*c*q   # 俯仰力矩
    Mapy = (Cmy - 0.25 * CZ) * Ap * c * q
    # Mapz=(Cmz*b+CY*0.12*c)*Ap*q   #偏航力矩，乘以b！！！√√？？√√
    Mapz = Cmz * Ap * b * q
    Map = np.array([Mapx, Mapy, Mapz]).reshape(3, 1)  # 机体坐标系下翼伞所受气动力矩
    """动力学方程描述"""
    B1 = Fab + Fgb - mb @ omigb1 * omigb1 @ rcb
    B2 = Fap + Fgp - mp @ omigp1 * omigp1 @ rcp
    B3 = Mab + Mcb - omigb1 @ Ib @ omiga_b
    B4 = Map + Mcp - omigp1 @ Ip @ omiga_p
    B = np.array([B1, B2, B3, B4]).reshape(-1, 1)

    A = np.zeros([12, 12])
    A[0:3, 0:3], A[0:3, 3:6], A[0:3, 6:9], A[0:3, 9:12] = mb @ T_g2b, -mb @ rcb1, np.zeros([3, 3]), -T_g2b
    A[3:6, 0:3], A[3:6, 3:6], A[3:6, 6:9], A[3:6, 9:12] = mp @ T_g2p, np.zeros([3, 3]), -mp @ rcp1, T_g2p
    A[6:9, 0:3], A[6:9, 3:6], A[6:9, 6:9], A[6:9, 9:12] = np.zeros([3, 3]), Ib, np.zeros([3, 3]), rcb1 @ T_g2b
    A[9:12, 0:3], A[9:12, 3:6], A[9:12, 6:9], A[9:12, 9:12] = np.zeros([3, 3]), np.zeros([3, 3]), Ip, -rcp1 @ T_g2p

    acc = np.linalg.inv(A) @ B
    # ydot = np.zeros([21, 1])
    # ydot[0:3, :], ydot[3:6, :], ydot[6:9, :], ydot[9:21, :] = Vcg, rate_b, rate_p, acc
    ydot = np.zeros(21)
    ydot[0:3], ydot[3:6], ydot[6:9], ydot[9:21] = Vcg.reshape(1, -1), rate_b.reshape(1, -1), rate_p.reshape(1, -1), acc.reshape(1, -1)

    return ydot


def main():
    # dynamics(0, y0)
    pass


if __name__ == '__main__':
    main()

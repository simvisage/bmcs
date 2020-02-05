'''
Created on 18.01.2016

@author: Yingxiong
'''
import numpy as np
from scipy.optimize import newton

# reinforcement
E_f = 210.  # kN/mm2
d_f = 16.  # mm
A_f = np.pi * d_f ** 2 / 4.0

# matrix
E_m = 354.  # kN/mm2
A_m = 10000.  # mm2

# embedded length
L = 40.  # mm

gamma = 1. / (A_f * E_f) + 1. / (A_m * E_m)

m_arr = np.array(
    [0, 3.49e-5, 1.11e-5, 8.10e-5, 6.60e-5, 1e-8, -6.07e-6, -3.68e-7])
T_arr = np.array(
    [0, 8.72e-6, 1.15e-5, 1.35e-5, 1.52e-5, 1.52e-5, 6.07e-6, 1e-8])
s_arr = np.array([0, 0.25, 0.50, 0.75, 1.00, 3.00, 4.00, 12.0])


def d_x_g0(q, q_1, m, T, T_1):
    # for m>0
    return 1. / np.sqrt(m) * np.log((np.sqrt(m) * q + T) / (np.sqrt(m) * q_1 + T_1))


def d_x_l0(q, q_1, m, T, T_1):
    # for m<0
    return 1. / np.sqrt(-m) * np.arcsin(np.sqrt(-m) * (T * q_1 - T_1 * q) / (T_1 ** 2 - m * q_1 ** 2))


def d_x_bar_g0(q, m, T):
    # for m>0
    return 1. / np.sqrt(m) * np.log((np.sqrt(m) * q + T) / np.sqrt(T ** 2 - m * q ** 2))


def d_x_bar_l0(q, m, T):
    # for m<0
    return 1. / np.sqrt(-m) * np.arcsin(np.sqrt(-m) * np.sqrt(T ** 2 - m * q ** 2) * q / (T ** 2 - m * q ** 2))


def q_1(q, m, s, s_1, T_1):
    return q ** 2 - m * (s - s_1) ** 2 - 2 * T_1 * (s - s_1)

p_f = []

for i in np.arange(1, 8):

    if i == 1:
        if m_arr[1] > 0:
            d_x_bar = lambda q: d_x_bar_g0(q, m_arr[1], T_arr[1]) - L
            p_f.append(newton(d_x_bar, 0))
        else:
            d_x_bar = lambda q: d_x_bar_l0(q, m_arr[1], T_arr[1]) - L
            p_f.append(newton(d_x_bar, 0))
        print(p_f)
    else:
        def e_L(q_i):
            j = i
            l = 0.
            while True:
                q_i_1 = q_1(
                    q_i, m_arr[j], s_arr[j], s_arr[j - 1], T_arr[j - 1])
                if q_i_1 > 0:
                    if m_arr[j] > 0:
                        d_x = d_x_g0(
                            q_i, q_i_1, m_arr[j], T_arr[j], T_arr[j - 1])
                    else:
                        d_x = d_x_l0(
                            q_i, q_i_1, m_arr[j], T_arr[j], T_arr[j - 1])
                    l += d_x
                    q_i = q_i_1
                    j = j - 1
                else:
                    break

            if m_arr[j] > 0:
                d_x_bar = d_x_bar_g0(q_i, m_arr[j], T_arr[j])
            else:
                d_x_bar = d_x_bar_l0(q_i, m_arr[j], T_arr[j])
            l += d_x_bar
            return l

        solve = lambda q_i: e_L(q_i) - L

        p_f.append(newton(solve, 0., maxiter=5000000))
        print([np.array(p_f) / gamma])

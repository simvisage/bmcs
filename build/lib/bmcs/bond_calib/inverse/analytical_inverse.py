import numpy as np
from scipy.optimize import newton, brentq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


# reinforcement
E_f = 170000.  # N/mm2
# d_f = 16.  # mm
# A_f = np.pi * d_f ** 2 / 4.0
A_f = 9 * 1.85  # mm2

# matrix
E_m = 28484.  # N/mm2
A_m = 100 * 8 - 9 * 1.85  # mm2

# embedded length
L = 150.  # mm
gamma = 1. / (A_f * E_f) + 1. / (A_m * E_m)


# p_arr = np.array([0.0,  1.42888176e+01,   1.90808270e+01,   2.16062547e+01,
#                   2.45121859e+01,   2.53688656e+01,   1.01638353e+01,
#                   1.66934090e-02])
#
# w_arr = np.array([0, 0.25, 0.50, 0.75, 1.00, 3.00, 4.00, 12.0])

# w_arr, p_arr = np.loadtxt('D:\\Default_Dataset.csv', delimiter=',').T
fpath = 'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V3_R3_f.asc'

x, y = np.loadtxt(fpath,  delimiter=';')

x[0] = 0.

interp = interp1d(x / 2., y)

w_arr = np.linspace(0, 10., 200)

p_arr = interp(w_arr)

x = np.linspace(0, 10., 200)

interp = interp1d(w_arr, p_arr)

y = interp(x) * 1000.

q_arr = y * gamma


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

T = np.zeros_like(x)

for i, psi in enumerate(x):
    if i == 0:
        T[i] = 0
    elif i == 1:
        def d_x_bar(Ti):
            m = (Ti - T[i - 1]) / (x[i] - x[i - 1])
            if m > 0:
                return d_x_bar_g0(q_arr[i], m, Ti)
            else:
                return d_x_bar_l0(q_arr[i], m, Ti)
        a = lambda t: d_x_bar(t) - L
        T[i] = brentq(a, 0., 10.)
    else:
        m_arr = (T[1:i + 1] - T[0:i]) / (x[1:i + 1] - x[0:i])

        def d_x(Ti):
            T[i] = Ti
            m_arr[-1] = (Ti - T[i - 1]) / (x[i] - x[i - 1])
            j = i
            l = 0.
            q_i = q_arr[i]
            while True:
                q_i_1 = q_1(
                    q_i, m_arr[j - 1], x[j], x[j - 1], T[j - 1])
                if q_i_1 > 0:
                    if m_arr[j - 1] > 0:
                        d_x = d_x_g0(
                            q_i, q_i_1, m_arr[j - 1], T[j], T[j - 1])
                    else:
                        d_x = d_x_l0(
                            q_i, q_i_1, m_arr[j - 1], T[j], T[j - 1])
                    l += d_x
                    q_i = q_i_1
                    j = j - 1
                else:
                    break

            if m_arr[j - 1] > 0:
                d_x_bar = d_x_bar_g0(q_i, m_arr[j - 1], T[j])
            else:
                d_x_bar = d_x_bar_l0(q_i, m_arr[j - 1], T[j])
            l += d_x_bar
            return l

        solve = lambda Ti: d_x(Ti) - L
        T[i] = brentq(solve,  1e-9, 10.)
#         print T

plt.plot(x, T / gamma)
# T_arr = np.array(
#     [0, 8.72e-6, 1.15e-5, 1.35e-5, 1.52e-5, 1.52e-5, 6.07e-6, 1e-8])
# s_arr = np.array([0, 0.25, 0.50, 0.75, 1.00, 3.00, 4.00, 12.0])
# plt.plot(s_arr, T_arr)
plt.show()

'''
Created on Feb 10, 2020

@author: rch
'''

import numpy as np
import sympy as sp

phi, t_1, t_2, n_1, n_2, R_1, R_2 = sp.symbols(
    'phi, t_1, t_2, n_1, n_2, R_1, R_2')
eta = sp.symbols('eta')
P_1, P_2 = sp.symbols('P_1, P_2')
w = sp.Symbol('w')

R_vec = sp.Matrix([[R_1, R_2]]).T
P_vec = sp.Matrix([[P_1, P_2]]).T
t_vec = sp.Matrix([[t_1, t_2]]).T
n_vec = sp.Matrix([[n_1, n_2]]).T

Q_vec = P_vec + w * n_vec + eta * t_vec

RP_2 = (P_vec - R_vec).T * (P_vec - R_vec)

RQ_2 = (Q_vec - R_vec).T * (Q_vec - R_vec)

eq_eta = sp.simplify(RP_2 - RQ_2)[0]
eq_eta_simplified = sp.simplify(sp.expand(eq_eta))
eta_sol = sp.solve(eq_eta_simplified, eta)[0]

get_eta = sp.lambdify(
    (n_1, n_2, t_1, t_2, P_1, P_2, R_1, R_2, w),
    eta_sol
)

get_Q_vec = sp.lambdify(
    (eta, n_1, n_2, t_1, t_2, P_1, P_2, R_1, R_2, w),
    Q_vec
)

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


def get_phi(T_ab, x_fps_a, x_rot_a, w_f_t):
    T_args = tuple(T_ab.flatten())
    args = T_args + (x_fps_a[0], x_fps_a[1], x_rot_a[0], x_rot_a[1], w_f_t)

    eta_val = get_eta(*args)

    Q_args = (eta_val,) + args
    Q_val = get_Q_vec(*Q_args)

    v1 = x_fps_a - x_rot_a
    v2 = Q_val[:, 0] - x_rot_a

    cross_v1_v2 = np.einsum('ijk,j,k->...i',
                            EPS[:, :-1, :-1], v1, v2)[2]
    dot_v1_v2 = np.einsum('i,i',
                          v1, v2)
    v1_norm = np.sqrt(np.einsum('k,k', v1, v1))
    v2_norm = np.sqrt(np.einsum('k,k', v2, v2))
    cos_phi = dot_v1_v2 / (v1_norm * v2_norm)
    phi = np.arccos(cos_phi)
    return phi


if __name__ == '__main__':
    w_f_t = 0.03  # np.sqrt(2) / 2
    x_rot_a = np.array([400, 10], dtype=np.float_)
    x_fps_a = np.array([400, 0], dtype=np.float_)
    T_ab = np.array([[1.0, 0.0],  # normal vector
                     [0.0, 1.0]])  # line vector
    # T_ab = np.array([[0.0, 1.0],
    #                  [-1.0, 0.0]])

    phi = get_phi(T_ab, x_fps_a, x_rot_a, w_f_t)
    print(phi)

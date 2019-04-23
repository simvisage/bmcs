'''
Created on 01.03.2019

@author: Mario Aguilar Rueda
'''

import os

from traits.api import  \
    Float, List
import numpy as np
from .mats_bondslip import MATSBondSlipBase


class MATS1DDesmorat(MATSBondSlipBase):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

    tree_node_list = List([])

    E_b = Float(19000,
                label="E_b",
                MAT=True,
                symbol=r'E_\mathrm{b}',
                unit='MPa/mm',
                desc='elastic bond stiffness',
                enter_set=True,
                auto_set=False)

    E_m = Float(16000,
                label="E_m",
                MAT=True,
                symbol=r'E_\mathrm{m}',
                unit='MPa/mm',
                desc='matrix elastic stiffness',
                enter_set=True,
                auto_set=False)

    gamma = Float(110,
                  label="Gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(130,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    S = Float(476e-6,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(75,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    sv_names = ['tau_e',
                'tau',
                's_p',
                'z',
                'X',
                'D',
                'Y',
                's_cum'
                ]

    def get_next_state(self, s, s_vars):

        tau_e, tau, s_p, z, X, D, Y, s_cum, = s_vars
        v = len(s) - 1
        for i in range(v):

            # trial stress - assuming elastic increment.
            tau_trial = (self.E_m + self.E_b) * \
                (1. - D[i]) * s[i + 1] - \
                self.E_b * (1. - D[i]) * s_p[i]
            tau_e_trial = self.E_b * (s[i + 1] - s_p[i])
            f_trial = np.abs(tau_e_trial - X[i]) - self.tau_bar - self.K * z[i]

            if f_trial <= 0:
                tau_e[i + 1] = tau_e_trial
                tau[i + 1] = tau_trial
                s_p[i + 1] = s_p[i]
                z[i + 1] = z[i]
                X[i + 1] = X[i]
                D[i + 1] = D[i]
                Y[i + 1] = Y[i]
                s_cum[i + 1] = s_cum[i]

            # identify values beyond the elastic limit
            else:

                # plastic multiplier
                delta_pi = f_trial / \
                    (self.E_b + (self.K + self.gamma) * (1. - D[i]))

                # return mapping for isotropic and kinematic hardening
                grad_f = np.sign(tau_e_trial - X[i])
                s_p[i + 1] = s_p[i] + delta_pi * grad_f
                s_cum[i + 1] = s_cum[i] + delta_pi
                Y[i + 1] = 0.5 * self.E_m * s[i + 1]**2. + 0.5 * \
                    self.E_b * (s[i + 1] - s_p[i + 1])**2.
                D_trial = D[i] + (Y[i + 1] / self.S) * \
                    delta_pi * (1. - D[i])
                if D_trial > 1.0:
                    D[i + 1] = 1.0
                else:
                    D[i + 1] = D_trial
                z[i + 1] = z[i] + (1. - D[i + 1]) * delta_pi
                X[i + 1] = X[i] + self.gamma * \
                    (1. - D[i + 1]) * (s_p[i + 1] - s_p[i])

                # apply damage law to the effective stress
                tau_e[i + 1] = self.E_b * \
                    (1. - D[i + 1]) * (s[i + 1] - s_p[i + 1])
                tau[i + 1] = (self.E_m + self.E_b) * \
                    (1. - D[i + 1]) * s[i + 1] - \
                    self.E_b * (1. - D[i + 1]) * s_p[i + 1]

        return s_p, Y, D, z, X, tau_e, tau, s_cum


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = MATS1DDesmorat()

    # Monotonic
    s_levels = np.linspace(0, 0.05, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= 1
    s_history = s_levels.flatten()
    s = np.hstack([np.linspace(s_history[i], s_history[i + 1], 200)
                   for i in range(len(s_levels) - 1)])

    s_vars = [np.zeros_like(s)
              for sv in m.sv_names]
    s_p, Y, D, z, X, tau_e, tau, s_cum = m.get_next_state(s, s_vars)
    plt.plot(s, tau)

    home_dir = os.path.expanduser('~')
    path = os.path.join(home_dir, 'Uniaxial Desmorat')
    path = os.path.join(path, 'Parametric reversibility limit')
    path = os.path.join(path, 'strain driven')
    if os.path.exists(path) == False:
        os.makedirs(path)

    np.save(os.path.join(path, 's.npy'), s)
    np.save(os.path.join(path, 'tau75.npy'), tau)

    #plt.plot(eps_1, sig_arr_1)
#     # Fatigue
#
#     sigma_max = f_max1
#     print(sigma_max)
#     sigma_max_min = 0.
#     cycles = 800
#     inc = 100
#     points = 100
#     N_log = np.zeros(points)
#     s_max_1 = np.zeros(points)
#     for j in range(points):
#         m = MATS1DDesmorat()
#         s_max_1[j] = sigma_max - (sigma_max - sigma_max_min) * j / points
#         s_levels_2 = np.linspace(0, 10, cycles * 2)
#         s_levels_2.reshape(-1, 2)[:, 0] = s_max_1[j]
#         s_levels_2.reshape(-1, 2)[:, 1] = 0.0
#         s_levels_2[0] = 0.0
#         s_history_2 = s_levels_2.flatten()
#
#         sig_arr_2 = np.zeros(1)
#         for i in range(len(s_levels_2) - 1):
#             sig_part = np.linspace(s_history_2[i], s_history_2[i + 1], inc)
#             sig_arr_2 = np.hstack((sig_arr_2, sig_part[:-1]))
#
#         sigma_pi = np.zeros_like(sig_arr_2)
#         eps = np.zeros_like(sig_arr_2)
#         eps_pi = np.zeros_like(sig_arr_2)
#         gamma = np.zeros_like(sig_arr_2)
#         r = np.zeros_like(sig_arr_2)
#         omega = np.zeros_like(sig_arr_2)
#         eps_cum = np.zeros_like(sig_arr_2)
#         Y = np.zeros_like(sig_arr_2)
#         n = np.zeros(1)
#         f_max2 = np.zeros(1)
#         flag = 0
#         sig_max2 = s_max_1[j]
#
#         sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max2, n, sig_max2, flag = m.get_next_state(
#             sig_arr_2, sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max2, n, sig_max2, flag)
#         if flag == 0:
#             break
#         N_log[j] = n
#         s_max_1[j] = s_max_1[j] / f_max1
#         if N_log[j] == N_log[j - 1]:
#             s_max_1[j] = s_max_1[j - 1]
# #     #plt.semilogx(N_log, s_max_1)
# #     #plt.plot(eps, sig_arr_2)
#
#     s_max_1 = s_max_1[np.nonzero(N_log)]
#     N_log = N_log[np.nonzero(N_log)]
#
#     home_dir = os.path.expanduser('~')
#     path = os.path.join(home_dir, 'Uniaxial Desmorat')
#     path = os.path.join(path, 'Parametric isotropic')
#     #path = os.path.join(path, 'S-N curves')
#     if os.path.exists(path) == False:
#         os.makedirs(path)
#
#     np.save(os.path.join(path, 'N_S_1e2.npy'), N_log)
#     np.save(os.path.join(path, 'S_max_1e2.npy'), s_max_1)
#     plt.subplot(111)
#     axes = plt.gca()
#     axes.set_ylim([0, 1.2])
#     plt.semilogx(N_log, s_max_1)
#
#
# #     np.save(os.path.join(path, 'sigma_f.npy'), sig_arr_2)
# #     np.save(os.path.join(path, 'eps_f.npy'), eps)
# #     np.save(os.path.join(path, 'eps_cum_f.npy'), eps_cum)
# #     np.save(os.path.join(path, 'omega_f.npy'), omega)
# #
#     np.save(os.path.join(path, 'sigma_m_1e2.npy'), sig_arr_1)
#     np.save(os.path.join(path, 'eps_m_1e2.npy'), eps_1)
#     np.save(os.path.join(path, 'fmax_1e2.npy'), f_max1)

    plt.show()
# m.configure_traits()

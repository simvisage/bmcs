'''
Created on 07.12.2018

@author: Mario Aguilar Rueda
'''

import os

from traits.api import \
    Float, List

import numpy as np

from .mats_bondslip import MATSBondSlipBase


class MATS1DDesmorat(MATSBondSlipBase):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

    tree_node_list = List([])

    E2 = Float(19000,
               label="E_b",
               MAT=True,
               symbol=r'E_\mathrm{b}',
               unit='MPa/mm',
               desc='elastic bond stiffness',
               enter_set=True,
               auto_set=False)

    E1 = Float(16000,
               label="E_m",
               MAT=True,
               symbol=r'E_\mathrm{m}',
               unit='MPa/mm',
               desc='matrix elastic stiffness',
               enter_set=True,
               auto_set=False)

    C = Float(110,
              label="Gamma",
              desc="kinematic hardening modulus",
              MAT=True,
              symbol=r'\gamma',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    K = Float(1e2,
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

    sigma_0 = Float(6,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    sv_names = ['sigma_pi',
                'eps',
                'eps_pi',
                'gamma',
                'r',
                'omega',
                'eps_cum',
                'Y'
                ]

    def get_next_state(self, sigma, sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max, n, sig_max, flag):

        for i in range(1, len(sigma)):

            # trial stress - assuming elastic increment.

            sigma_pi_trial = (self.E2 * sigma[i] / (self.E1 + self.E2)) - (
                (self.E1 * self.E2 * (1. - omega[i - 1])) * eps_pi[i - 1] / (self.E1 + self.E2))

            sigma2_pi_trial = sigma_pi_trial / (1. - omega[i - 1])

            eps_trial = (sigma[i] - sigma_pi_trial) / \
                ((1 - omega[i - 1]) * self.E1)

            f_trial = np.fabs(sigma2_pi_trial - self.C *
                              gamma[i - 1]) - self.sigma_0 - self.K * r[i - 1]

            if f_trial <= 0:
                eps[i] = eps_trial
                eps_pi[i] = eps_pi[i - 1]
                r[i] = r[i - 1]
                omega[i] = omega[i - 1]
                gamma[i] = gamma[i - 1]
                eps_cum[i] = eps_cum[i - 1]
                if sigma[i] == sig_max:
                    n = n + 1

            # identify values beyond the elastic limit
            else:

                if sigma[i] == sig_max:
                    n = n + 1

                # Return mapping and updating the state variables

                delta_pi = f_trial / \
                    ((self.E2 / (1. - omega[i - 1])) + self.C + self.K)

                eps_pi[i] = eps_pi[i - 1] + delta_pi * \
                    np.sign(sigma2_pi_trial - self.C *
                            gamma[i - 1]) / (1. - omega[i - 1])

                eps_cum[i] = eps_cum[i - 1] + delta_pi / (1. - omega[i - 1])

                sigma_pi[i] = (self.E2 * sigma[i] / (self.E1 + self.E2)) - (
                    ((self.E1 * self.E2) * (1. - omega[i - 1])) * eps_pi[i] / (self.E1 + self.E2))

                Y[i] = 0.5 * ((sigma[i] - sigma_pi_trial)**2.0 / (self.E1 * (
                    1. - omega[i - 1])**2.0)) + 0.5 * ((sigma_pi_trial)**2.0 / (self.E1 * (1. - omega[i - 1])**2))

                delta_lambda = delta_pi * (1. - omega[i - 1])

                omega[i] = omega[i - 1] + \
                    delta_lambda * (1. - omega[i - 1]
                                    )**(-1.) * (Y[i] / self.S)

                if omega[i] > 0.9999999:
                    f_max = sigma[i - 1]
                    flag = 1
                    break
                else:

                    eps[i] = (sigma[i] - sigma_pi[i]) / \
                        ((1 - omega[i]) * self.E1)

                    r[i] = r[i - 1] + delta_pi

                    gamma[i] = gamma[i - 1] + \
                        delta_pi * np.sign(sigma2_pi_trial -
                                           self.C * gamma[i - 1])

        return sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max, n, sig_max, flag


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = MATS1DDesmorat()

    # Monotonic
    s_levels = np.linspace(0, 200, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= 1
    s_history = s_levels.flatten()
    sig_arr_1 = np.hstack([np.linspace(s_history[i], s_history[i + 1], 200)
                           for i in range(len(s_levels) - 1)])

    sigma_pi_1 = np.zeros_like(sig_arr_1)
    eps_1 = np.zeros_like(sig_arr_1)
    eps_pi_1 = np.zeros_like(sig_arr_1)
    gamma_1 = np.zeros_like(sig_arr_1)
    r_1 = np.zeros_like(sig_arr_1)
    omega_1 = np.zeros_like(sig_arr_1)
    eps_cum_1 = np.zeros_like(sig_arr_1)
    Y_1 = np.zeros_like(sig_arr_1)
    n_1 = np.zeros(1)
    f_max1 = np.zeros(1)
    sig_max2 = 100
    flag = 0

    sigma_pi_1, eps_1, eps_pi_1, gamma_1, r_1, omega_1, eps_cum_1, Y_1, f_max1, n_1, sig_max2, flag = m.get_next_state(
        sig_arr_1, sigma_pi_1, eps_1, eps_pi_1, gamma_1, r_1, omega_1, eps_cum_1, Y_1, f_max1, n_1, sig_max2, flag)

    sigma_pi_1 = sigma_pi_1[1:np.amax(np.nonzero(omega_1))]
    eps_1 = eps_1[1:np.amax(np.nonzero(omega_1))]
    eps_pi_1 = eps_pi_1[1:np.amax(np.nonzero(omega_1))]
    gamma_1 = gamma_1[1:np.amax(np.nonzero(omega_1))]
    r_1 = r_1[1:np.amax(np.nonzero(omega_1))]
    eps_cum_1 = eps_cum_1[1:np.amax(np.nonzero(omega_1))]
    Y_1 = Y_1[1:np.amax(np.nonzero(omega_1))]
    sig_arr_1 = sig_arr_1[1:np.amax(np.nonzero(omega_1))]
    omega_1 = omega_1[1:np.amax(np.nonzero(omega_1))]
    #plt.plot(eps_1, sig_arr_1)
    # Fatigue

    sigma_max = f_max1
    print(sigma_max)
    sigma_max_min = 0.
    cycles = 800
    inc = 100
    points = 100
    N_log = np.zeros(points)
    s_max_1 = np.zeros(points)
    for j in range(points):
        m = MATS1DDesmorat()
        s_max_1[j] = sigma_max - (sigma_max - sigma_max_min) * j / points
        s_levels_2 = np.linspace(0, 10, cycles * 2)
        s_levels_2.reshape(-1, 2)[:, 0] = s_max_1[j]
        s_levels_2.reshape(-1, 2)[:, 1] = 0.0
        s_levels_2[0] = 0.0
        s_history_2 = s_levels_2.flatten()

        sig_arr_2 = np.zeros(1)
        for i in range(len(s_levels_2) - 1):
            sig_part = np.linspace(s_history_2[i], s_history_2[i + 1], inc)
            sig_arr_2 = np.hstack((sig_arr_2, sig_part[:-1]))

        sigma_pi = np.zeros_like(sig_arr_2)
        eps = np.zeros_like(sig_arr_2)
        eps_pi = np.zeros_like(sig_arr_2)
        gamma = np.zeros_like(sig_arr_2)
        r = np.zeros_like(sig_arr_2)
        omega = np.zeros_like(sig_arr_2)
        eps_cum = np.zeros_like(sig_arr_2)
        Y = np.zeros_like(sig_arr_2)
        n = np.zeros(1)
        f_max2 = np.zeros(1)
        flag = 0
        sig_max2 = s_max_1[j]

        sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max2, n, sig_max2, flag = m.get_next_state(
            sig_arr_2, sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y, f_max2, n, sig_max2, flag)
        if flag == 0:
            break
        N_log[j] = n
        s_max_1[j] = s_max_1[j] / f_max1
        if N_log[j] == N_log[j - 1]:
            s_max_1[j] = s_max_1[j - 1]
#     #plt.semilogx(N_log, s_max_1)
#     #plt.plot(eps, sig_arr_2)

    s_max_1 = s_max_1[np.nonzero(N_log)]
    N_log = N_log[np.nonzero(N_log)]

    home_dir = os.path.expanduser('~')
    path = os.path.join(home_dir, 'Uniaxial Desmorat')
    path = os.path.join(path, 'Parametric isotropic')
    #path = os.path.join(path, 'S-N curves')
    if os.path.exists(path) == False:
        os.makedirs(path)

    np.save(os.path.join(path, 'N_S_1e2.npy'), N_log)
    np.save(os.path.join(path, 'S_max_1e2.npy'), s_max_1)
    plt.subplot(111)
    axes = plt.gca()
    axes.set_ylim([0, 1.2])
    plt.semilogx(N_log, s_max_1)


#     np.save(os.path.join(path, 'sigma_f.npy'), sig_arr_2)
#     np.save(os.path.join(path, 'eps_f.npy'), eps)
#     np.save(os.path.join(path, 'eps_cum_f.npy'), eps_cum)
#     np.save(os.path.join(path, 'omega_f.npy'), omega)
#
    np.save(os.path.join(path, 'sigma_m_1e2.npy'), sig_arr_1)
    np.save(os.path.join(path, 'eps_m_1e2.npy'), eps_1)
    np.save(os.path.join(path, 'fmax_1e2.npy'), f_max1)

    plt.show()
# m.configure_traits()

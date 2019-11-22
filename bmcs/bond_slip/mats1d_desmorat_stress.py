'''
Created on 07.12.2018

@author: Mario Aguilar Rueda
'''

from traits.api import \
    Float, List
from traitsui.api import View, VGroup, Item, Group

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

    K = Float(130,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    S = Float(0.000476,
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

    def get_next_state(self, sigma, sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y):

        v = len(sigma) - 1

        for i in range(1, len(sigma)):

            # trial stress - assuming elastic increment.

            sigma_pi_trial = (self.E2 * sigma[i] / (self.E1 + self.E2)) - (
                (self.E1 * self.E2 * (1. - omega[i - 1])) * eps_pi[i - 1] / (self.E1 + self.E2))

            sigma_pi_trial = (self.E2 / (self.E1 + self.E2)) * sigma[i] - (
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

            # identify values beyond the elastic limit
            else:

                # Return mapping and updating the state variables

                delta_pi = f_trial / \
                    (self.E2 + (1. - omega[i - 1]) * (self.C + self.K))

                eps_pi[i] = eps_pi[i - 1] + delta_pi * \
                    np.sign(sigma2_pi_trial - self.C * gamma[i - 1])

                eps_cum[i] = eps_cum[i - 1] + delta_pi

                sigma_pi[i] = (self.E2 * sigma[i] / (self.E1 + self.E2)) - (
                    ((self.E1 * self.E2) * (1 - omega[i - 1])) * eps_pi[i] / (self.E1 + self.E2))

                Y[i] = 0.5 * ((sigma[i] - sigma_pi_trial)**2.0 / (self.E1 * (
                    1 - omega[i - 1])**2.0)) + 0.5 * ((sigma_pi_trial)**2.0 / (self.E1 * (1 - omega[i - 1])**2))

                delta_lambda = delta_pi * (1. - omega[i - 1])

                omega[i] = omega[i - 1] + \
                    delta_lambda * (1 - omega[i - 1]
                                    )**(-1) * (Y[i] / self.S)

                if omega[i] > 1.0:
                    break
                else:

                    eps[i] = (sigma[i] - sigma_pi[i]) / \
                        ((1 - omega[i]) * self.E1)

                    r[i] = r[i - 1] + delta_pi

                    gamma[i] = gamma[i - 1] + \
                        delta_pi * np.sign(sigma2_pi_trial -
                                           self.C * gamma[i - 1])

        return sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y

    traits_view = View(
        Group(
            VGroup(
                Item('E_b', full_size=True, resizable=True),
                Item('E_m'),
                Item('gamma'),
                Item('K'),
                Item('S'),
                Item('tau_bar'),
            ),

        ),
        width=0.4,
        height=0.8,
    )

    tree_view = traits_view


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = MATS1DDesmorat()

    s_levels_1 = np.linspace(0, -40, 10)
    s_levels_1.reshape(-1, 2)[:, 0] = -10
    s_levels_1.reshape(-1, 2)[:, 1] = -40
    s_levels_1[0] = 0
    s_history_1 = s_levels_1.flatten()
    sigma = np.hstack([np.linspace(s_history_1[i], s_history_1[i + 1], 10, dtype=np.float_)
                       for i in range(len(s_levels_1) - 1)])
#     s_levels_2 = np.linspace(0, -46, 4)
#     s_levels_2.reshape(-1, 2)[:, 0] = -5
#     s_levels_2.reshape(-1, 2)[:, 1] = -46
#     s_levels_2[0] = -48
#     s_history_2 = s_levels_2.flatten()
#     s_2 = np.hstack([np.linspace(s_history_2[i], s_history_2[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_2) - 1)])
#
#     s_levels_3 = np.linspace(0, -44, 4)
#     s_levels_3.reshape(-1, 2)[:, 0] = -5
#     s_levels_3.reshape(-1, 2)[:, 1] = -44
#     s_levels_3[0] = -46
#     s_history_3 = s_levels_3.flatten()
#     s_3 = np.hstack([np.linspace(s_history_3[i], s_history_3[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_3) - 1)])
#
#     s_levels_4 = np.linspace(0, -42, 4)
#     s_levels_4.reshape(-1, 2)[:, 0] = -5
#     s_levels_4.reshape(-1, 2)[:, 1] = -42
#     s_levels_4[0] = -44
#     s_history_4 = s_levels_4.flatten()
#     s_4 = np.hstack([np.linspace(s_history_4[i], s_history_4[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_4) - 1)])
#
#     sigma = np.hstack((s_1, s_2, s_3, s_4))

    sigma_pi = np.zeros_like(sigma)
    eps = np.zeros_like(sigma)
    eps_pi = np.zeros_like(sigma)
    gamma = np.zeros_like(sigma)
    r = np.zeros_like(sigma)
    omega = np.zeros_like(sigma)
    eps_cum = np.zeros_like(sigma)
    Y = np.zeros_like(sigma)

    sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y = m.get_next_state(
        sigma, sigma_pi, eps, eps_pi, gamma, r, omega, eps_cum, Y)
    plt.plot(eps, sigma)

    plt.show()
#   m.configure_traits()

'''
Created on 15.10.2018

@author: aguilar
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

    gamma = Float(3e7,
                  label="Gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(1e7,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    S = Float(1e-7,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(1,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)


#=========================================================================
#     omega_fn_type = Trait('li',
#                           dict(li=LiDamageFn,
#                                jirasek=JirasekDamageFn,
#                                abaqus=AbaqusDamageFn,
#                                FRP=FRPDamageFn,
#                                ),
#                           MAT=True,
#                           )
#
#     @on_trait_change('omega_fn_type')
#     def _reset_omega_fn(self):
#         #print 'resetting damage function to', self.omega_fn_type
#         #self.omega_fn = self.omega_fn_type_()
#
#     omega_fn = Instance(IDamageFn,
#                         report=True)
#
#     def _omega_fn_default(self):
#         # return JirasekDamageFn()
#         return LiDamageFn(alpha_1=1.,
#                           alpha_2=100.
#                           )
#=========================================================================

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

#     def get_corr_pred(self, s, d_s, tau, t_n, t_n1, s_vars):
#
#         s_p, alpha, z, kappa, omega = s_vars
#
#         n_e, n_ip, n_s = s.shape
#         D = np.zeros((n_e, n_ip, n_s, n_s))
#         D[:, :, 0, 0] = self.E_m
#         D[:, :, 2, 2] = self.E_f
#
#         sig_pi_trial = self.E_b * (s[:, :, 1] - s_p)
#
#         Z = self.K * z
#         X = self.gamma * alpha
#         f = np.fabs(sig_pi_trial - X) - self.tau_bar - Z
#
#         elas = f <= 1e-6
#         plas = f > 1e-6
#
#         d_tau = np.einsum('...st,...t->...s', D, d_s)
#         tau += d_tau
#
#         # Return mapping
#         delta_lamda = f / (self.E_b + self.gamma + self.K) * plas
#         # update all the state variables
#
#         s_p = s_p + delta_lamda * np.sign(sig_pi_trial - X)
#         z = z + delta_lamda
#         alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)
#
#         kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)
#         omega = self.g_fn(kappa)
#
#         tau[:, :, 1] = (1 - omega) * self.E_b * (s[:, :, 1] - s_p)
#
#         # Consistent tangent operator
#
#         g_fn = self.g_fn_get_function()
#         D_ed = -self.E_b / (self.E_b + self.K + self.gamma) * derivative(g_fn, kappa, dx=1e-6) * self.E_b * (s[:, :, 1] - s_p) \
#             + (1 - omega) * self.E_b * (self.K + self.gamma) / \
#             (self.E_b + self.K_bar + self.H_bar)
#
#         D[:, :, 1, 1] = (1 - omega) * self.E_b * elas + D_ed * plas
#
#         return tau, D, s_p, alpha, z, kappa, omega


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = MATS1DDesmorat()
    #############################################################
    s_levels_1 = np.linspace(0, -0.0022, 2)
    s_levels_1.reshape(-1, 2)[:, 0] = 0.0
    s_levels_1.reshape(-1, 2)[:, 1] = -0.0022
    s_levels_1[0] = 0
    s_history_1 = s_levels_1.flatten()
    s = np.hstack([np.linspace(s_history_1[i], s_history_1[i + 1], 50, dtype=np.float_)
                   for i in range(len(s_levels_1) - 1)])
    ############################################################

#     s_levels_1 = np.linspace(0, -0.0022, 8)
#     s_levels_1.reshape(-1, 2)[:, 0] = -0.00075
#     s_levels_1.reshape(-1, 2)[:, 1] = -0.0022
#     s_levels_1[0] = 0
#     s_history_1 = s_levels_1.flatten()
#     s = np.hstack([np.linspace(s_history_1[i], s_history_1[i + 1], 50, dtype=np.float_)
#                    for i in range(len(s_levels_1) - 1)])

#     s_levels_2 = np.linspace(0, -0.0026, 8)
#     s_levels_2.reshape(-1, 2)[:, 0] = -0.0010
#     s_levels_2.reshape(-1, 2)[:, 1] = -0.0026
#     s_levels_2[0] = -0.0022
#     s_history_2 = s_levels_2.flatten()
#     s_2 = np.hstack([np.linspace(s_history_2[i], s_history_2[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_2) - 1)])
#
#     s_levels_3 = np.linspace(0, -0.0028, 8)
#     s_levels_3.reshape(-1, 2)[:, 0] = -0.0013
#     s_levels_3.reshape(-1, 2)[:, 1] = -0.0028
#     s_levels_3[0] = -0.0026
#     s_history_3 = s_levels_3.flatten()
#     s_3 = np.hstack([np.linspace(s_history_3[i], s_history_3[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_3) - 1)])
#
#     s_levels_4 = np.linspace(0, -0.003, 8)
#     s_levels_4.reshape(-1, 2)[:, 0] = -0.0015
#     s_levels_4.reshape(-1, 2)[:, 1] = -0.003
#     s_levels_4[0] = -0.0028
#     s_history_4 = s_levels_4.flatten()
#     s_4 = np.hstack([np.linspace(s_history_4[i], s_history_4[i + 1], 50, dtype=np.float_)
#                      for i in range(len(s_levels_4) - 1)])
#
#     s = np.hstack((s_1, s_2, s_3, s_4))

    s_vars = [np.zeros_like(s)
              for sv in m.sv_names]
    s_p, Y, D, z, X, tau_e, tau, s_cum = m.get_next_state(s, s_vars)
    plt.plot(s, tau)

    plt.show()
#   m.configure_traits()

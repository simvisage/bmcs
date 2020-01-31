'''
Created on 10.10.2018

@author: Mario Aguilar
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

    tau_bar = Float(50,
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
                ]

    def get_next_state(self, s, s_vars):

        tau_e, tau, s_p, z, X, D, Y, = s_vars

        # trial stress - assuming elastic increment.
        tau_trial = self.E_m * (1. - D) * s + \
            self.E_b * (1. - D) * (s - s_p)
        tau_e_trial = self.E_b * (s - s_p)
        zero_h = np.zeros_like(z)
        h_2d = np.vstack([zero_h, (self.tau_bar + self.K * z)])
        h = np.max(h_2d, axis=0)
        f_trial = np.abs(tau_e_trial - X) - self.tau_bar - self.K * z
        tau_e = tau_e_trial
        tau = tau_trial
        # identify values beyond the elastic limit
        plas_idx = np.where(f_trial > self.ZERO_THRESHOLD)[0]

        # plastic multiplier
        delta_pi = f_trial[plas_idx] / \
            (self.E_b + (self.K + self.gamma) * (1. - D[plas_idx]))

        # return mapping for isotropic and kinematic hardening
        grad_f = np.sign(tau_e_trial[plas_idx] - X[plas_idx])
        s_p[plas_idx] += delta_pi * grad_f
        Y[plas_idx] = 0.5 * self.E_m * s[plas_idx]**2. + 0.5 * \
            self.E_b * (s[plas_idx] - s_p[plas_idx])**2.
        D[plas_idx] += (Y[plas_idx] / self.S) * delta_pi
        D[D > 1.] = 1.
        # print D
        z[plas_idx] += (1. - D[plas_idx]) * delta_pi
        X[plas_idx] += self.gamma * (1. - D[plas_idx]) * delta_pi * grad_f

        # apply damage law to the effective stress
        tau_e = self.E_b * \
            (1. - D) * (s - s_p)
        tau = (self.E_m + self.E_b) * \
            (1. - D) * s - self.E_b * (1. - D) * s_p

        return s_p, Y, D, z, X, tau_e, tau

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
    s_levels = np.linspace(0, -0.05, 100)
    print(s_levels)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= 0
    #print (s_levels.reshape)
    s_history = s_levels.flatten()
    s = np.hstack([np.linspace(s_history[i], s_history[i + 1], 1)
                   for i in range(len(s_levels) - 1)])
    print(s)
    s_vars = [np.zeros_like(s)
              for sv in m.sv_names]
    s_p, Y, D, z, X, tau_e, tau = m.get_next_state(s, s_vars)
    plt.plot(s, D)
    plt.show()
#   m.configure_traits()

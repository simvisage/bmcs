'''
Created on 05.12.2016

@author: abaktheer
'''

#from scipy.misc import derivative
from scipy.optimize import newton
from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List

import numpy as np


class MATSEBondSlipEP(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(12900,
                label="G",
                desc="Shear Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(60,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)
    tau_bar = Float(5,
                    label="Tau_bar ",
                    desc="Reversibility limit",
                    enter_set=True,
                    auto_set=False)

    alpha = Float(1.0)
    beta = Float(1.0)
    g = lambda self, k: 1. / (1 + np.exp(-self.alpha * k + 6.)) * self.beta

#     def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q, kappa):
#         #         g = lambda k: 0.8 - 0.8 * np.exp(-k)
#         #         g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))
#         n_e, n_ip, n_s = eps.shape
#         D = np.zeros((n_e, n_ip, 3, 3))
#         D[:, :, 0, 0] = self.E_m
#         D[:, :, 2, 2] = self.E_f
#
#         sig_trial = sig[:, :, 1] / \
#             (1 - self.g(kappa)) + self.E_b * d_eps[:, :, 1]
#         xi_trial = sig_trial - q
#         f_trial = abs(xi_trial) - (self.tau_bar + self.K * alpha)
#
#         # print'f_trial', f_trial
#         elas = f_trial <= 1e-8
#         plas = f_trial > 1e-8
#         d_sig = np.einsum('...st,...t->...s', D, d_eps)
#         sig += d_sig
#
#         d_gamma = f_trial / (self.E_b + self.K + self.gamma) * plas
#         alpha += d_gamma
#         kappa += d_gamma
#         q += d_gamma * self.gamma * np.sign(xi_trial)
#         w = self.g(kappa)
#         # print'w=',w
#
#         sig_e = sig_trial - d_gamma * self.E_b * np.sign(xi_trial)
#         sig[:, :, 1] = (1 - w) * sig_e
#
# #         E_p = -self.E_b / (self.E_b + self.K + self.gamma) * derivative(self.g, kappa, dx=1e-6) * sig_e \
# #             + (1 - w) * self.E_b * (self.K + self.gamma) / \
# #             (self.E_b + self.K + self.gamma)
#
#         D[:, :, 1, 1] = (1 - w) * self.E_b * elas + E_p * plas
#         print'D_ed=', D[:, :, 1, 1]
#         return sig, D, alpha, q, kappa

    n_s = Constant(3)

    def get_bond_slip(self, s_arr):
        '''
        for plotting the bond slip relationship
        '''
        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)
        kappa = np.zeros_like(s_arr)
        #s_arr_p = np.zeros_like(s_arr)

        sig_e = 0.
        z = 0.
        alpha = 0.
        kappa[0] = 0.
        s_arr_p = 0.

        for i in range(1, len(s_arr)):

            d_eps = s_arr[i] - s_arr[i - 1]
            sig_e_trial = self.E_b * (s_arr[i] - s_arr_p)
            f_trial = abs(sig_e_trial - self.gamma * alpha) - \
                (self.tau_bar + self.K * z)

            kappa[i] = abs(s_arr[i])
            kappa[i] = max(kappa[i - 1], abs(s_arr[i]))
            w = self.g(abs(kappa[i]))

            if f_trial <= 1e-8:
                sig_e = sig_e_trial
            else:
                d_gamma = f_trial / (self.E_b + self.K + self.gamma)
                z += d_gamma
                alpha += d_gamma * np.sign(sig_e_trial - self.gamma * alpha)
                s_arr_p += d_gamma * \
                    np.sign(sig_e_trial - self.gamma * alpha)
                sig_e = sig_e_trial - d_gamma * self.E_b * \
                    np.sign(sig_e_trial - self.gamma * alpha)
                #sig_e = self.E_b * (s_arr[i] - s_arr_p)

            w_arr[i] = w
            sig_n_arr[i] = (1. - w) * sig_e
            sig_e_arr[i] = sig_e

        return sig_n_arr, sig_e_arr, w_arr

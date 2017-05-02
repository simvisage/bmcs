'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval
from traits.api import implements, Int, Array, \
    Constant, Float

from mats_damage_fn import LiDamageFn, JirasekDamageFn, AbaqusDamageFn


import numpy as np


class MATSEvalFatigue(MATSEval):

    E_m = Float(30000, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(200000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(200,
                label="G",
                desc="Shear Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(1,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1,
              label="r",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1,
              label="c",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(5,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    pressure = Float(-5,
                     label="Pressure",
                     desc="Lateral pressure",
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    n_s = Constant(4)

    state_array_size = Int(4)

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, w):

        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        Y = 0.5 * self.E_b * (eps[:, :, 1] - xs_pi) ** 2
        sig_pi_trial = self.E_b * (eps[:, :, 1] - xs_pi)

        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(sig_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * self.pressure / 3

        elas = f <= 1e-6
        plas = f > 1e-6

        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        # Return mapping
        delta_lamda = f / (self.E_b / (1 - w) + self.gamma + self.K) * plas
        # update all the state variables

        xs_pi = xs_pi + delta_lamda * np.sign(sig_pi_trial - X) / (1 - w)
        Y = 0.5 * self.E_b * (eps[:, :, 1] - xs_pi) ** 2

        w = w + (1 - w) ** self.c * (delta_lamda * (Y / self.S) ** self.r)

        sig[:, :, 1] = (1 - w) * self.E_b * (eps[:, :, 1] - xs_pi)
        #X = X + self.gamma * delta_lamda * np.sign(sig_pi_trial - X)
        alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)
        z = z + delta_lamda

        # Consistent tangent operator
        D_ed = self.E_b * (1 - w) - ((1 - w) * self.E_b ** 2) / (self.E_b + (self.gamma + self.K) * (1 - w))\
            - ((1 - w) ** self.c * (self.E_b ** 2) * ((Y / self.S) ** self.r)
               * np.sign(sig_pi_trial - X) * (eps[:, :, 1] - xs_pi)) / ((self.E_b / (1 - w)) + self.gamma + self.K)

        D[:, :, 1, 1] = (1 - w) * self.E_b * elas + D_ed * plas

        return sig, D, xs_pi, alpha, z, w


class MATSBondSlipDP(MATSEval):

    E_m = Float(30000, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(200000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(12900,
                label="E_b",
                desc="Bond stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(100,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(1000,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5,
                    label="Tau_pi_bar",
                    desc="Reversibility limit",
                    enter_set=True,
                    auto_set=False)

    state_array_size = Int(5)

    alpha_1 = 1.
    alpha_2 = 100.

    def omega(self, k):
        return 1. / (1 + np.exp(-1. * self.alpha_2 * k + 6.)) * self.alpha_1

    def omega_dereviative(self, k):
        return (self.alpha_1 * self.alpha_2 * np.exp(-1. * self.alpha_2 * k + 6.)) / (1 + np.exp(-1. * self.alpha_2 * k + 6.)) ** 2

    def get_corr_pred(self, s, d_s, tau, t_n, t_n1, s_p, alpha, z, kappa, omega):

        n_e, n_ip, n_s = s.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        sig_pi_trial = self.E_b * (s[:, :, 1] - s_p)

        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(sig_pi_trial - X) - self.tau_bar - Z

        elas = f <= 1e-6
        plas = f > 1e-6

        d_tau = np.einsum('...st,...t->...s', D, d_s)
        tau += d_tau

        # Return mapping
        delta_lamda = f / (self.E_b + self.gamma + self.K) * plas
        # update all the state variables

        s_p = s_p + delta_lamda * np.sign(sig_pi_trial - X)
        z = z + delta_lamda
        alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)

        kappa = np.max(np.array([kappa, np.fabs(s[:, :, 1])]), axis=0)
        omega = self.omega(kappa)
        tau[:, :, 1] = (1 - omega) * self.E_b * (s[:, :, 1] - s_p)

        # Consistent tangent operator
        D_ed = -self.E_b / (self.E_b + self.K + self.gamma) * self.omega_dereviative(kappa) * self.E_b * (s[:, :, 1] - s_p) \
            + (1 - omega) * self.E_b * (self.K + self.gamma) / \
            (self.E_b + self.K + self.gamma)

        D[:, :, 1, 1] = (1 - omega) * self.E_b * elas + D_ed * plas

        return tau, D, s_p, alpha, z, kappa, omega

    n_s = Constant(5)

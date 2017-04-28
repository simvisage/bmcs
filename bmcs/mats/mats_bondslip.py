'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval
from traits.api import implements, Int, Array, \
    Constant, Float

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

    def get_bond_slip(self, s_arr):
        '''for plotting the bond slip fatigue - Initial version modified modified threshold with cumulation-2 implicit
        '''
        # arrays to store the values
        # nominal stress
        tau_arr = np.zeros_like(s_arr)
        # damage factor
        w_arr = np.zeros_like(s_arr)
        # sliding slip
        xs_pi_arr = np.zeros_like(s_arr)
        # max sliding
        s_max = np.zeros_like(s_arr)
        # max stress
        tau_max = np.zeros_like(s_arr)
        # cumulative sliding
        xs_pi_cum = np.zeros_like(s_arr)

        # state variables
        tau_i = 0
        alpha_i = 0.
        xs_pi_i = 0
        z_i = 0.
        w_i = 0.  # damage
        X_i = self.gamma * alpha_i
        delta_lamda = 0
        Z = self.K * z_i
        xs_pi_cum_i = 0

        for i in range(1, len(s_arr)):
            # print 'increment', i
            s_i = s_arr[i]

            tau_i = (1 - w_i) * self.E_b * (s_i - xs_pi_i)

            tau_i_1 = self.E_b * (s_i - xs_pi_i)

            Y_i = 0.5 * self.E_b * (s_i - xs_pi_i) ** 2

            # Threshold
            f_pi_i = np.fabs(tau_i_1 - X_i) - \
                self.tau_pi_bar - Z + self.a * self.pressure / 3

            if f_pi_i > 1e-6:
                # Return mapping
                delta_lamda = f_pi_i / \
                    (self.E_b / (1 - w_i) + self.gamma + self.K)
                # update all the state variables

                xs_pi_i = xs_pi_i + delta_lamda * \
                    np.sign(tau_i_1 - X_i) / (1 - w_i)

                Y_i = 0.5 * self.E_b * (s_i - xs_pi_i) ** 2

                w_i = w_i + ((1 - w_i) ** self.c) * \
                    (delta_lamda * (Y_i / self.S) ** self.r)

                tau_i = self.E_b * (1 - w_i) * (s_i - xs_pi_i)
                X_i = X_i + self.gamma * delta_lamda * np.sign(tau_i_1 - X_i)
                alpha_i = alpha_i + delta_lamda * np.sign(tau_i_1 - X_i)
                z_i = z_i + delta_lamda
                xs_pi_cum_i = xs_pi_cum_i + delta_lamda

            tau_arr[i] = tau_i
            w_arr[i] = w_i
            xs_pi_arr[i] = xs_pi_i
            xs_pi_cum[i] = xs_pi_cum_i

        return tau_arr, w_arr, xs_pi_arr, xs_pi_cum


class MATSBondSlipDP(MATSEval):

    E_m = Float(30000, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(200000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(200,
                label="E_b",
                desc="Bond stiffness",
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

    tau_bar = Float(5,
                    label="Tau_pi_bar",
                    desc="Reversibility limit",
                    enter_set=True,
                    auto_set=False)

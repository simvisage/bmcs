'''
Created on 05.12.2016

@author: abaktheer
'''

#from scipy.misc import derivative
from ibvpy.api import MATSEval, IMATSEval
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str
from traitsui.api import EnumEditor

import numpy as np


class MATSBondSlipBase(MATSEval):

    implements(IMATSEval)

    material = WeakRef

    s_names = List(Str)


class MATSBondSlipEP(MATSBondSlipBase):
    '''Elastic plastic model of the bond
    '''

    def _material_changed(self):
        self.set(E_b=self.material.E_b,
                 gamma=self.material.gamma,
                 tau_bar=self.material.tau_bar,
                 K=self.material.K,
                 alpha=self.material.alpha,
                 beta=self.material.beta
                 )

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

    n_s = Constant(3)

    sv_names = ['tau',
                'tau_e',
                'z',
                'alpha',
                'kappa',
                'omega',
                's_p']

    def init_state_vars(self):
        return (np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_))

    def get_next_state(self, s, d_s, s_vars):

        print 's', s
        print 'd_s', d_s
        tau, tau_e, z, alpha, kappa, omega, s_p = s_vars

        tau_e_trial = self.E_b * (s - s_p)
        f_trial = abs(tau_e_trial - self.gamma * alpha) - \
            (self.tau_bar + self.K * z)

        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)
        omega = self.g(kappa)

        if f_trial <= 1e-8:
            tau_e = tau_e_trial
        else:
            d_gamma = f_trial / (self.E_b + self.K + self.gamma)
            z += d_gamma
            alpha += d_gamma * np.sign(tau_e_trial - self.gamma * alpha)
            s_p += d_gamma * \
                np.sign(tau_e_trial - self.gamma * alpha)
            tau_e = tau_e_trial - d_gamma * self.E_b * \
                np.sign(tau_e_trial - self.gamma * alpha)
            #sig_e = self.E_b * (s_arr[i] - s_arr_p)

        tau = (1. - omega) * tau_e
        return tau, tau_e, z, alpha, kappa, omega, s_p

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


class MATSBondSlipD(MATSBondSlipBase):

    sv_names = ['tau',
                'tau_e',
                'omega',
                ]

    '''Damage model of bond.
    '''

    def init_state_vars(self):
        return (np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_))

    def get_next_state(self, s, d_s, s_vars):
        pass


class MATSBondSlipDP(MATSBondSlipBase):
    sv_names = ['tau',
                'tau_e',
                'z',
                'xxxxxxxxxxxxx',
                's_p']

    '''Damage - plasticity model of bond.
    '''

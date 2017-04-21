'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval, IMATSEval
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property
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
                 K=self.material.K
                 )

    E_b = Float(12900,
                label="E_b",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="Isotropic harening modulus",
              enter_set=True,
              auto_set=False)
    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="yield stress",
                    enter_set=True,
                    auto_set=False)

    sv_names = ['tau',
                'tau_e',
                'z',
                'alpha',
                's_p'
                ]

    def init_state_vars(self):
        return (np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_))

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, z, alpha, s_p = s_vars

        tau_e_trial = self.E_b * (s - s_p)
        f_trial = abs(tau_e_trial - self.gamma * alpha) - \
            (self.tau_bar + self.K * z)

        if f_trial <= 1e-8:
            tau = tau_e_trial

        else:
            d_gamma = f_trial / (self.E_b + self.K + self.gamma)
            z += d_gamma
            alpha += d_gamma * self.gamma * \
                np.sign(tau_e_trial - self.gamma * alpha)
            s_p += d_gamma * \
                np.sign(tau_e_trial - self.gamma * alpha)
            tau = tau_e_trial - d_gamma * self.E_b * \
                np.sign(tau_e_trial - self.gamma * alpha)

        return tau, tau_e, z, alpha, s_p

    def get_bond_slip(self, s_arr):
        '''
        for plotting the bond slip relationship
        '''
        sig_n_arr = np.zeros_like(s_arr)
        s_arr_p = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)

        sig_e = 0.
        z = 0.
        alpha = 0.
        s_p = 0.

        for i in range(1, len(s_arr)):

            sig_e_trial = self.E_b * (s_arr[i] - s_p)
            f_trial = abs(sig_e_trial - self.gamma * alpha) - \
                (self.tau_bar + self.K * z)

            if f_trial <= 1e-8:
                sig_e = sig_e_trial

            else:
                d_gamma = f_trial / (self.E_b + self.K + self.gamma)
                z += d_gamma
                alpha += d_gamma * self.gamma * \
                    np.sign(sig_e_trial - self.gamma * alpha)
                s_p += d_gamma * \
                    np.sign(sig_e_trial - self.gamma * alpha)
                sig_e = sig_e_trial - d_gamma * self.E_b * \
                    np.sign(sig_e_trial - self.gamma * alpha)

            sig_n_arr[i] = sig_e
            s_arr_p[i] = s_p

        return sig_n_arr, s_arr_p, w_arr


class MATSBondSlipD(MATSBondSlipBase):
    '''Damage model of bond.
    '''

    def _material_changed(self):
        self.set(E_b=self.material.E_b,
                 tau_bar=self.material.tau_bar,
                 alpha=self.material.alpha,
                 beta=self.material.beta
                 )

    E_b = Float(12900,
                label="E_b",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="yield stress",
                    enter_set=True,
                    auto_set=False)

    alpha = Float(1.0,
                  label="alpha",
                  desc="parameter controls the damage function",
                  enter_set=True,
                  auto_set=False)

    beta = Float(1.0,
                 label="beta",
                 desc="parameter controls the damage function",
                 enter_set=True,
                 auto_set=False)

    g = lambda self, k: 1. / (1 + np.exp(-self.alpha * k + 6.)) * self.beta

    sv_names = ['tau',
                'tau_e',
                'kappa',
                'omega',
                ]

    s0 = Property(depends_on='tau_bar,E_b')

    @cached_property
    def _get_s0(self):
        return self.tau_bar / self.E_b

    def init_state_vars(self):
        return (np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_),
                np.array([0], dtype=np.float_))

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, kappa, omega = s_vars

        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)

        if kappa > self.s0:
            omega = self.g(np.fabs(kappa))
        else:
            omega = 0

        tau_e = self.E_b * s
        tau = (1. - omega) * tau_e
        return (tau, tau_e, kappa, omega)

    def get_bond_slip(self, s_arr):
        '''
        for plotting the bond slip relationship
        '''
        sig_e_arr = np.zeros_like(s_arr)
        sig_n_arr = np.zeros_like(s_arr)
        w_arr = np.zeros_like(s_arr)
        kappa = np.zeros_like(s_arr)

        sig_e = 0.
        kappa[0] = 0.
        s_0 = self.tau_bar / self.E_b

        for i in range(1, len(s_arr)):

            kappa[i] = abs(s_arr[i])
            kappa[i] = max(kappa[i - 1], abs(s_arr[i]))
            if kappa[i] > s_0:
                w = self.g(abs(kappa[i]))
            else:
                w = 0

            sig_e = self.E_b * s_arr[i]
            w_arr[i] = w
            sig_n_arr[i] = (1. - w) * sig_e
            sig_e_arr[i] = sig_e

        return sig_n_arr, sig_e_arr, w_arr


class MATSBondSlipDP(MATSBondSlipBase):
    sv_names = ['tau',
                'tau_e',
                'z',
                'xxxxxxxxxxxxx',
                's_p']

    '''Damage - plasticity model of bond.
    '''

    def _material_changed(self):
        self.set(E_b=self.material.E_b,
                 gamma=self.material.gamma,
                 tau_bar=self.material.tau_bar,
                 K=self.material.K,
                 alpha=self.material.alpha,
                 beta=self.material.beta
                 )

    E_b = Float(12900,
                label="G",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(1,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(1,
              label="K",
              desc="Isotropic harening modulus",
              enter_set=True,
              auto_set=False)
    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="yield stress",
                    enter_set=True,
                    auto_set=False)

    alpha = Float(1.0,
                  label="alpha",
                  desc="parameter controls the damage function",
                  enter_set=True,
                  auto_set=False)

    beta = Float(1.0,
                 label="beta",
                 desc="parameter controls the damage function",
                 enter_set=True,
                 auto_set=False)

    g = lambda self, k: 1. / (1 + np.exp(-self.alpha * k + 6.)) * self.beta

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
        s_arr_p = np.zeros_like(s_arr)

        sig_e = 0.
        z = 0.
        alpha = 0.
        kappa[0] = 0.
        s_p = 0.
        s_0 = self.tau_bar / self.E_b

        for i in range(1, len(s_arr)):

            sig_e_trial = self.E_b * (s_arr[i] - s_p)
            f_trial = abs(sig_e_trial - self.gamma * alpha) - \
                (self.tau_bar + self.K * z)

            kappa[i] = abs(s_arr[i])
            kappa[i] = max(kappa[i - 1], abs(s_arr[i]))
            if kappa[i] > s_0:
                w = self.g(abs(kappa[i]))
            else:
                w = 0

            if f_trial <= 1e-8:
                sig_e = sig_e_trial

            else:
                d_lamda = f_trial / (self.E_b + self.K + self.gamma)
                z += d_lamda
                alpha += d_lamda * self.gamma * \
                    np.sign(sig_e_trial - self.gamma * alpha)
                s_p += d_lamda * \
                    np.sign(sig_e_trial - self.gamma * alpha)
#                 sig_e = sig_e_trial - d_gamma * self.E_b * \
#                     np.sign(sig_e_trial - self.gamma * alpha)
                sig_e = self.E_b * (s_arr[i] - s_p)

            w_arr[i] = w
            s_arr_p[i] = s_p
            sig_n_arr[i] = (1. - w) * sig_e
            sig_e_arr[i] = sig_e

        return sig_n_arr, sig_e_arr, w_arr

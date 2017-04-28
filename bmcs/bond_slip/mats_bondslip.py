'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval, IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    HasStrictTraits, Instance, Callable

import numpy as np


class MATSBondSlipBase(MATSEval):

    implements(IMATSEval)

    ZERO_THRESHOLD = Constant(1e-8)

    material = WeakRef
    '''Link to a material record where the parameters are stored.
    '''

    sv_names = List(Str)
    '''Names of the state variables
    '''

    def init_state_vars(self):
        '''Initialize the state variable array.
        '''
        n_sv = len(self.sv_names)
        return [np.array([0], dtype=np.float_) for i in range(n_sv)]


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

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, z, alpha, s_p = s_vars

        # trial stress - assuming elastic increment.
        tau_e_trial = self.E_b * (s - s_p)
        X = self.gamma * alpha
        h = max(0., (self.tau_bar + self.K * z))
        f_trial = np.abs(tau_e_trial - X) - h
        tau = tau_e_trial

        # identify values beyond the elastic limit
        plas_idx = np.where(f_trial > self.ZERO_THRESHOLD)[0]

        # plastic multiplier
        d_lambda = f_trial[plas_idx] / (self.E_b + abs(self.K) + self.gamma)

        # return mapping for isotropic and kinematic hardening
        grad_f = np.sign(tau_e_trial[plas_idx] - X)
        s_p[plas_idx] += d_lambda * grad_f
        z[plas_idx] += d_lambda
        alpha[plas_idx] += d_lambda * grad_f
        tau[plas_idx] = self.E_b * (s[plas_idx] - s_p[plas_idx])

        return tau, tau_e, z, alpha, s_p


class MATSBondSlipD(MATSBondSlipBase):
    '''Damage model of bond.
    '''

    def _material_changed(self):
        self.set(E_b=self.material.E_b,
                 tau_bar=self.material.tau_bar,
                 g_fn=self.material.omega_fn
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

    g_fn = Callable

    sv_names = ['tau',
                'tau_e',
                'kappa',
                'omega',
                ]

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, kappa, omega = s_vars

        # get the maximum slip achieved so far
        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)

        omega = self.g_fn(np.fabs(kappa))

        tau_e = self.E_b * s
        tau = (1. - omega) * tau_e
        return (tau, tau_e, kappa, omega)


class MATSBondSlipDP(MATSBondSlipBase):

    '''Damage - plasticity model of bond.
    '''

    def _material_changed(self):
        print 'MATERIAL RESET TO', self.material.omega_fn.node_name
        self.set(E_b=self.material.E_b,
                 gamma=self.material.gamma,
                 tau_bar=self.material.tau_bar,
                 K=self.material.K,
                 g_fn=self.material.omega_fn
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

    g_fn = Callable

    sv_names = ['tau',
                'tau_ep',
                'z',
                'alpha',
                'kappa',
                'omega',
                's_p']

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_ep, z, alpha, kappa, omega, s_p = s_vars

        # trial stress - assuming elastic increment.
        tau_e_trial = self.E_b * (s - s_p)
        X = self.gamma * alpha
        h = max(0., (self.tau_bar + self.K * z))
        f_trial = np.abs(tau_e_trial - X) - h
        tau_ep = tau_e_trial

        # identify values beyond the elastic limit
        plas_idx = np.where(f_trial > self.ZERO_THRESHOLD)[0]

        # plastic multiplier
        d_lambda = f_trial[plas_idx] / (self.E_b + abs(self.K) + self.gamma)

        # return mapping for isotropic and kinematic hardening
        grad_f = np.sign(tau_e_trial[plas_idx] - X)
        s_p[plas_idx] += d_lambda * grad_f
        z[plas_idx] += d_lambda
        alpha[plas_idx] += d_lambda * grad_f
        tau_ep[plas_idx] = self.E_b * (s[plas_idx] - s_p[plas_idx])

        # apply damage law to the effective stress
        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)
        omega = self.g_fn(kappa)
        tau = (1. - omega) * tau_ep
        return tau, tau_ep, z, alpha, kappa, omega, s_p

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

'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval, IMATSEval
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property

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
        f_trial = np.abs(tau_e_trial - X) - self.K * z - self.tau_bar
        tau = tau_e_trial

        # identify values beyond the elastic limit
        plas_idx = np.where(f_trial > self.ZERO_THRESHOLD)[0]

        # plastic multiplier
        d_lambda = f_trial[plas_idx] / (self.E_b + self.K + self.gamma)

        # return mapping for isotropic and kinematic hardening
        grad_f = np.sign(tau_e_trial[plas_idx] - X)
        s_p[plas_idx] += d_lambda * grad_f
        z[plas_idx] += d_lambda
        alpha[plas_idx] += self.gamma * d_lambda * grad_f
        tau[plas_idx] = self.E_b * (s[plas_idx] - s_p[plas_idx])

        return tau, tau_e, z, alpha, s_p


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

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, kappa, omega = s_vars

        # get the maximum slip achieved so far
        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)

        elas_idx = np.where(kappa <= self.s0)[0]
        kappa[elas_idx] = self.s0
        omega = self.g(np.fabs(kappa))

        tau_e = self.E_b * s
        tau = (1. - omega) * tau_e
        return (tau, tau_e, kappa, omega)


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
        f_trial = np.abs(tau_e_trial - X) - self.K * z - self.tau_bar
        tau_ep = tau_e_trial

        # identify values beyond the elastic limit
        plas_idx = np.where(f_trial > self.ZERO_THRESHOLD)[0]

        # plastic multiplier
        d_lambda = f_trial[plas_idx] / (self.E_b + self.K + self.gamma)

        # return mapping for isotropic and kinematic hardening
        grad_f = np.sign(tau_e_trial[plas_idx] - X)
        s_p[plas_idx] += d_lambda * grad_f
        z[plas_idx] += d_lambda
        alpha[plas_idx] += self.gamma * d_lambda * grad_f
        tau_ep[plas_idx] = self.E_b * (s[plas_idx] - s_p[plas_idx])

        # apply damage law to the effective stress
        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)
        omega = self.g(kappa)
        tau = (1. - omega) * tau_ep
        return tau, tau_ep, z, alpha, kappa, omega, s_p

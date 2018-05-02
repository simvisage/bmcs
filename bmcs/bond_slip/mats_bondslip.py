'''
Created on 05.12.2016

@author: abaktheer
'''

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.api import MATSEval, IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

import numpy as np


class MATSBondSlipBase(MATSEval, BMCSTreeNode):

    implements(IMATSEval)

    ZERO_THRESHOLD = Constant(1e-8)

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

    node_name = 'bond model: plasticity'
    E_b = Float(12900,
                label="E_b",
                desc="bond stiffness",
                MAT=True,
                symbol=r'E_\mathrm{b}',
                unit='MPa/mm',
                enter_set=True,
                auto_set=False)

    gamma = Float(0,
                  label="Gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
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

    node_name = 'bond model: damage'
    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    E_b = Float(12900,
                label="E_b",
                symbol='E_\mathrm{b}',
                unit='MPa',
                desc="bond stiffness",
                enter_set=True,
                auto_set=False)

    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="yield stress",
                    unit='MPa',
                    symbol='\bar{\tau}',
                    enter_set=True,
                    auto_set=False)

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn,
                               FRP=FRPDamageFn,
                               ),
                          symbol='option',
                          unit='-',
                          desc='damage function [li,jirasek,abaqus,FRP]',
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_()

    omega_fn = Instance(IDamageFn,
                        report=True,
                        desc='object definng the damage function')

    def _omega_fn_default(self):
        # return JirasekDamageFn()
        return LiDamageFn(alpha_1=1.,
                          alpha_2=100.
                          )

    sv_names = ['tau',
                'tau_e',
                'kappa',
                'omega',
                ]

    def get_next_state(self, s, d_s, s_vars):

        tau, tau_e, kappa, omega = s_vars

        # get the maximum slip achieved so far
        kappa = np.max(np.array([kappa, np.fabs(s)]), axis=0)

        omega = self.omega_fn(np.fabs(kappa))

        tau_e = self.E_b * s
        tau = (1. - omega) * tau_e
        return (tau, tau_e, kappa, omega)


class MATSBondSlipDP(MATSBondSlipBase):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    E_b = Float(12900,
                label="E_b",
                MAT=True,
                symbol=r'E_\mathrm{b}',
                unit='MPa/mm',
                desc='elastic bond stiffness',
                enter_set=True,
                auto_set=False)

    gamma = Float(1,
                  label="Gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(1,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5,
                    label="Tau_0 ",
                    desc="Yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn,
                               FRP=FRPDamageFn,
                               ),
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_()

    omega_fn = Instance(IDamageFn,
                        report=True)

    def _omega_fn_default(self):
        # return JirasekDamageFn()
        return LiDamageFn(alpha_1=1.,
                          alpha_2=100.
                          )

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
        omega = self.omega_fn(kappa)
        tau = (1. - omega) * tau_ep
        return tau, tau_ep, z, alpha, kappa, omega, s_p

    tree_view = View(
        Group(
            VGroup(
                VGroup(
                    Item('E_b', full_size=True, resizable=True),
                    Item('gamma'),
                    Item('K'),
                    Item('tau_bar'),
                ),
                VGroup(
                    Item('omega_fn_type'),
                ),
                UItem('omega_fn@')
            )
        ),
        width=0.4,
        height=0.5,
    )

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
    m = MATSBondSlipDP()
    m.configure_traits()

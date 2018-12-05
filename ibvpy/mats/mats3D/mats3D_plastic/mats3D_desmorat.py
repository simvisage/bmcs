'''
Created on 30.10.2018

@author: aguilar
'''
from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

import numpy as np
import traits.api as tr


class MATS3DDesmorat(MATS3DEval, MATS3D):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

    tr.implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    E_m = tr.Float(16e+3,
                   label="E_m",
                   desc="Young's Modulus",
                   auto_set=False,
                   input=True)
    E_b = tr.Float(19e+3,
                   label="E_b",
                   desc="Young's Modulus Bond",
                   auto_set=False,
                   input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    def _get_lame_m_params(self):
        la = self.E_m * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_m / (2. + 2. * self.nu)
        return la, mu

    D_m_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_m_abef(self):
        la = self._get_lame_m_params()[0]
        mu = self._get_lame_m_params()[1]
        delta = np.identity(3)
        D_m_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                    np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                    np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_m_abef

    def _get_lame_b_params(self):
        la = self.E_b * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_b / (2. + 2. * self.nu)
        return la, mu

    D_b_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_b_abef(self):
        la = self._get_lame_b_params()[0]
        mu = self._get_lame_b_params()[1]
        delta = np.identity(3)
        D_b_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                    np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                    np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_b_abef

    tree_node_list = List([])

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

    S = Float(476,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(9,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    state_array_shapes = {'tau_e': (3, 3),
                          'tau': (3, 3),
                          'eps_pi': (3, 3),
                          'X': (3, 3),
                          'z': (),
                          'omega': (),
                          'Y': ()}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, algorithmic,
                      tau_e, tau, eps_pi, X, z, omega, Y):
        r'''
        Corrector predictor computation.
        '''
        if update_state:
            eps_Emab_n = eps_Emab_n1 - deps_Emab

        sigma_Emab = np.einsum(
            '...abcd,...cd->...ab', self.D_abef, eps_Emab_n1
        )
        return self.D_abef, sigma_Emab

    def get_next_step(self, eps, eps_pi, Y, D, z, X, tau_e, tau, g):

        D_m = self.D_m_abef
        D_b = self.D_b_abef
        #v = len(s) - 1

        for i in range(len(s_levels) - 1):

            # trial stress - assuming elastic increment.
            eps_diff = eps[i + 1] - eps_pi[i]
            bn = (1. - D[i])
            tau_trial = (np.einsum('ijkl,lk->ij', D_m, eps[i + 1]) * (1. - D[i]) +
                         np.einsum('ijkl,lk->ij', D_b, eps_diff)) * (1. - D[i])
            tau_e_trial = (np.einsum('ijkl,lk->ij', D_b, eps_diff))
            norm = tau_e_trial - X[i]
            f_trial = np.sqrt(np.einsum('nj,nj', norm, norm)
                              ) - self.tau_bar - self.K * z[i]

            if f_trial <= 0:
                tau_e[i + 1] = tau_e_trial
                tau[i + 1] = tau_trial
                eps_pi[i + 1] = eps_pi[i]
                z[i + 1] = z[i]
                X[i + 1] = X[i]
                D[i + 1] = D[i]
                Y[i + 1] = Y[i]

            # identify values beyond the elastic limit
            else:
                # identify values beyond the elastic limit

                # sliding  multiplier
                delta_pi = f_trial / \
                    (self.E_b + (self.K + self.gamma) * (1. - D[i]))

         # return mapping for isotropic and kinematic hardening
                #grad_f = np.sign(tau_e_trial - X[i])
                norm2 = tau_e[i] - X[i]
                norm3 = np.sqrt(np.einsum('nj,nj', norm, norm))
                eps_pi[i + 1] = eps_pi[i] + norm2 * delta_pi / norm3
                Y[i + 1] = 0.5 * (np.einsum('ij,ijkl,lk', eps[i + 1], D_m, eps[i + 1])) + \
                    0.5 * (np.einsum('ij,ijkl,lk', eps_diff, D_b, eps_diff))
                D_trial = D[i] + (Y[i + 1] / self.S) * delta_pi  # * 1e6
                if D_trial > 1.0:
                    D[i + 1] = 1.0
                else:
                    D[i + 1] = D_trial
                g[i + 1] = g[i] + (1. - D[i + 1]) * (eps_pi[i + 1] - eps_pi[i])
                X[i + 1] = X[i] + self.gamma * g[i + 1]
                z[i + 1] = z[i] + delta_pi * (1. - D[i + 1])

                # apply damage law to the effective stress
                eps_diff = eps[i + 1] - eps_pi[i + 1]
                tau[i + 1] = (np.einsum('ijkl,lk->ij', D_m, eps[i + 1]) * (1. - D[i + 1]) + (np.einsum('ijkl,lk->ij',
                                                                                                       D_m, eps[i + 1])) * (1. - D[i + 1]) - np.einsum('ijkl,lk->ij', D_b, eps_pi[i + 1])) * (1. - D[i + 1])
                tau_e[i + 1] = (np.einsum('ijkl,lk->ij', D_b, eps_diff))
        return eps_pi, Y, D, z, X, tau_e, tau, g

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    m = MATS3DDesmorat()
    s_levels = np.linspace(0, -0.0035, 10)

    eps = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])

    for i in range(len(s_levels)):
        eps[i][0][0] = s_levels[i]

    eps_pi = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])
    X = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])
    g = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])
    tau_e = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])
    tau = np.array([np.zeros((3, 3)) for _ in range(len(s_levels))])
    Y = np.zeros_like(s_levels)
    D = np.zeros_like(s_levels)
    z = np.zeros_like(s_levels)
    eps_pi, Y, D, z, X, tau_e, tau, g = m.get_next_state(
        eps, eps_pi, Y, D, z, X, tau_e, tau, g)
    plt.plot(eps[:, 0, 0], tau[:, 0, 0])
    plt.show()
#   m.configure_traits()

'''
Created on 06.11.2018

@author: aguilar
'''
from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

from mats_bondslip import MATSBondSlipBase
import numpy as np
import traits.api as tr


class MATS3DDesmorat(MATSBondSlipBase):

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

    S = Float(476e-6,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(6,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    sv_names = ['tau_e',
                'tau',
                'eps_pi',
                'X'
                'z',
                'D',
                'Y',
                'eps_cum',
                ]

    def get_next_state(self, eps, eps_pi, Y, D, z, X, tau_e, tau, g, N, eps_cum):

        D_m = self.D_m_abef
        D_b = self.D_b_abef

        for i in range(len(s) - 1):

            # trial stress - assuming elastic increment.
            eps_diff = eps[i + 1] - eps_pi[i]
            tau_trial = (np.einsum('ijkl,lk->ij', D_m, eps[i + 1]) * (1. - D[i]) + (np.einsum('ijkl,lk->ij',
                                                                                              D_b, eps[i + 1])) * (1. - D[i]) - np.einsum('ijkl,lk->ij', D_b, eps_pi[i])) * (1. - D[i])
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
                g[i + 1] = g[i]
                N[i] = i + 1
                eps_cum[i + 1] = eps_cum[i]

            # identify values beyond the elastic limit
            else:
                # identify values beyond the elastic limit

                # sliding  multiplier
                delta_pi = f_trial / \
                    (self.E_b + (self.K + self.gamma) * (1. - D[i]))

         # return mapping for isotropic and kinematic hardening
                norm2 = tau_e_trial - X[i]
                norm3 = np.sqrt(np.einsum('nj,nj', norm2, norm2))
                eps_pi[i + 1] = eps_pi[i] + norm2 * delta_pi / norm3
                eps_cum[i + 1] = eps_cum[i] + norm2 * \
                    norm2 * delta_pi / (norm3 * norm3)
                Y[i + 1] = 0.5 * (np.einsum('ij,ijkl,lk', eps[i + 1], D_m, eps[i + 1])) + \
                    0.5 * (np.einsum('ij,ijkl,lk', eps_diff, D_b, eps_diff))
                D_trial = D[i] + (Y[i + 1] / self.S) * delta_pi
                N[i] = i + 1
                if D[i] > 0.9:
                    break
                else:
                    D[i + 1] = D_trial
                g[i + 1] = g[i] + (1. - D[i + 1]) * norm2 * delta_pi / norm3
                X[i + 1] = self.gamma * g[i + 1]
                z[i + 1] = z[i] + delta_pi * (1. - D[i + 1])

                # apply damage law to the effective stress
                eps_diff = eps[i + 1] - eps_pi[i + 1]
                tau[i + 1] = (np.einsum('ijkl,lk->ij', D_m, eps[i + 1]) * (1. - D[i + 1]) + (np.einsum('ijkl,lk->ij',
                                                                                                       D_b, eps[i + 1])) * (1. - D[i + 1]) - np.einsum('ijkl,lk->ij', D_b, eps_pi[i + 1])) * (1. - D[i + 1])
                tau_e[i + 1] = (np.einsum('ijkl,lk->ij', D_b, eps_diff))
        return eps_pi, Y, D, z, X, tau_e, tau, g, N, eps_cum

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
    N_R_1 = np.zeros(20)
    s_max_1 = np.zeros(20)
    for j in range(20):
        m = MATS3DDesmorat()
        s_max_1[j] = 0.0025 - 0.002 * j / (j + 1)
        s_levels_1 = np.linspace(0, s_max_1[j], 1000)
        s_levels_1.reshape(-1, 2)[:, 0] = 0
        s_levels_1.reshape(-1, 2)[:, 1] = s_max_1[j]
        s_levels_1[0] = 0
        s_history_1 = s_levels_1.flatten()
        s = np.hstack([np.linspace(s_history_1[i], s_history_1[i + 1], 1, dtype=np.float_)
                       for i in range(len(s_levels_1) - 1)])

        eps = np.array([np.zeros((3, 3)) for _ in range(len(s))])

        for i in range(len(s)):
            eps[i][0][0] = s[i]

        eps_pi = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        X = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        g = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        tau_e = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        tau = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        Y = np.zeros_like(s)
        D = np.zeros_like(s)
        z = np.zeros_like(s)
        N = np.zeros(1000)
        eps_pi, Y, D, z, X, tau_e, tau, g, N, eps_cum = m.get_next_state(
            eps, eps_pi, Y, D, z, X, tau_e, tau, g, N, eps_cum)
        n = np.amax(N) / 2
        N_R_1[j] = np.floor(n) + 1
    plt.semilogx(N_R_1, s_max_1)
    plt.ylim(0, 0.004)
    plt.show()
#   m.configure_traits()

'''
Created on 06.11.2018

@author: aguilar
'''
from ibvpy.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import provides,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

import numpy as np
import traits.api as tr

from .mats_bondslip import MATSBondSlipBase


@provides(IMATSEval)
class MATS3DDesmorat(MATSBondSlipBase):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

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

    def get_next_state(self, eps, eps_pi, Y, D, z, X, tau_e, tau, g, eps_cum):

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
                eps_cum[i + 1] = eps_cum[i]

            # identify values beyond the elastic limit
            else:
                # identify values beyond the elastic limit

                # sliding  multiplier
                delta_pi = f_trial / \
                    (self.E_b + (self.K + self.gamma) * (1. - D[i]))

         # return mapping for isotropic and kinematic hardening
                numerador = tau_e_trial - X[i]
                norm = np.sqrt(np.einsum('nj,nj', numerador, numerador))
                sign = numerador / norm
                eps_pi[i + 1] = eps_pi[i] + delta_pi * sign
                eps_cum[i + 1] = eps_cum[i] + delta_pi * \
                    np.einsum('ij,lk->ik', sign, sign)
                Y[i + 1] = 0.5 * (np.einsum('ij,ijkl,lk', eps[i + 1], D_m, eps[i + 1])) + \
                    0.5 * (np.einsum('ij,ijkl,lk', eps_diff, D_b, eps_diff))
                D_trial = D[i] + (Y[i + 1] / self.S) * delta_pi
                if D[i] > 0.9:
                    break
                else:
                    D[i + 1] = D_trial
                g[i + 1] = g[i] + (1. - D[i + 1]) * delta_pi * sign
                X[i + 1] = self.gamma * g[i + 1]
                z[i + 1] = z[i] + delta_pi * (1. - D[i + 1])

                # apply damage law to the effective stress
                eps_diff = eps[i + 1] - eps_pi[i + 1]
                tau[i + 1] = (np.einsum('ijkl,lk->ij', D_m, eps[i + 1]) * (1. - D[i + 1]) + (np.einsum('ijkl,lk->ij',
                                                                                                       D_b, eps[i + 1])) * (1. - D[i + 1]) - np.einsum('ijkl,lk->ij', D_b, eps_pi[i + 1])) * (1. - D[i + 1])
                tau_e[i + 1] = (np.einsum('ijkl,lk->ij', D_b, eps_diff))
#                 if tau[i, 0, 0] > 0:
#                     tau[i, 0, 0] = 0
#                     D[i] = 0
#                     s_cum = [i, 0, 0]
#                     break
        return eps_pi, Y, D, z, X, tau_e, tau, g, eps_cum

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
    s_levels_1 = np.linspace(0, -0.0022, 4)
    s_levels_1.reshape(-1, 2)[:, 0] = 0.0
    s_levels_1.reshape(-1, 2)[:, 1] = -0.0022
    s_levels_1[0] = 0
    s_history_1 = s_levels_1.flatten()
    s_1 = np.hstack([np.linspace(s_history_1[i], s_history_1[i + 1], 50, dtype=np.float_)
                     for i in range(len(s_levels_1) - 1)])

    s_levels_2 = np.linspace(0, -0.0026, 4)
    s_levels_2.reshape(-1, 2)[:, 0] = -0.0010
    s_levels_2.reshape(-1, 2)[:, 1] = -0.0026
    s_levels_2[0] = -0.0022
    s_history_2 = s_levels_2.flatten()
    s_2 = np.hstack([np.linspace(s_history_2[i], s_history_2[i + 1], 50, dtype=np.float_)
                     for i in range(len(s_levels_2) - 1)])

    s_levels_3 = np.linspace(0, -0.0028, 4)
    s_levels_3.reshape(-1, 2)[:, 0] = -0.0013
    s_levels_3.reshape(-1, 2)[:, 1] = -0.0028
    s_levels_3[0] = -0.0026
    s_history_3 = s_levels_3.flatten()
    s_3 = np.hstack([np.linspace(s_history_3[i], s_history_3[i + 1], 50, dtype=np.float_)
                     for i in range(len(s_levels_3) - 1)])

    s_levels_4 = np.linspace(0, -0.003, 4)
    s_levels_4.reshape(-1, 2)[:, 0] = -0.0015
    s_levels_4.reshape(-1, 2)[:, 1] = -0.003
    s_levels_4[0] = -0.0028
    s_history_4 = s_levels_4.flatten()
    s_4 = np.hstack([np.linspace(s_history_4[i], s_history_4[i + 1], 50, dtype=np.float_)
                     for i in range(len(s_levels_4) - 1)])

    s = np.hstack((s_1, s_2, s_3, s_4))

    eps = np.array([np.zeros((3, 3)) for _ in range(len(s))])

    for i in range(len(s)):
        eps[i][0][0] = s[i]

    eps_pi = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    eps_cum = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    X = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    g = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    tau_e = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    tau = np.array([np.zeros((3, 3)) for _ in range(len(s))])
    Y = np.zeros_like(s)
    D = np.zeros_like(s)
    z = np.zeros_like(s)
    eps_pi, Y, D, z, X, tau_e, tau, g, eps_cum = m.get_next_state(
        eps, eps_pi, Y, D, z, X, tau_e, tau, g, eps_cum)
    np.save(
        r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_original.npy', eps)
    np.save(
        r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\sigma_original.npy', tau)
    np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_cum_original.npy', eps_cum)
    np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\D_original.npy', D)
    plt.plot(eps[:, 0, 0], tau[:, 0, 0])
    print(tau)
    #plt.plot(eps[:, 0, 0], D)
    plt.show()
#   m.configure_traits()

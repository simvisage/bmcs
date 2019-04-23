'''
Created on 10.12.2018

@author: Mario Aguilar Rueda
'''
from ibvpy.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.mats.mats_eval import IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import provides, \
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

    lambda1 = Float(12500,
                    label='lambda',
                    desc="1st Lame parameter",
                    auto_set=False,
                    input=True)

    mu = Float(18750,
               label='mu',
               desc="2nd Lame parameter",
               auto_set=False,
               input=True)

    C1 = Float(19e-4,
               label="C1",
               desc="C1 material parameter",
               MAT=True,
               symbol=r'\C1',
               unit='-',
               enter_set=True,
               auto_set=False)

    K = Float(485e-5,
              label="K",
              desc="K material parameter",
              MAT=True,
              symbol='K',
              unit='-',
              enter_set=True,
              auto_set=False)

    n = Float(10,
              label="n",
              desc="n material parameter",
              MAT=True,
              symbol='n',
              unit='-',
              enter_set=True,
              auto_set=False)
    alpha = Float(2237.5,
                  label="alpha",
                  desc="alpha material parameter",
                  MAT=True,
                  symbol='alpha',
                  unit='-',
                  enter_set=True,
                  auto_set=False)

    beta = Float(-2116.5,
                 label="beta",
                 desc="beta material parameter",
                 MAT=True,
                 symbol='beta',
                 unit='-',
                 enter_set=True,
                 auto_set=False)

    g = Float(-10,
              label="g",
              desc="g material parameter",
              MAT=True,
              symbol='g',
              unit='-',
              enter_set=True,
              auto_set=False)

    C0 = Float(0.0,
               label="C0",
               desc="C0 material parameter",
               MAT=True,
               symbol='C0',
               unit='-',
               enter_set=True,
               auto_set=False)

    sv_names = ['sigma',
                'eps_pos',
                'omega',
                'Y',
                'Y_pos',
                ]

    def get_next_state(self, eps, sigma, eps_pos, omega, Y, Y_pos, N):

        for i in range(len(s)):

            # trial stress - assuming elastic increment.

            sigma[i] = (self.lambda1 * np.trace(eps[i]) + 2 * self.mu) * eps[i] + self.g * omega[i] + self.alpha * ((np.trace(np.einsum('ij,jk->ik', eps[i], omega[i])))
                                                                                                                    * np.identity(3) + np.trace(eps[i]) * omega[i]) + 2 * self.beta * (np.einsum('ij,jk->ik', eps[i], omega[i]) + np.einsum('ij,jk->ik', omega[i], eps[i]))

            Y[i] = -self.g * eps[i] - self.alpha * \
                (np.trace(eps[i])) * eps[i] - 2 * self.beta * \
                (np.einsum('ij,jk->ik', eps[i], eps[i]))
            aux = eps[i]

            for j in range(eps[i].shape[0]):
                for k in range(eps[i].shape[1]):
                    if eps[i][j][k] > 0:
                        eps_pos[i][j][k] = eps[i][j][k]
                    else:
                        eps_pos[i][j][k] = 0

            eps_eigenpos = np.linalg.eigvals(eps_pos[i])
            eps_norm = np.sqrt(
                eps_eigenpos[0]**2 + eps_eigenpos[1]**2 + eps_eigenpos[2]**2)

            f = self.g * eps_norm / np.sqrt(2) - \
                (self.C0 - self.C1 * np.trace(omega[i]))

            for j in range(Y[i].shape[0]):
                for k in range(Y[i].shape[1]):
                    if Y[i][j][k] > 0:
                        Y_pos[i][j][k] = Y[i][j][k]
                    else:
                        Y[i][j][k] = 0

            m = (np.einsum('ij,ij', Y[i], Y[i])) - \
                (np.einsum('ij,ij', Y[i - 1], Y[i - 1]))

            if m > 0:
                omega[i] = omega[i - 1] + ((f / self.K)**self.n) * ((np.einsum('ij,ij', eps_pos[i], eps_pos[i] - eps_pos[i - 1])) / self.C1 * np.trace(
                    eps_pos[i])) * (eps_pos[i] / np.sqrt(2 * np.trace(np.einsum('ij,jk->ik', eps_pos[i], eps_pos[i]))))
            else:
                omega[i] = omega[i - 1]

            if np.any(omega[i]) > 1:
                break

            N[i] = i

        return sigma, eps_pos, omega, Y, Y_pos, N


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    cycles = 10
    points = 100
    eps_max = 0.003
    eps_max_min = 0.0
    eps_min = 0
    N_R_1 = np.zeros(points)
    d = np.zeros(points)
    s_max_1 = np.zeros(points)
    auxN = 0
    for j in range(1):
        m = MATS3DDesmorat()
        s_max_1[j] = eps_max  # - (eps_max - eps_max_min) * j / points
        s_levels_1 = np.linspace(0, s_max_1[j], cycles * 2)
        s_levels_1.reshape(-1, 2)[:, 0] = 0.001
        s_levels_1.reshape(-1, 2)[:, 1] = s_max_1[j]
#         s_levels_1[0] = 0
#         s_history_1 = s_levels_1.flatten()
        s = np.hstack([np.linspace(s_levels_1[i], s_levels_1[i + 1], 5, dtype=np.float_)
                       for i in range(len(s_levels_1) - 1)])

        eps = np.array([np.zeros((3, 3)) for _ in range(len(s))])

        eps = np.array([np.zeros((3, 3)) for _ in range(len(s))])

        for i in range(len(s)):
            eps[i][0][0] = s[i]
        print(eps)

        sigma = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        eps_pos = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        omega = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        Y = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        Y_pos = np.array([np.zeros((3, 3)) for _ in range(len(s))])
        N = np.zeros_like(s)

        sigma, eps_pos, omega, Y, Y_pos, N = m.get_next_state(
            eps, sigma, eps_pos, omega, Y, Y_pos, N)
        n = np.amax(N) / 2
        N_R_1[j] = np.floor(n) + 1
#     np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_original.npy', eps)
#     np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\sigma_original.npy', tau)
#     np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_cum_original.npy', eps_cum)
# np.save(r'C:\Users\mario\Desktop\Master\HiWi\Desmorat
# 3D\Original\D_original.npy', D)
    plt.plot(eps[:][0][0], omega[:][0][0])
    #plt.plot(eps[:, 0, 0], D)
    plt.show()
#   m.configure_traits()

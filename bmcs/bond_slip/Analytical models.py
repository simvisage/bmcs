'''
Created on 03.01.2019

@author: Mario Aguilar Rueda
'''

import os

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group

import matplotlib.pyplot as plt
from mats_bondslip import MATSBondSlipBase
import numpy as np


class MATSANALYTICAL(MATSBondSlipBase):

    node_name = 'Mats Analytical simplified models'

    tree_node_list = List([])

    # S-level model, Fatigue Strain and Fatigue Modulus of Concrete

    eps0 = Float(0.7,
                 label="eps0",
                 MAT=True,
                 unit='-',
                 enter_set=True,
                 auto_set=False)

    alpha = Float(0.25,
                  label="alpha",
                  MAT=True,
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    p = Float(2,
              label="p",
              MAT=True,
              unit='-',
              enter_set=True,
              auto_set=False)

    def get_next_state_S_level(self, n, eps, beta, p, eps0, alpha, N):

        for i in range(n):
            p[i] = self.p * (1. + float(i))
            eps0[i] = self.eps0 * (1. - 0.05 * float(i))
            alpha[i] = self.alpha * (1. - 0.02 * float(i))
            beta[i] = ((1. - eps0[i]) / alpha[i])**(-p[i]) + 1.

            eps[i, :] = eps0[i] + alpha[i] * \
                ((beta[i] / (beta[i] - N)) - 1.)**(1. / p[i])

        return eps


if __name__ == '__main__':

    m = MATSANALYTICAL()

    # S_levels

    n = 6
    N = np.linspace(0, 1, 100)
    eps = np.zeros((n, len(N)))
    beta = np.zeros(n)
    p = np.zeros(n)
    eps0 = np.zeros(n)
    alpha = np.zeros(n)

    eps = m.get_next_state_S_level(n, eps, beta, p, eps0, alpha, N)

    plt.subplot(111)
    axes = plt.gca()
    axes.set_ylim([0.2, 1.1])
    for i in range(n):
        plt.plot(N, eps[i])

    plt.show()
# m.configure_traits()

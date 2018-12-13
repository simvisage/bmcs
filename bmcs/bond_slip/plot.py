'''
Created on 20.10.2018

@author: Mario Aguilar
'''

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group

import matplotlib.pyplot as plt
from mats_bondslip import MATSBondSlipBase
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from io import StringIO

    N_R_1 = np.loadtxt(fname="C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\N=0.txt")
    N_R_2 = np.loadtxt(fname="C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\N=-epsMax.txt")
    s_max_1 = np.loadtxt(fname="C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\epsMin=0.txt")
    s_max_2 = np.loadtxt(fname="C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\epsMin=-epsMax.txt")

    eps_monotonic = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_monotonic.npy")
    sigma_monotonic = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\sigma_monotonic.npy")

    eps = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_original.npy")
    sigma = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\sigma_original.npy")
    eps_cum = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\eps_cum_original.npy")
    D = np.load("C:\Users\mario\Desktop\Master\HiWi\Desmorat 3D\Original\D_original.npy")

    plt.subplot(221)
    plt.plot(eps_monotonic[:, 0, 0], sigma_monotonic[:, 0, 0], 'k')
    plt.xlabel('strain')
    plt.ylabel('stress(MPa)')

    plt.subplot(222)
    plt.plot(eps[:, 0, 0], sigma[:, 0, 0], 'k')
    plt.xlabel('strain')
    plt.ylabel('stress(MPa)')

    plt.subplot(223)
    plt.plot(eps_cum[:, 0, 0], D[:], 'k')
    plt.xlabel('cumulative sliding')
    plt.ylabel('damage')

    plt.subplot(224)
    plt.semilogx(N_R_1[:], s_max_1[:])
    plt.semilogx(N_R_2[:], s_max_2[:], 'k')
    plt.xlabel('Nlog')
    plt.ylabel('Eps max')
    plt.ylim(0, 0.004)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                        right=0.95, hspace=0.1, wspace=0.35)

    plt.show()

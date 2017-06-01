'''
Created on May 23, 2017
'''

from traits.api import Float, Property
from traitsui.api import View, Item, VGroup, UItem

import numpy as np
import pylab as p


def phi_inv(sig, f_t, G_f):
    '''Inverted softening law'''
    return -f_t / G_f * np.log(sig / f_t)


f_t = 2.4
G_f = 0.090
E = 20000.0
eps_0 = f_t / E
eps_ch = G_f / f_t
L_ch = E * G_f / f_t**2
h_b = L_ch


L = 45.0
u_max = 0.15
eps_max = eps_ch

n_T = 100
K_max = 200

for N in [1.0, 2.0, 3.0, 6.0]:
    sig_t = [0, f_t]
    eps_t = [0, f_t / E]
    L_e = (N - 1) / N * L
    L_s = 1.0 / N * L
    for sig in np.linspace(f_t, 0.1, n_T):
        u_e = sig / E * L_e
        u_s = phi_inv(sig, f_t, G_f)
        u = u_e + u_s
        eps_t.append(u / L)
        sig_t.append(sig)

    p.plot(eps_t, sig_t, label='N=%d' % N)
p.legend()
p.show()

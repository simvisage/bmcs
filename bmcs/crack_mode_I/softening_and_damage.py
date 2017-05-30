'''
Created on May 23, 2017
'''

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn
from traits.api import Float

import numpy as np
import pylab as p


E = 34000.0
f_t = 4.5
G_f = 0.004


def phi(eps_s, f_t, G_f, L_s):
    '''Softening law'''
    return f_t * np.exp(-f_t * eps_s * L_s / G_f)


def d_phi(eps_s, f_t, G_f):
    '''Derivative of the softening law'''
    return - f_t**2.0 / G_f * np.exp(-f_t / G_f * eps_s)


eps_0 = f_t / E
eps_ch = G_f / f_t
L = 1.0
u_max = L * eps_ch
eps_max = eps_ch
eps = np.linspace(0, eps_max, 100)


class GfDamageFn(PlottableFn):

    L_s = Float(1.0)

    def __call__(self, eps):
        L_s = self.L_s
        g_eps = 1 - f_t * np.exp(-f_t * (eps - eps_0) * L_s / G_f) \
            / (E * eps)
        g_eps[np.where(eps - eps_0 < 0)] = 0
        return g_eps

    def diff(self, eps):
        L_s = self.L_s
        d_g_eps = (f_t * np.exp(L_s * (eps_0 - eps) * f_t / G_f)
                   / (E * G_f * eps**2) *
                   (G_f + L_s * eps * f_t))
        d_g_eps[np.where(eps - eps_0 < 0)] = 0.0
        return d_g_eps


omega_fn = LiDamageFn(alpha_1=1.0, alpha_2=200, s_0=eps_0)
#omega_fn = JirasekDamageFn(s_0=eps_0)
#omega_fn = GfDamageFn()

omega_fn.configure_traits()
n_T = 100
K_max = 200
u_max = 0.1

for N in [1.00001, 2.0]:
    sig_t = []
    eps_t = []
    L_el = (N - 1.0) / N * L
    L_s = 1.0 / N * L
    eps_s = 0.0000
    eps_s_arr = np.array([eps_s], dtype=np.float_)
    u_t = np.linspace(0, u_max, n_T)
    omega_fn.L_s = 1.0
    for u in u_t:
        for K in range(K_max):
            R = 1 / L_el * (u - eps_s_arr * L_s) - \
                (1 - omega_fn(eps_s_arr)) * eps_s_arr
            if np.fabs(R) < 1e-8:
                break
            dR = -L_s / L_el + omega_fn.diff(eps_s_arr) * \
                eps_s_arr - (1 - omega_fn(eps_s_arr))
            d_eps_s = -R / dR
            eps_s_arr += d_eps_s
            if K == K_max - 1:
                raise ValueError, 'No convergence'
        sig = ((1.0 - omega_fn(eps_s_arr)) * E * eps_s_arr)[0]
        sig_t.append(sig)
        eps_t.append(u / L)

    p.plot(u_t, sig_t)
p.show()

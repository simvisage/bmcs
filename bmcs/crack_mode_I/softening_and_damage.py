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


def phi(eps_w, f_t, G_f, L_w):
    '''Softening law'''
    return f_t * np.exp(-f_t * eps_w * L_w / G_f)


def d_phi(eps_w, f_t, G_f):
    '''Derivative of the softening law'''
    return - f_t**2.0 / G_f * np.exp(-f_t / G_f * eps_w)


eps_0 = f_t / E
eps_ch = G_f / f_t
L = 1.0
u_max = L * eps_ch
eps_max = eps_ch
eps = np.linspace(0, eps_max, 100)

u_max = 0.00361632973221


class GfDamageFn(PlottableFn):

    L_w = Float(1.0)

    def __call__(self, eps):
        L_w = self.L_w
        g_eps = 1 - f_t * np.exp(-f_t * (eps - eps_0) * L_w / G_f) \
            / (E * eps)
        g_eps[np.where(eps - eps_0 < 0)] = 0
        return g_eps

    def diff(self, eps):
        L_w = self.L_w
        d_g_eps = (f_t * np.exp(L_w * (eps_0 - eps) * f_t / G_f)
                   / (E * G_f * eps**2) *
                   (G_f + L_w * eps * f_t))
        d_g_eps[np.where(eps - eps_0 < 0)] = 0.0
        return d_g_eps


omega_fn = LiDamageFn(alpha_1=1.0, alpha_2=20000, s_0=eps_0)
#omega_fn = JirasekDamageFn(s_0=eps_0)
omega_fn = GfDamageFn()
# p.plot(eps, omega_fn(eps))
# p.plot(eps, omega_fn_gf(eps))
# p.show()

n_T = 500
K_max = 1000

for N in [1.00000001, 2, 3, 4, 6, 7]:
    sig_t = []
    eps_t = []
    L_el = (N - 1.0) / N * L
    L_w = 1.0 / N * L
    eps_w = 0.00001
    eps_w_arr = np.array([eps_w], dtype=np.float_)
    u_t = np.linspace(0, u_max, n_T)
    omega_fn.L_w = L_w
    for u in u_t:
        for K in range(K_max):
            R = 1 / L_el * (u - eps_w_arr * L_w) - \
                (1 - omega_fn(eps_w_arr)) * eps_w_arr
            if np.fabs(R) < 1e-8:
                print 'step %g converged in %d steps' % (u, K)
                break
            dR = -L_w / L_el + omega_fn.diff(eps_w_arr) * \
                eps_w_arr - (1 - omega_fn(eps_w_arr))
            d_eps_w = -R / dR
            eps_w_arr += d_eps_w
        if K == K_max - 2:
            raise ValueError, 'No convergence'
        sig = ((1.0 - omega_fn(eps_w_arr)) * E * eps_w_arr)[0]
        sig_t.append(sig)
        eps_t.append(u / L)

    p.plot(u_t, sig_t)
p.show()

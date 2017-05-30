'''
Created on May 23, 2017
'''

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
eps_max = 1.0 * eps_ch
eps = np.linspace(0, eps_max, 100)

print 'eps_0', eps_0


def g(eps, L_w):
    g_eps = 1 - f_t * np.exp(-f_t * (eps - eps_0) * L_w / G_f) \
        / (E * eps)
    g_eps[np.where(eps - eps_0 < 0)] = 0
    return g_eps


def d_g(eps, L_w):
    d_g_eps = (f_t * np.exp(L_w * (eps_0 - eps) * f_t / G_f)
               / (E * G_f * eps**2) *
               (G_f + L_w * eps * f_t))
    d_g_eps[np.where(eps - eps_0 < 0)] = 0.0
    return d_g_eps


n_T = 100
u_max = 0.004

L = 1.0
N = 30.0

for N in [10, 20, 30]:
    sig_t = []
    eps_t = []
    L_el = (N - 1.0) / N * L
    L_w = 1.0 / N * L
    eps_w = 0.00001
    eps_w_arr = np.array([eps_w], dtype=np.float_)
    u_t = np.linspace(0, u_max, n_T)
    h_e = L_w
    for u in u_t:
        for K in range(100):
            R = 1 / L_el * (u - eps_w_arr * L_w) - \
                (1 - g(eps_w_arr, h_e)) * eps_w_arr
            if np.fabs(R) < 1e-8:
                print 'converged in ', K
                break
            dR = -L_w / L_el + d_g(eps_w_arr, h_e) * \
                eps_w_arr - (1 - g(eps_w_arr, h_e))
            d_eps_w = -R / dR
            eps_w_arr += d_eps_w
        if K == 99:
            raise ValueError, 'No convergence'
        sig = ((1.0 - g(eps_w_arr, L_w)) * E * eps_w_arr)[0]
        sig_t.append(sig)
        eps_t.append(u / L)

    p.plot(u_t, sig_t)
p.show()

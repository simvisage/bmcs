'''
Created on May 23, 2017

@author: rch
'''

import numpy as np
import pylab as p


f_t = 2.4  # [MPa]
E = 20000.0  # [MPa]
L = 450.0  # [mm]
h_e_arr = np.array([50, 100, 150, 200], dtype=np.float_)
G_f = 0.09  # [N/mm]
S = 10000.0  # 10.0 * 100.0
colors = ['orange', 'red', 'green', 'blue', 'gray', 'yellow']


def f(w, f_t, G_f):
    '''Softening law'''
    return f_t * np.exp(-f_t / G_f * w)


def F(w, f_t, G_f):
    '''Integral of the softening law'''
    return G_f - G_f * np.exp(-f_t / G_f * w)


l_ch = E * G_f / f_t**2
print('l', l_ch, L)

w_ch = G_f / f_t
w_max = 8.0 * w_ch
w = np.linspace(0, w_max, 100)
e_w = w / l_ch

for h_e, c in zip(h_e_arr, colors):

    L_el = L
    eps_el = [0, f_t / E]
    sig_el = [0, f_t]
    W_el = [0, f_t**2 / 2 / E * S * L]
    U_el = [0, f_t**2 / 2 / E * S * L]

    eps_w = 1 / E * f(e_w * h_e, f_t, G_f) + e_w * h_e / L
    sig_w = f(e_w * h_e, f_t, G_f)
    W_w = 1. / 2. / E * S * L_el * \
        f(e_w * h_e, f_t, G_f)**2 + S * F(e_w * h_e, f_t, G_f)
    U_w = 1. / 2. / E * S * L_el * \
        f(e_w * h_e, f_t, G_f)**2 + 1. / 2. * \
        S * f(e_w * h_e, f_t, G_f) * e_w * h_e

    eps = np.hstack([eps_el, eps_w])
    sig = np.hstack([sig_el, sig_w])
    W = np.hstack([W_el, W_w])
    U = np.hstack([U_el, U_w])

    p.subplot(2, 2, 1)
    p.plot(eps * L, S * sig, lw=3, color=c)
    p.fill_between(eps, 0, sig, facecolor=c, alpha=0.2)
    p.subplot(2, 2, 2)
    p.plot(w, f(w, f_t, G_f), color=c)
    p.fill_between(w, 0, f(w, f_t, G_f), facecolor=c, alpha=0.2)
    p.plot([0, w_ch], [f_t, 0])
    p.subplot(2, 2, 3)
    p.plot(eps, W, lw=3, color=c)
    p.plot(eps, U, lw=3, color=c)
    p.fill_between(eps, U, W, facecolor=c, alpha=0.15)
    p.subplot(2, 2, 4)
    p.plot(eps, W - U, lw=3, color=c)
    p.fill_between(eps, W - U, facecolor=c, alpha=0.15)

p.show()

'''
Created on May 23, 2017


'''

import matplotlib.pyplot as plt
import numpy as np

E = 20000.0
f_t = 2.4
G_f = 0.09

eps_0 = f_t / E
eps_f = G_f / f_t
eps_max = 1.0 * eps_f

# model with exponential softening law
n_T = 100   # number of increments
L = 450.0
n_E = 30

plt.subplot(121)
n_E_list = list(range(4, n_E, 5))
for n_e in n_E_list:  # n: number of elements
    sig_t = [0]
    eps_t = [0]
    L_s = 1. / n_e
    for sig in np.linspace(f_t, 0.1, n_T):
        eps_s = -f_t / G_f * np.log(sig / f_t)
        u_e = sig / E * (L - L_s)
        u_s = L_s * eps_s
        u = u_e + u_s
        sig_t.append(sig)  # store the values of stress level
        eps_t.append(u / L)  # store the values of the average strain
    plt.plot(eps_t, sig_t, label='n=%i' % n_e)
    plt.legend(loc=1)

plt.xlabel('strain')
plt.ylabel('stress')
plt.subplot(122)

# model with linear softening law
for n_e in n_E_list:  # n: number of elements
    eps = np.array([0.0, f_t / E, eps_f / n_e])
    sig = np.array([0.0, f_t, 0.0])
    plt.plot(eps, sig, label='n=%i' % n_e)
    plt.legend(loc=1)

plt.xlabel('strain')
plt.ylabel('stress')
plt.show()

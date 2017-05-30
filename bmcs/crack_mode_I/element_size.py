'''
Created on May 23, 2017


'''

from scipy.optimize import newton
import matplotlib.pyplot as plt
import numpy as np

E = 34000.0
f_t = 4.5
G_f = 0.004

eps_0 = f_t / E
eps_f = G_f / f_t
eps_max = 1.0 * eps_f
eps = np.linspace(0, eps_max, 100)

# model with exponential softening law
n_T = 100   # number of increments
n_E = 100
sig_t = np.zeros((n_T,), dtype=np.float)
eps_t = np.zeros((n_T,), dtype=np.float)

plt.subplot(111)
n_E_list = range(10, n_E, 5)
for n_e in [7.]:  # n_E_list:  # n: number of elements
    sig_t[0] = 0
    eps_t[0] = 0
    for i in range(1, n_T, 1):
        # stress level starting from (ft=E*eps_0) until 0
        sig_i = f_t * (1 - i * 1.0 / n_T * 1.0)
        eps_s = eps_0 - (eps_f - eps_0) * np.log(sig_i / (eps_0 * E))
        # calculating the average strain
        L_e = 1. / n_e
        eps_i = (sig_i / E) + L_e * (eps_s - (sig_i / E))
        sig_t[i] = sig_i  # store the values of stress level
        eps_t[i] = eps_i  # store the values of the average strain
    plt.plot(eps_t, sig_t, label='n=%i' % n_e)
    plt.legend(loc=1)

print eps_t[-1]
plt.xlabel('strain')
plt.ylabel('stress')
plt.show()

# model with linear softening law
for n in range(1, n_E, 1):  # n: number of elements
    eps_m = (1. / n) * eps_f  # calculating the average strain
    # the strain values for plotting (we need only 3 values)
    eps = np.array([0.0, eps_0, eps_m])
    # the strain values for plotting (we need only 3 values)
    sigma = np.array([0.0, E * eps_0, 0.0])
    ax = plt.subplot(122)
    ax.plot(eps, sigma, label='n=%i' % n)
    plt.legend(loc=1)
    ax.set_xlabel('strain')
    ax.set_ylabel('stress')

plt.show()

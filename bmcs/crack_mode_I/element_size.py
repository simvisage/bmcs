'''
Created on May 23, 2017


'''

from scipy.optimize import newton

import matplotlib.pyplot as plt
import numpy as np

E = 1000.0
eps_0 = 0.002
eps_f = 0.02


def w(eps): return 1 - ((eps_0 / eps) *
                        np.exp(-(eps - eps_0) / (eps_f - eps_0)))


m = 100
sigma = np.zeros((m,), dtype=np.float)
eps_m = np.zeros((m,), dtype=np.float)

for n in range(1, 11, 1):

    sigma[0] = 0
    eps_m[0] = 0

    for i in range(1, m, 1):
        sigma_i = E * eps_0 * (1 - i * 1.0 / m * 1.0)

        def f_n(eps_s): return sigma_i - \
            (eps_0 / eps_s) * \
            np.exp(-(eps_s - eps_0) / (eps_f - eps_0)) * E * eps_s
        eps_s = newton(f_n, eps_0, tol=1e-6, maxiter=50)

        eps_m_i = (sigma_i / E) + (1. / n) * (eps_s - (sigma_i / E))

        sigma[i] = sigma_i
        eps_m[i] = eps_m_i

    ax = plt.subplot(121)

    ax.plot(eps_m, sigma, label='n=%i' % n)
    plt.legend(loc=1)
    ax.set_xlabel('strain')
    ax.set_ylabel('stress')

for n in range(1, 11, 1):
    eps_m = (1. / n) * eps_f
    x = np.array([0.0, eps_0, eps_m])
    y = np.array([0.0, E * eps_0, 0.0])

    ax = plt.subplot(122)
    ax.plot(x, y, label='n=%i' % n)

    plt.legend(loc=1)
    ax.set_xlabel('strain')
    ax.set_ylabel('stress')

plt.show()

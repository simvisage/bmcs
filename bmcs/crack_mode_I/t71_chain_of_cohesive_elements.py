import matplotlib.pyplot as plt
import numpy as np
E = 20000.0
f_t = 2.4
G_f = 0.090
eps_f = G_f / f_t
G_f = 0.09
n_E = 30
n_E_list = [5, 10, 15, 20, 25, 30]
# run a loop over the different discretizations
for n_e in n_E_list:  # n: number of elements
    eps = np.array([0.0, f_t / E, eps_f / n_e])
    sig = np.array([0.0, f_t, 0.0])
    plt.plot(eps, sig, label='n=%i' % n_e)
    plt.legend(loc=1)

plt.xlabel('strain')
plt.ylabel('stress')
plt.show()

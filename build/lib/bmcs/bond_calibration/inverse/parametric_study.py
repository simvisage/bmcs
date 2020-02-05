'''
Created on 25.02.2017

@author: Yingxiong
'''
from ibvpy.api import BCDof

from .fem_inverse import TStepper, TLoop
import matplotlib.pyplot as plt
import numpy as np


def constant_bond(tau, w):
    E_f = 170000.
    E_m = 28484.
    A_f = 9. * 1.85
    A_m = 120. * 13. - 9. * 1.85

    p = 1.  # interface perimeter

    # composite stiffness
    s_c = E_f * A_f + E_m * A_m

    F = np.sqrt(2. * w * p * tau * E_f * A_f * E_m * A_m / s_c)
    return F


def linear_bond(k, L, w):
    E_f = 170000.
    E_m = 28484.
    A_f = 9. * 1.85
    A_m = 120. * 13. - 9. * 1.85
    p = 1.  # interface perimeter

    s_c = 1. / (E_f * A_f) + 1. / (E_m * A_m)

    alpha = np.sqrt(p * s_c * k)

    c = w / (np.exp(alpha * L) + np.exp(-alpha * L))

    F = k * c / alpha * (np.exp(alpha * L) - np.exp(-alpha * L))

#     F1 = k / alpha * np.tanh(alpha * L) * w
#
#     print F
#     print F1

    return F

# w, f = np.loadtxt('D:\\1.txt')
# f_delta = np.copy(f)
# from random import sample
# idx = sample(xrange(60), 25)
# f_delta[idx] += 0.005 * np.mean(f_delta) * (np.random.rand(25) - 0.5)
#
# np.savetxt('D:\\2.txt', np.vstack((w, f_delta)))


# w, f_delta = np.loadtxt('D:\\2.txt')
# #
# f1 = constant_bond(40., w)
w = np.linspace(0, 3, 500)
f1 = linear_bond(20., 2000., w)
# # #
plt.plot(w, f1)
# plt.plot(w, f_delta)
# plt.plot(w, f1, '--', lw=2)
# #
plt.show()
#
# print fdfsfsdf

ts = TStepper(L_x=2000.,  # half of speciment length
              n_e_x=100  # number of elements
              )

n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

w_arr = np.linspace(0, 3, 10)

# w_arr = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.5, 2.5, 3.5, 4.5])

# w_arr = np.hstack((0, w_arr))
# w_arr[1] = 1e-5

# pf_arr = constant_bond(40, w_arr)

pf_arr = linear_bond(10., 2000., w_arr)

# pf_arr = np.interp(w_arr, w, f1)

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, regularization=True)

slip, bond = tl.eval()

print(slip, bond)

plt.plot(slip, bond)
plt.legend(loc='best')
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.show()

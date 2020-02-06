'''
Created on 20.03.2018

@author: Yingxiong
'''

from bmcs.bond_calib.inverse.fem_inverse import \
    MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof

import matplotlib.pyplot as plt
import numpy as np


mats = MATSEval(E_m=32701)

d = 16.0
r = d / 2.0
A_f = np.pi * r**2
p = np.pi * d
A_m = (10 * d) ** 2

fets = FETS1D52ULRH(A_m=A_m,
                    A_f=A_f,
                    L_b=p)

ts = TStepper(mats_eval=mats,
              fets_eval=fets,
              L_x=10.0 * d,  # half of specimen length
              n_e_x=50  # number of elements
              )

n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

w_arr = np.hstack(
    (np.linspace(0, 0.015, 13), np.linspace(0.04, 0.1, 31)))

w_exp = np.array([0.0, 0.01, 0.02, 0.05, 0.1], np.float_)
P_exp = np.array([0.0, 13000, 19000, 30000, 44000], np.float_)
pf_arr = np.interp(w_arr, w_exp, P_exp)

plt.plot(w_arr, pf_arr)
plt.show()

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, n_reg=3)

slip, bond = tl.eval()

np.set_printoptions(precision=4)
print('slip')
print([np.array(slip)])
print('bond')
print([np.array(bond)])

plt.plot(slip, bond)
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.show()

'''
Created on 20.03.2018

@author: Yingxiong
'''
from os.path import join

from bmcs.bond_calib.inverse.fem_inverse import MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof

import matplotlib.pyplot as plt
import numpy as np


w_data = [0, 0.01, 0.05, 0.1, 0.2, 0.5,    1, 2,     3,   4,   5,   6, 7, 8]
f_data = [0, 1.0, 1.8, 2.4, 2.6, 2.75, 2.8, 2.7, 2.5, 1.6, 0.7, 0.4, 0.3, 0.2]

w = np.array(w_data, dtype=np.float_)
f = np.array(f_data, dtype=np.float_)
plt.plot(w, f)
plt.show()

n_layers = 1
A_roving = 0.49
r_roving = np.sqrt(A_roving / np.pi)
P_roving = 2 * np.pi * r_roving
s_roving = 8.3
h_section = 40.0
b_section = 100.0
n_roving = 11.0
A_f = n_roving * n_layers * A_roving
A_c = h_section * b_section
A_m = A_c - A_f
#V_f = A_f / A_c
P_b = P_roving * n_layers


mats = MATSEval(E_m=25000,
                E_f=182000)

fets = FETS1D52ULRH(A_m=A_m,
                    A_f=A_f)

ts = TStepper(mats_eval=mats,
              fets_eval=fets,
              L_x=500.,  # half of specimen length
              n_e_x=40  # number of elements
              )

n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

w_arr = np.hstack(
    (np.linspace(0, 0.20, 80),))

# w_arr = np.linspace(0., 4.0, 61)

pf_arr = np.interp(w_arr, w / 2., f) * 1000.

# plt.plot(w_arr, pf_arr)
# plt.show()

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, n_reg=4)

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

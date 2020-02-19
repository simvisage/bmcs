'''
Created on 09.05.2017

@author: Yingxiong
'''
from .fem_inverse_free_end import TStepper, TLoop
from ibvpy.api import BCDof
import matplotlib.pyplot as plt
import numpy as np

ts = TStepper(L_x=12.,  # half of speciment length
              n_e_x=20  # number of elements
              )

n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

# w_arr = np.linspace(0, 5, 101)

# w_arr = np.linspace(0., 0.5, 30)

w_arr = np.linspace(0.04, 0.15, 20)

pf_arr = np.zeros_like(w_arr)

# w, f = np.loadtxt('D:\\loaded.txt')

# pf_arr = np.interp(w_arr, w, f)

w_free, f_free = np.loadtxt('D:\\1_numerical.txt')

# s = open('D:\\1_numerical.txt').read().replace(',', '.')
# import StringIO
# a = np.loadtxt(StringIO.StringIO(s))
#
# f_free = a[:, 0]
# w_free = a[:, 1]
#
# w_free[w_free < 0.] = 0.


plt.plot(w_free, f_free)
plt.show()

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, w_free=w_free,
           f_free=f_free, regularization=True)

slip, bond = tl.eval()

print(slip, bond)

U_record, F_record = np.array(tl.U_record), np.array(tl.F_record)


plt.plot(slip, bond)
plt.legend(loc='best')
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')

plt.figure()
plt.plot(U_record[:, 1], F_record[:, -1], label='free')


plt.show()

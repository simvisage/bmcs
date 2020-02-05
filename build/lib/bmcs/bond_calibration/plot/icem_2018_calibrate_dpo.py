'''
Created on 20.03.2018

@author: Yingxiong
'''
from os.path import join

from ibvpy.api import BCDof

from bmcs.bond_calibration.inverse.fem_inverse import \
    MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
from matresdev.db.simdb.simdb import simdb
import numpy as np


folder = join(simdb.exdata_dir,
              'double_pullout_tests',
              '2018-02-14_DPO_Leipzig',
              )

# import data

# measured data
filenames = ['DPOUC21A', 'DPOUC22A', 'DPOUC23A', 'DPOUC31A']
dm1, fm1 = np.loadtxt(join(folder, 'DPOUC21A.txt'))
dm2, fm2 = np.loadtxt(join(folder,  'DPOUC22A.txt'))
dm3, fm3 = np.loadtxt(join(folder,  'DPOUC23A.txt'))
dm4, fm4 = np.loadtxt(join(folder,  'DPOUC31A.txt'))

# plt.plot(dm1, fm1)
# plt.plot(dm2, fm2)
plt.plot(dm3, fm3)
# plt.plot(dm4, fm4)
plt.show()


# skip the first part where the remaining concrete at the notch is intact
d1 = np.hstack((0, np.linspace(0.135, 8, 100)))
f1 = np.interp(d1, dm1, fm1)

d2 = np.hstack((0, np.linspace(0.155, 8, 100)))
f2 = np.interp(d2, dm2, fm2)

d3 = np.hstack((0, np.linspace(0.135, 8, 100)))
f3 = np.interp(d3, dm3, fm3)


#plt.plot(d1, f1 * 1000.)
# plt.plot(d2, f2)
# plt.plot(d3, f3)

# plt.plot(dm4, fm4)
# plt.ylim(0,)
# plt.show()

mats = MATSEval(E_m=32701)

fets = FETS1D52ULRH(A_m=100. * 15. - 8. * 1.85,
                    A_f=8. * 1.85)

ts = TStepper(mats_eval=mats,
              fets_eval=fets,
              L_x=100.,  # half of specimen length
              n_e_x=20  # number of elements
              )

n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

w_arr = np.hstack(
    (np.linspace(0, 0.15, 13), np.linspace(0.4, 4.2, 31)))

# w_arr = np.linspace(0., 4.0, 61)

pf_arr = np.interp(w_arr, d3 / 2., f3) * 1000.

# plt.plot(w_arr, pf_arr)
# plt.show()

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, n=3)

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

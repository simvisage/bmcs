'''
Created on 26.03.2018

@author: Yingxiong
'''
from os.path import join

from ibvpy.api import BCDof

from bmcs.bond_calibration.inverse.fe_nls_solver_incre \
    import MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
from matresdev.db.simdb.simdb import simdb
import numpy as np

folder = join(simdb.exdata_dir,
              'double_pullout_tests',
              '2018-02-14_DPO_Leipzig',
              )

dm1, fm1 = np.loadtxt(join(folder, 'DPOUC21A.txt'))
dm2, fm2 = np.loadtxt(join(folder,  'DPOUC22A.txt'))
dm3, fm3 = np.loadtxt(join(folder,  'DPOUC23A.txt'))
dm4, fm4 = np.loadtxt(join(folder,  'DPOUC31A.txt'))


# skip the first part where the remaining concrete at the notch is intact
d1 = np.hstack((0, np.linspace(0.135, 8, 100)))
f1 = np.interp(d1, dm1, fm1)

d2 = np.hstack((0, np.linspace(0.155, 8, 100)))
f2 = np.interp(d2, dm2, fm2)

d3 = np.hstack((0, np.linspace(0.135, 8, 100)))
f3 = np.interp(d3, dm3, fm3)


# V1
slip1 = np.array([0.,  0.0312,  0.0812,  0.1313,  0.59,  1.0967,  1.6033,
                  2.11,  2.6167,  3.1233,  3.63,  3.9467,  4.0733,  4.2])
bond1 = np.array([0.,  15.6951,  21.7692,  24.4666,  32.9197,  37.0011,
                  38.9149,  40.6868,  42.2204,  43.973,  45.6608,  46.3805,
                  46.3724,  46.381])

# V2
slip2 = np.array([0.,  0.0312,  0.0812,  0.1313,  0.59,  1.0967,  1.6033,
                  2.11,  2.6167,  3.1233,  3.63,  3.9467,  4.0733,  4.2])
bond2 = np.array([0.,   6.4231,  22.0442,  23.4289,  35.8386,  41.772,
                  43.021,  44.8226,  46.4185,  47.8164,  48.8358,  49.1945,
                  49.0277,  49.2267])

# V3
slip3 = np.array([0.,  0.0312,  0.0812,  0.1313,  0.59,  1.0967,  1.6033,
                  2.11,  2.6167,  3.1233,  3.63,  3.9467,  4.0733,  4.2])
bond3 = np.array([0.,  11.6568,  20.602,  22.6514,  30.7579,  33.2772,
                  35.0215,  36.2112,  36.9413,  37.6049,  38.0808,  38.1766,
                  38.1426,  38.1675])

plt.plot(slip1, bond1)
plt.plot(slip2, bond2)
plt.plot(slip3, bond3)

slip_avg = np.linspace(0, 4, 50)
bond_avg = np.interp(slip_avg, slip1, bond1) / 3. + np.interp(slip_avg,
                                                              slip2, bond2) / 3. + np.interp(slip_avg, slip3, bond3) / 3.

print([slip_avg])
print([bond_avg])

plt.plot(slip_avg, bond_avg, 'k--', lw=3)

plt.show()

mats = MATSEval(E_m=32701)

fets = FETS1D52ULRH(A_m=100. * 15. - 8. * 1.85,
                    A_f=8. * 1.85)

ts = TStepper(fets_eval=fets,
              L_x=100.,  # half of speciment length
              n_e_x=20  # number of elements
              )
n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=4.0)]
tl = TLoop(ts=ts)


def plt_expri(L_x, slip, bond, d, f, label, color):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf, sig_m, sig_f = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color=color, markevery=5)
    plt.plot(d[d <= 8.0] / 2., f[d <= 8.0] * 1000.,
             '--', color=color, label=label)


plt.figure()
plt_expri(100, slip1, bond1, d1, f1, label='V1', color='k')
plt_expri(100, slip2, bond2, d2, f2, label='V2', color='g')
plt_expri(100, slip3, bond3, d3, f3, label='V3', color='b')
plt.show()

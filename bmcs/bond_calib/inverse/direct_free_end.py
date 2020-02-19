'''
Created on 10.05.2017

@author: Yingxiong
'''
import io

from ibvpy.api import BCDof
from scipy.optimize import minimize

from apps.bond_calibration.xcbfe.fe_nls_solver_incre import \
    MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
import numpy as np


s = open('D:\\1.txt').read().replace(',', '.')
a = np.loadtxt(io.StringIO(s))

f_free = a[:, 0]
w_free = a[:, 1]

w_free[w_free < 0.] = 0.

w_arr = np.linspace(0, 0.025, 30)
s_arr = np.interp(w_arr, w_free, f_free)

mat = MATSEval(E_m=49200,
               E_f=29500,
               slip=[0, 1e-6, 0.01, 0.03, 0.06, 0.15],
               # bond=[0.0, 15.5, 51.666666666666664, 77.5, 103.33333333333333,
               # 155.0])
               #                bond=[0.,  21.56011347,   37.07040289,  103.88867176,
               #                      97.4254847,    57.45118206])
               # bond=[0.0, 19.789, 37.79, 107.42, 85.93, 84.82])
               # slip=[0, 1e-6, 0.005, 0.035, 0.065, 0.095, 0.15],
               # bond=[0, 10., 44.36342501,   80., 100., 120., 150.])
               bond=[0., 10.12901536,   39.9247595,    84.22654625,  101.35300195,  134.23784515, 158.97974139])


fet = FETS1D52ULRH(A_m=10000. - 2.2,
                   A_f=2.2)
ts = TStepper(mats_eval=mat,
              fets_eval=fet)
ts.L_x = 12.
ts.n_e_x = 20

n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=0.14)]

tl = TLoop(ts=ts)

U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, 1], F_record[:, n_dof], label='prediction_new')
plt.plot(w_free, f_free, label='experimental')
tl.ts.mats_eval.slip = [0, 1e-6, 0.01, 0.03, 0.06, 0.15]
tl.ts.mats_eval.bond = [
    0.0, 15.5, 51.666666666666664, 77.5, 103.33333333333333, 155.0]
U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
plt.plot(U_record[:, 1], F_record[:, n_dof], label='prediction_original')
plt.legend()
plt.show()

tl.ts.mats_eval.slip = [0, 1e-6, 0.005, 0.035, 0.065, 0.095, 0.15]


def f(x):
    tl.ts.mats_eval.bond = [0.] + x.tolist()
    U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    s_arr2 = np.interp(w_arr, U_record[:, 1], F_record[:, n_dof])
    delta = s_arr2 - s_arr
    return np.sum(delta ** 2)


# np.savetxt('D:\\1_numerical.txt',  np.vstack((
#     U_record[:, 1], F_record[:, n_dofs - 1])))

x0 = [10., 44.36342501,   80., 100., 120., 150.]

res = minimize(f, x0, method='TNC', tol=1e-6,
               bounds=((5, 20), (30, 50), (50, 100), (90, 120), (110, 140), (130, 200)))

print(res.x)
# plt.plot(w_free, f_free, label='experimental')
# plt.legend(loc='best')
#
# plt.show()

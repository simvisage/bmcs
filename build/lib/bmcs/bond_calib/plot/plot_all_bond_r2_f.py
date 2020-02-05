'''
Created on 29.02.2016

@author: Yingxiong
'''
from ibvpy.api import BCDof

from bmcs.bond_calibration.inverse.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
import numpy as np


# 30-v4-r2_f
slip1 = np.array([0.,  0.0643,  0.15,  0.2357,  0.45,  0.9227,  1.4136,
                  1.9045,  2.3955,  2.8864,  3.3773])
bond1 = np.array([0.,  12.625,  29.4584,  46.2918,  48.1391,  56.5632,
                  61.6553,  67.1319,  73.3276,  79.4907,  85.0926])

# 40-v2-r2-f
slip2 = np.array([0.,  0.144,  0.336,  0.815,  1.52,  2.4,  3.28])
bond2 = np.array(
    [0.,  26.0018,  50.6709,  60.7943,  71.4303,  83.3053,  95.0733])

slip2 = np.array([0.,  0.24,  0.815,  1.52,  2.4,  3.28])
bond2 = np.array([0.,  43.3364,  60.7943,  71.4303,  83.3053,  95.0733])

# 40-v3-r2-f
slip3 = np.array([0.,  0.24,  0.815,  1.52,  2.4,  3.28])
bond3 = np.array([0.,   43.6,   60.2941,   71.683,   86.7652,  102.8843])

# 40-v5-r2-f
slip4 = np.array([0.,  0.24,  0.815,  1.52,  2.4,  3.28])
bond4 = np.array([0.,  46.9026,  59.6912,  74.1217,  88.8444,  98.4792])


slip_avg = np.linspace(0, 3.28, 100)
bond_avg = (np.interp(slip_avg, slip1, bond1) + np.interp(slip_avg, slip2, bond2) +
            np.interp(slip_avg, slip3, bond3) + np.interp(slip_avg, slip4, bond4)) / 4.

print([bond_avg])

# plt.plot(slip1, bond1, label='30-v4-r2_f')
# plt.plot(slip2, bond2, '--', label='40-v2-r2-f')
# plt.plot(slip3, bond3, '--', label='40-v3-r2-f')
# plt.plot(slip4, bond4, '--', label='40-v5-r2-f')
# plt.plot(slip_avg, bond_avg, lw=2, label='average')
# plt.legend(ncol=2, loc='best')
# plt.xlabel('slip [mm]')
# plt.ylabel('bond [N/mm]')
# plt.show()


# reproduce
ts = TStepper()
n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=3.28)]
tl = TLoop(ts=ts)
plt.figure()


def plt_expri(L_x, slip, bond, fpath, label, color):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color='k', markevery=5)
    d, f = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x, np.interp(x, d / 2., f * 1000.),
#              '--', color=color, label=label)
    plt.plot(d[d <= 7.] / 2., f[d <= 7.] * 1000.,
             '--', color=color, label=label)

# plt_expri(150, slip1, bond1,
#           'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V4_R2_f.txt', '30-v4-r2_f', 'b')
# plt_expri(200, slip2, bond2,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V2_R2_f.txt', '40-v2-r2_f', 'k')
# plt_expri(200, slip3, bond3,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V3_R2_f.txt', '40-v3-r2_f', 'k')
# plt_expri(200, slip4, bond4,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V5_R2_f.txt', '40-v5-r2_f', 'k')
# plt.plot(0, 0, marker='.', color='k', label='numerical')
# plt.legend(loc='best', ncol=2)
# plt.xlabel('displacement[mm]')
# plt.ylabel('pull-out force[N]')
# plt.ylim(0,)
# plt.show()


# prediction
def predict(L_x, slip, bond):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color='k', markevery=5)


predict(350, slip_avg, bond_avg)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-V1_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='70-v1_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-0-3300SBR-V2_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='70-v2_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-0-3300SBR-V3g_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='70-v3_R2_f')
plt.plot(0, 0, marker='.', color='k', label='prediction')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)


plt.figure()
predict(250, slip_avg, bond_avg)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V3_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v3_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V4_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v4_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V5_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v5_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V6g_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v6g_R2_f')
plt.plot(0, 0, marker='.', color='k', label='prediction')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)

plt.figure()
predict(400, slip_avg, bond_avg)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-80cm-0-3300SBR-V1_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='80-v1_R2_f')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-80cm-0-3300SBR-V2_R2_f.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='80-v2_R2_f')
plt.plot(0, 0, marker='.', color='k', label='prediction')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)
plt.show()

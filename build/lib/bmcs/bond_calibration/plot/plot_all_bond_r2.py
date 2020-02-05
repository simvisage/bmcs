'''
Created on 29.02.2016

@author: Yingxiong
'''
import matplotlib.pyplot as plt
import numpy as np
from cbfe.scratch.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof

# 30-v1-r2
slip1 = np.array([0.,  0.045,  0.105,  0.3,  0.85,  1.4167,  1.9722,
                  2.5278,  3.0833,  3.5])
bond1 = np.array([0.,  17.6348,  41.1478,  44.5731,  51.4566,  55.1305,
                  59.3039,  64.095,  69.2009,  71.7077])

# 30-v3-r2
slip2 = np.array([0.,  0.075,  0.175,  0.35,  0.85,  1.4167,  1.9722,
                  2.5278,  3.0833,  3.5])
bond2 = np.array([0.,  13.1682,  30.7257,  38.291,  43.5131,  49.3085,
                  55.624,  61.5943,  65.208,  68.2165])

# 40-v1-r2
slip3 = np.array([0.,  0.215,  0.79,  1.61,  2.45,  3.29])
bond3 = np.array([0.,  49.8489,  58.966,  63.7347,  72.4276,  81.956])

# 40-v4-r2

slip4 = np.array([0.,  0.12,  0.28,  0.65,  1.3,  2.1,  2.9,  3.5])
bond4 = np.array(
    [0.,  23.8021,  55.5382,  42.1767,  53.0621,  63.584, 69.8933,  76.1544])


slip_avg = np.linspace(0, 3.28, 100)
bond_avg = (np.interp(slip_avg, slip1, bond1) + np.interp(slip_avg, slip2, bond2) +
            np.interp(slip_avg, slip3, bond3) + np.interp(slip_avg, slip4, bond4)) / 4.

print([bond_avg])


plt.plot(slip1, bond1, label='30-v1-r2')
plt.plot(slip2, bond2, label='30-v3-r2')
plt.plot(slip3, bond3, '--', label='40-v1-r2')
plt.plot(slip4, bond4, '--', label='40-v4-r2')
plt.plot(slip_avg, bond_avg, lw=2, label='average')
plt.legend(ncol=2, loc='best')
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.show()


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
    U_record, F_record, tau = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color='k', markevery=5)
    d, f = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x, np.interp(x, d / 2., f * 1000.),
#              '--', color=color, label=label)
    plt.plot(d[d <= 6.56] / 2., f[d <= 6.56] * 1000.,
             '--', color=color, label=label)

plt_expri(150, slip1, bond1,
          'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V1_R2.txt', '30-v1-r2', 'b')
plt_expri(150, slip2, bond2,
          'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V3_R2.txt', '30-v3-r2', 'b')
plt_expri(200, slip3, bond3,
          'D:\\data\\pull_out\\all\\DPO-40cm-V1_R2.txt', '40-v1-r2', 'k')
plt_expri(200, slip4, bond4,
          'D:\\data\\pull_out\\all\\DPO-40cm-V4_R2.txt', '40-v4-r2', 'k')
plt.plot(0, 0, marker='.', color='k', label='numerical')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[N]')
plt.ylim(0,)
plt.xlim(0,)
plt.show()


# prediction
def predict(L_x, slip, bond):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color='k', markevery=5)


predict(300, slip_avg, bond_avg)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V1_R2.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='60-v1_R2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V2_R2.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='60-v2_R2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V3_R2.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='60-v3_R2')
plt.plot(0, 0, marker='.', color='k', label='prediction')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)

plt.figure()
predict(250, slip_avg, bond_avg)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-V1_R2.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v1_R2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-V2_R2.txt',  delimiter=';')
plt.plot(slip_avg, np.interp(
    slip_avg, d / 2., f * 1000.), '--', label='50-v2_R2')
plt.plot(0, 0, marker='.', color='k', label='prediction')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)
plt.show()

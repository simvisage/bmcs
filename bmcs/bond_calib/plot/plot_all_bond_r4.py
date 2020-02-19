'''
Created on 03.02.2016

@author: Yingxiong
'''
import matplotlib.pyplot as plt
import numpy as np
from cbfe.scratch.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof

# 10-v1_r4
slip1 = np.array([0.,  0.087,  0.203,  0.395,  0.9,  1.3158,  1.7368,
                  2.1579,  2.5789,  3.,  3.4211,  3.8421,  4.2632,  4.6842,  5.])
bond1 = np.array([0.,  66.7758,  63.2982,  30.0288,  23.2427,  24.1146,
                  23.411,  22.2785,  21.1775,  20.2859,  19.3556,  18.7234,
                  17.7756,  16.003,  14.8037])

# 10-v2_r4
slip2 = np.array([0.,  0.087,  0.203,  0.395,  0.9,  1.3158,  1.7368,
                  2.1579,  2.5789,  3.,  3.4211,  3.8421,  4.2632,  4.6842,  5.])
bond2 = np.array([0.,  58.6269,  56.7344,  40.4381,  41.9298,  45.2674,
                  43.8398,  42.3467,  43.3514,  38.7607,  30.4322,  26.643,
                  23.938,  22.7784,  21.7454])

# 10-v3_r4
slip3 = np.array([0.,  0.087,  0.203,  0.395,  0.9,  1.3158,  1.7368,
                  2.1579,  2.5789,  3.,  3.4211,  3.8421,  4.2632,  4.6842,  5.])
bond3 = np.array([0.,  68.3241,  63.7992,  44.6511,  38.4701,  38.239,
                  37.4661,  36.0095,  33.3709,  27.6996,  23.5459,  20.8,
                  19.0301,  17.4953,  16.4045])

# 20-v1_r4
slip4 = np.array([0.,  0.1667,  0.3889,  0.6111,  0.8333,  1.1,  1.5,
                  1.9,  2.2368,  2.5526,  2.8684,  3.1842,  3.5,  3.8158,
                  4.1316,  4.4474,  4.7632,  5.])
bond4 = np.array([0.,  41.1773,  58.0326,  53.1771,  39.6873,  37.4619,
                  39.0617,  38.3736,  36.6979,  35.6186,  34.7431,  33.3538,
                  32.4116,  31.4137,  30.5421,  29.4738,  28.1598,  26.6878])

# 20-v2_r4
slip5 = np.array([0.,  0.09,  0.21,  0.375,  0.55,  0.9,  1.3,
                  1.7,  2.0789,  2.3947,  2.7105,  3.0263,  3.3421,  3.6579,
                  3.9737,  4.2895,  4.6053,  4.9211])
bond5 = np.array([0.,  29.5678,  42.5258,  50.7939,  35.2928,  36.5483,
                  38.4572,  40.5057,  39.312,  39.2864,  38.4141,  38.4688,
                  38.4259,  37.9824,  37.2672,  35.9085,  34.1742,  32.3675])
# 20-v3_r4
slip6 = np.array([0.,  0.15,  0.35,  0.6,  0.9,  1.3,  1.7,
                  2.0789,  2.3947,  2.7105,  3.0263,  3.3421,  3.6579,  3.9737,
                  4.2895,  4.6053,  4.9211])
bond6 = np.array([0.,  43.8783,  58.4392,  62.2875,  40.2567,  44.9145,
                  45.3774,  45.278,  44.0557,  43.0492,  42.5949,  42.3297,
                  41.5637,  40.7701,  40.1358,  39.4887,  38.1358])
# 30-v1_r4
slip7 = np.array([0.,  0.135,  0.315,  0.625,  1.1,  1.5,  1.9,
                  2.2769,  2.6462,  3.0154,  3.3846,  3.7538,  4.1231,  4.4])
bond7 = np.array([0.,  26.5963,  62.0581,  55.1268,  46.0377,  46.3377,
                  47.1189,  47.5699,  46.8934,  45.4601,  43.7733,  41.5235,
                  39.0197,  37.7846])

# 30-v2_r4
slip8 = np.array([0.,  0.159,  0.371,  0.665,  1.1,  1.5,  1.9,
                  2.2769,  2.6462,  3.0154,  3.3846,  3.7538,  4.1231,  4.4])
bond8 = np.array([0.,  22.0997,  51.566,  42.1413,  42.7147,  45.1197,
                  47.1615,  48.7817,  49.2332,  46.3459,  47.3256,  47.8547,
                  47.5372,  43.8442])

# 30-v3_r4
slip9 = np.array([0.,  0.24,  0.56,  0.95,  1.3,  1.7,  2.0923,
                  2.4615,  2.8308,  3.2,  3.5692,  3.9385,  4.3077])
bond9 = np.array([0.,  23.3807,  56.0735,  47.0785,  44.6768,  45.7113,
                  47.9654,  44.5799,  45.0322,  43.5071,  44.7903,  45.0199,  45.069])

x = np.linspace(0, 4.3, 500)

# y = (np.interp(x, slip2, bond2) + np.interp(x, slip3, bond3) +
#      np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5)) / 4.

# a = 'D:\\bondlaw4.txt'
# np.savetxt(a, np.vstack((x, y)))

plt.plot(x, np.interp(x, slip1, bond1), 'g--', label='10-v1_r4')
plt.plot(x, np.interp(x, slip2, bond2), 'g--', label='10-v2_r4')
plt.plot(x, np.interp(x, slip3, bond3), 'g--', label='10-v3_r4')
plt.plot(x, (np.interp(x, slip2, bond2) + np.interp(x, slip3, bond3)) /
         2., 'g', label='10-avg', marker='^', markevery=50, lw=2)

plt.plot(x, np.interp(x, slip4, bond4), 'k--', label='20-v1_r4')
plt.plot(x, np.interp(x, slip5, bond5), 'k--', label='20-v2_r4')
plt.plot(x, np.interp(x, slip6, bond6), 'k--', label='20-v3_r4')
plt.plot(x, (np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5) + np.interp(x, slip6, bond6)) /
         3., 'k', label='20-avg', marker='x', markevery=50, lw=2)

plt.plot(x, np.interp(x, slip7, bond7), 'b--', label='30-v1_r4')
plt.plot(x, np.interp(x, slip8, bond8), 'b--', label='30-v2_r4')
plt.plot(x, np.interp(x, slip9, bond9), 'b--', label='30-v3_r4')
plt.plot(x, (np.interp(x, slip7, bond7) + np.interp(x, slip8, bond8) + np.interp(x, slip9, bond9)) /
         3., 'b', label='30-avg', marker='.', markevery=50, lw=2)

y = (np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5) + np.interp(x, slip9, bond9) + np.interp(x, slip2, bond2) +
     np.interp(x, slip3, bond3) + np.interp(x, slip6, bond6) + np.interp(x, slip7, bond7) + np.interp(x, slip8, bond8)) / 8.
plt.plot(x, y, '-r', label='avg-all', lw=2)

# plt.legend(loc='best', ncol=3)
# plt.show()

#
# plt.plot(x, np.interp(x, slip9, bond9), 'r', lw=2, label='20-v3_r3_unloading')
# plt.plot(x, np.interp(x, slip10, bond10), 'k',
#          lw=2, label='15-v1_r3_unloading')
#
# print [x[x <= 1.5]]
# print [y[x <= 1.5]]
#
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.legend(loc='best', ncol=3)
#
# plt.show()

ts = TStepper()
n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=4.4)]
tl = TLoop(ts=ts)

# reproduce
plt.figure()


def plt_expri(L_x, slip, bond, fpath, label, color):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color='k', markevery=5)
    d, f = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x, np.interp(x, d / 2., f * 1000.),
#              '--', color=color, label=label)
    plt.plot(d[d <= 11.] / 2., f[d <= 11.] * 1000.,
             '--', color=color, label=label)


plt_expri(
    50, slip1, bond1, 'D:\\data\\pull_out\\all\\DPO-10cm-V1_R4_f.asc', '10-v1', 'g')
# plt.show()

plt_expri(
    50, slip2, bond2, 'D:\\data\\pull_out\\all\\DPO-10cm-V2_R4_f.asc',
    '10-v2', 'g')
plt_expri(
    50, slip3, bond3, 'D:\\data\\pull_out\\all\\DPO-10cm-V3_R4_f.asc',
    '10-v3', 'g')
plt_expri(
    100, slip4, bond4, 'D:\\data\\pull_out\\all\\DPO-20cm-V1_R4_f.asc',
    '20-v1', 'k')
plt_expri(
    100, slip5, bond5, 'D:\\data\\pull_out\\all\\DPO-20cm-V2_R4_f.asc',
    '20-v2', 'k')
plt_expri(
    100, slip6, bond6, 'D:\\data\\pull_out\\all\\DPO-20cm-V3_R4_f.asc',
    '20-v3', 'k')
plt_expri(
    150, slip7, bond7, 'D:\\data\\pull_out\\all\\DPO-30cm-V1_R4_f.asc',
    '30-v1', 'b')
plt_expri(
    150, slip8, bond8, 'D:\\data\\pull_out\\all\\DPO-30cm-V2_R4_f.asc',
    '30-v2', 'b')
# plt.show()
# plt_expri(
#     150, slip9, bond9, 'D:\\data\\pull_out\\all\\DPO-30cm-V3_R4_f.asc', '30-v3', 'b')
#
#
# plt.plot(0, 0, marker='.', color='k', label='numerical')
# plt.legend(loc='best', ncol=2)
# plt.xlabel('displacement[mm]')
# plt.ylabel('pull-out force[N]')
# plt.ylim(0, )


# prediction


tl.ts.mats_eval.slip = x.tolist()
tl.ts.mats_eval.bond = y.tolist()
#
plt.figure()
tl.ts.L_x = 200.
U_record, F_record, sf_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', label='predicted', markevery=5)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-40cm-V1_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='40-v1')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-40cm-V2_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='40-v2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-40cm-V3_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='40-v3')

plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[N]')
plt.ylim(0,)


plt.figure()
tl.ts.L_x = 250.
U_record, F_record, sf_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', markevery=5, label='predicted')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-V1_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='50-v1')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-V2_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='50-v2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-V3_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='50-v3')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[N]')
plt.ylim(0,)

plt.figure()
tl.ts.L_x = 300.
U_record, F_record, sf_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', label='predicted', markevery=5)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V1_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v1')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V2_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-V3_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v3')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[N]')
plt.ylim(0,)


plt.figure()
tl.ts.L_x = 350.
U_record, F_record, sf_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', label='predicted', markevery=5)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-V1_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='70-v1')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-V2_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='70-v2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-V3_R4_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='70-v3')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[N]')
plt.ylim(0,)

plt.show()
# compare with lorenz

plt.figure()
x = np.linspace(0, 4.0, 1000)
# plt.plot(x, np.interp(x, slip1, bond1), 'k--', label='10-v1_r4')
plt.plot(x, np.interp(x, slip2, bond2) / 9., 'k--', alpha=0.5)
plt.plot(x, np.interp(x, slip3, bond3) / 9., 'k--', alpha=0.5)

plt.plot(x, np.interp(x, slip4, bond4) / 9., 'k--', alpha=0.5)
plt.plot(x, np.interp(x, slip5, bond5) / 9., 'k--', alpha=0.5)
plt.plot(x, np.interp(x, slip6, bond6) / 9., 'k--', alpha=0.5)

plt.plot(x, np.interp(x, slip7, bond7) / 9., 'k--', alpha=0.5)
plt.plot(x, np.interp(x, slip8, bond8) / 9., 'k--', alpha=0.5)
plt.plot(x, np.interp(x, slip9, bond9) / 9., 'k--', alpha=0.5)

y = (np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5) + np.interp(x, slip9, bond9) + np.interp(x, slip2, bond2) +
     np.interp(x, slip3, bond3) + np.interp(x, slip6, bond6) + np.interp(x, slip7, bond7) + np.interp(x, slip8, bond8)) / 8. / 9.
plt.plot(x, y, 'k--', label='DPO', lw=2)


slip1 = np.array([0.0, 0.0167, 0.033399999999999999, 0.050099999999999999, 0.066799999999999998, 0.089999999999999997, 0.10777777777777778, 0.12555555555555556, 0.14333333333333331, 0.16111111111111109, 0.17888888888888888,
                  0.19666666666666666, 0.21444444444444444, 0.23222222222222222, 0.25, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond1 = np.array([0.0, 7.6088333549124725, 15.21766681176379, 22.826500268615117, 30.435333725466453, 26.22680501263744, 21.72889377287015, 20.64915445555593, 7.446138959763085, 6.744324296530344, 6.535446157571006,
                  6.387625831079449, 6.277306042775117, 6.195209036126417, 6.144971305702221, 5.943285240411173, 6.086392458385862, 6.040626787139278, 6.109113424693133, 6.137670292907936, 6.115076842532124, 6.103835908601188, 6.184784029252357])

slip2 = np.array([0.0, 0.022349999999999998, 0.044699999999999997, 0.070000000000000007, 0.090000000000000011, 0.11000000000000001, 0.13, 0.15000000000000002, 0.17000000000000001, 0.19,
                  0.21000000000000002, 0.23000000000000001, 0.25, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond2 = np.array([0.0, 17.701306700132097, 35.402613432976686, 34.41342764904337, 31.208356362179508, 27.370911463157036, 20.4260351641487, 14.72146743080458, 13.359017564362818, 12.862749686231835, 12.439410417802359,
                  11.564533516572686, 11.390762471483313, 11.108035475066938, 10.907086484456359, 10.927076200478599, 11.256443606270846, 11.110249743579576, 11.126353400658614, 11.024596683726228, 11.111594980524512])

slip3 = np.array([0.0, 0.030849999999999999, 0.061699999999999998, 0.12, 0.13999999999999999, 0.16, 0.17999999999999999, 0.20000000000000001, 0.22, 0.23999999999999999, 0.26000000000000001,
                  0.28000000000000003, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond3 = np.array([0.0, 19.531870935846296, 39.06374191200411, 35.13245161327588, 34.86040395302558, 34.118784844044605, 33.164131887624976, 17.570528297506577, 15.64891401963593, 13.45859492092352,
                  11.23505687665982, 10.562053272750111, 10.02457117454648, 9.402856922596214, 8.693647863640045, 8.6230584738389, 8.68140796212051, 8.61385286179438, 8.450586540242105, 8.481673272919226, 8.54978551615499])

slip4 = np.array([0.0, 0.057500000000000002, 0.115, 0.14999999999999999, 0.16666666666666666, 0.18333333333333332, 0.20000000000000001, 0.21666666666666667, 0.23333333333333334, 0.25, 0.26666666666666666,
                  0.28333333333333333, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond4 = np.array([0.0, 15.293392386987824, 31.10578484956067, 29.08624455044272, 28.122932687946694, 25.124475509005038, 22.037427941448023, 21.334682957600144, 8.885563341721895, 8.46724264720299,
                  8.400087115745553, 8.390761710431336, 8.327999814223086, 8.47124979441856, 8.950601800999078, 9.164418152185474, 9.722647682676415, 10.118971686018229, 10.372509959011435, 10.43814261528225, 10.578404516063465])

slip5 = np.array([0.0, 0.047750000000000001, 0.095500000000000002, 0.12, 0.13999999999999999, 0.16, 0.17999999999999999, 0.20000000000000001, 0.22, 0.23999999999999999, 0.26000000000000001,
                  0.28000000000000003, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond5 = np.array([0.0, 15.56219604033105, 31.81953493892064, 26.104446008346834, 23.977774950282, 22.312332862871667, 9.54675528459605, 8.751285405457157, 8.407042658775277, 8.085145115761568,
                  7.863267674383558, 7.80766101866833, 7.823214104431405, 7.8489372473932475, 8.003063589054706, 8.051872262440984, 8.022129065893166, 7.986851855831132, 7.809959004410668, 7.720275331593954, 7.6178569667237515])

slip6 = np.array([0.0, 0.030849999999999999, 0.061699999999999998, 0.089999999999999997, 0.11333333333333333, 0.13666666666666666, 0.15999999999999998, 0.18333333333333332, 0.20666666666666667, 0.22999999999999998,
                  0.2533333333333333, 0.27666666666666662, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond6 = np.array([0.0, 15.680044228140858, 31.36008850780363, 27.051761218813624, 23.671453961241415, 8.427861663021929, 6.643642738255938, 5.978452210173134, 5.422476049188153, 5.100749880111076,
                  5.005226087357627, 4.8558570794714235, 4.784333227890493, 4.919874878109198, 5.476339958388238, 6.077992141937612, 6.542295737296617, 6.903326356820516, 7.2840610394798615, 7.708948778394801, 8.236999765777279])

slip7 = np.array([0.0, 0.044999999999999998, 0.089999999999999997, 0.12, 0.13999999999999999, 0.16, 0.17999999999999999, 0.20000000000000001, 0.22, 0.23999999999999999, 0.26000000000000001,
                  0.28000000000000003, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond7 = np.array([0.0, 15.462963888886003, 31.528284932814625, 32.075213536139664, 32.88795940325237, 33.72689195515675, 31.12751614644016, 29.93821749670961, 11.633678248015123, 10.660910470361458,
                  10.39549975793086, 10.164809287634503, 10.118190243849511, 10.11888897109841, 9.969239565994119, 10.219964838281948, 10.467254855123107, 10.694617095749473, 10.665265059092091, 10.574387510391208, 10.64353546765347])

slip8 = np.array([0.0, 0.052999999999999999, 0.106, 0.14000000000000001, 0.15777777777777779, 0.17555555555555558, 0.19333333333333333, 0.21111111111111111, 0.22888888888888889, 0.24666666666666665,
                  0.26444444444444443, 0.28222222222222221, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond8 = np.array([0.0, 17.52232098843854, 35.04464201114325, 30.393006402216656, 30.06545088665049, 29.73789533868206, 7.154277705110144, 6.524232593913113, 6.252544826713279, 6.121011820770713,
                  5.987578887883472, 5.9051308298037535, 5.885357006730958, 5.7958927251824734, 5.723336602238125, 5.746305887642299, 5.841020272617518, 5.958081494172905, 6.057469247156753, 6.151418224617531, 6.322017709239996])

slip9 = np.array([0.0, 0.068000000000000005, 0.14000000000000001, 0.15777777777777779, 0.17555555555555558, 0.19333333333333333, 0.21111111111111111, 0.22888888888888889, 0.24666666666666665,
                  0.26444444444444443, 0.28222222222222221, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond9 = np.array([0.0, 22.27077327623037, 24.049614815523725, 24.298086737413268, 23.113843536170066, 8.075206037936516, 7.548952177455171, 7.274727342642916, 7.128226105265697, 7.03600514791491,
                  6.9313589955015535, 6.898229278366353, 6.937731981962904, 7.19706785535528, 7.285418198392122, 7.329356973063341, 7.426872276678061, 7.258346772067938, 7.3049539123992435, 7.086839124634319])

slip10 = np.array([0.0, 0.092700000000000005, 0.14000000000000001, 0.15777777777777779, 0.17555555555555558, 0.19333333333333333, 0.21111111111111111, 0.22888888888888889, 0.24666666666666665,
                   0.26444444444444443, 0.28222222222222221, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond10 = np.array([0.0, 33.634056245090775, 27.255419956987847, 26.79563837185104, 9.877309186150079, 9.00290455620587, 8.761121488933817, 8.676251137277644, 8.616412504256395, 8.593444181811302,
                   8.56845218287728, 8.52604742348907, 8.583767657705087, 9.14530840941739, 9.334744679731626, 9.503887533535286, 9.462907944097832, 9.47930585104469, 9.504463943551565, 9.698642630331797])

slip11 = np.array([0.0, 0.126, 0.14999999999999999, 0.16666666666666666, 0.18333333333333332, 0.20000000000000001, 0.21666666666666667, 0.23333333333333334, 0.25, 0.26666666666666666,
                   0.28333333333333333, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond11 = np.array([0.0, 29.582213488721056, 24.680785160160525, 24.270093580851924, 6.7554759181538655, 6.29149983330559, 6.200309402070533, 6.158428430896445, 6.1078024292315725, 6.082726050036531,
                   6.0823689064917135, 6.058499860059061, 6.064142696511199, 6.658856926595552, 6.762527053915721, 6.918224328434968, 7.016693714742273, 7.012744732517623, 6.973047457550533, 7.017714123137747])

slip12 = np.array([0.0, 0.109, 0.13, 0.1488888888888889, 0.16777777777777778, 0.18666666666666665, 0.20555555555555555, 0.22444444444444445, 0.24333333333333332, 0.26222222222222219,
                   0.28111111111111109, 0.29999999999999999, 0.34999999999999998, 0.51428571428571423, 0.6785714285714286, 0.84285714285714286, 1.0071428571428571, 1.1714285714285713, 1.3357142857142859, 1.5])
bond12 = np.array([0.0, 27.745056490147885, 23.870070728610887, 23.2961194362025, 5.900114090993536, 5.40295223684805, 5.345668531187051, 5.346507811486545, 5.329607019516358, 5.368589821415174,
                   5.4406724846731285, 5.471101107257605, 5.671309366906551, 6.180676688601286, 6.65936035760773, 7.062846755793178, 7.079642638774973, 7.193520241104466, 7.241010036327608, 7.343785543861939])

slip_lst = [slip1, slip2, slip3, slip4, slip5, slip6,
            slip7, slip8, slip9, slip10, slip11, slip12]
bond_lst = [bond1, bond2, bond3, bond4, bond5, bond6,
            bond7, bond8, bond9, bond10, bond11, bond12]

# for j in range(12):
#     slip_lst[j] = np.append(slip_lst[j], 5.0)
#     bond_lst[j] = np.append(bond_lst[j], bond_lst[j][-1])

lorenz_avg = np.zeros_like(x)
for i in range(12):
    plt.plot(slip_lst[i], bond_lst[i], 'k', alpha=0.5)
    lorenz_avg += np.interp(x, slip_lst[i], bond_lst[i]) / 12.
plt.plot(x, lorenz_avg, 'k', lw=2, label='Lorenz')
plt.xlabel('slip [mm]')
plt.ylabel('bond per yarn [N/mm]')
plt.legend(loc='best', ncol=2)


# prediction length vs maximum force
#
ts = TStepper()
ts.fets_eval.A_f = 1.85
n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=4.0)]
tl = TLoop(ts=ts)


def predict_max(L_x, slip, bond):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    max_F_idx = np.argmax(F_record[:, n_dof])
    U = np.reshape(U_record[max_F_idx, :], (-1, 2)).T
    slip = U[1] - U[0]
    return F_record[:, n_dof][max_F_idx], slip[0], slip[-1]


def predict_max_limited_slip(L_x, slip, bond, s_limit):
    # the maximum pull-out force for give allowed slip s_limit
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    slip_left = U_record[:, ts.domain.n_active_elems + 1] - U_record[:, 0]
    idx = np.argmin(np.abs(slip_left - s_limit))
    pull_force = F_record[:, n_dof][0:idx + 1]
    return np.amax(pull_force)


L_arr = np.arange(10, 410, 50)

if __name__ == '__main__':

    figure = plt.figure()

    dpo_max = []

    slip_left = []
    slip_right = []
    lorenz_max = []
    for L in L_arr:
        F_max, sl, sr = predict_max(L, x, y)
        dpo_max.append(F_max)
        slip_left.append(sl)
        slip_right.append(sr)
    #     lorenz_max.append(predict_max(L, x, lorenz_avg))

    fig = figure.add_subplot(111)
    fig.plot(L_arr, np.array(dpo_max) / 1.83, label='DPO')
    # fig.plot(L_arr, np.array(lorenz_max) / 1.83, label='Lorenz')
    fig.set_ylim(0, )
    fig.set_xlim(0, )
    fig.set_xlabel('anchorage length [mm]')
    fig.set_ylabel('yarn stress[MPa]')
    plt.legend(loc='best')

    length_arr = np.array([50, 100, 150, 200, 250, 300, 350])
    length_arr = np.repeat(length_arr, 3)
    max_force_arr = np.array([3.114, 3.385, 3.768, 5.346, 5.953, 6.603, 8.194, 9.044, 9.222, 8.729,
                              10.099, 10.228, 12.567, 13.458, 14.013, 16.532, 17.332, 16.204, 15.431, 18.053, 18.624])
    plt.plot(length_arr, max_force_arr * 1000 / 9. / 1.83, 'k.')

    y1, y2 = fig.get_ylim()

    # ax2 = fig.twinx()
    # ax2.set_ylim(0, y2 * 1.83)
    # ax2.set_yticks(np.arange(0, y2 * 1.83, 500))
    # ax2.set_ylabel('Force per yarn [N]')

    ax2 = fig.twinx()
    ax2.plot(L_arr, np.array(slip_left), '--')
    ax2.plot(L_arr, np.array(slip_right), '--')
    ax2.set_ylim(0, 2)

    ax2.set_ylabel('corresponding slips [mm]')

    # plt.figure()
    # plt_expri(20, x, lorenz_avg, '20', 'm')
    # plt_expri(100, x, lorenz_avg, '100', 'y')
    # plt_expri(200, x, lorenz_avg, '200', 'k')
    # plt_expri(250, x, lorenz_avg, '250', 'g')
    # plt_expri(280, x, lorenz_avg, '280', 'c')
    # plt_expri(290, x, lorenz_avg, '290', 'l')
    # plt_expri(300, x, lorenz_avg, '300', 'b')
    # plt_expri(350, x, lorenz_avg, '350', 'r')
    plt.legend()

    plt.show()

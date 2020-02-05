'''
Created on 03.02.2016

@author: Yingxiong
'''
import matplotlib.pyplot as plt
import numpy as np
from cbfe.scratch.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof


# 30-v1g-r3-f
slip1 = np.array([0.0, 0.38461538461538458, 0.89743589743589736, 1.4102564102564101, 1.9230769230769229, 2.4358974358974357, 2.9487179487179485, 3.4615384615384612, 3.974358974358974, 4.4871794871794872,
                  5.0, 5.5128205128205128, 6.0256410256410255, 6.5384615384615383, 7.0512820512820511, 7.5641025641025639, 8.0769230769230766, 8.5897435897435876, 9.1025641025641022, 9.6153846153846132, 10.0])
bond1 = np.array([0.0, 41.131553809206011, 39.936804665067541, 43.088224620622753, 47.650913085297518, 52.479625415305165, 57.394429686101795, 62.251107904045313, 66.738355595565764, 70.767647986595335, 73.873975043302551,
                  76.009215625340516, 77.277633801765163, 76.270142784043799, 73.416598577944114, 69.69994680116541, 64.730320409930115, 58.933638345787244, 52.554181600645506, 45.626326986883868, 39.96573453999463])

# 30-v2-r3-f
slip2 = np.array([0.0, 0.055714285714285716, 0.13, 0.20428571428571429, 0.40500000000000003, 0.95862068965517244, 1.5931034482758619, 2.2275862068965515, 2.8620689655172411, 3.4965517241379307,
                  4.1310344827586203, 4.7655172413793103, 5.3999999999999986, 6.0344827586206886, 6.6689655172413786, 7.3034482758620687, 7.9379310344827569, 8.572413793103447, 9.2068965517241388, 9.841379310344827])
bond2 = np.array([0.0, 15.00385088356251, 35.008985394979184, 55.014119906395834, 43.133431746905352, 42.22129154275072, 46.457189937422498, 52.421832224015986, 58.214528732629347, 63.443110953129597,
                  65.081745830172082, 70.818555581926134, 76.171908502739569, 80.423245044030807, 83.434389826392007, 85.024361743793122, 82.174874390870798, 76.343148188571817, 66.095308161727061, 54.526649351153921])

# 30-v3-r3-f
slip3 = np.array([0.0, 0.061874999999999999, 0.144375, 0.22687499999999999, 0.30937500000000001, 0.72500000000000009, 1.2758620689655173, 1.9103448275862067, 2.5448275862068965, 3.1793103448275857,
                  3.8137931034482757, 4.4482758620689644, 5.0827586206896544, 5.7172413793103445, 6.3517241379310345, 6.9862068965517228, 7.6206896551724128, 8.2551724137931028, 8.8896551724137929, 9.5241379310344811, 10.0])
bond3 = np.array([0.0, 16.27428330083729, 37.973327701953664, 59.672372103070117, 43.400401511742785, 43.234649080448378, 48.701090597581455, 54.736977654692424, 62.048663399413229, 69.370097603432924, 77.083431690155379,
                  83.982393896019687, 88.794804003264773, 92.364321409231025, 92.988748502860574, 90.587882722159577, 85.05512495854191, 80.343909075206099, 75.103929360931232, 66.591754226257905, 62.36344706176229])

# 20-v1-r3-f
slip4 = np.array([0.0, 0.029062499999999998, 0.067812499999999998, 0.1065625, 0.14531250000000001, 0.29999999999999999, 0.75, 1.34375, 1.9354166666666668, 2.5270833333333336,
                  3.1187499999999999, 3.7104166666666667, 4.3020833333333339, 4.8937500000000007, 5.4854166666666675, 6.0770833333333343, 6.6687500000000011, 7.2604166666666679, 7.8520833333333337])
bond4 = np.array([0.0, 16.539040266437738, 38.591093955021357, 60.643147643604991, 55.492896981222152, 33.361147503574301, 41.750737025152148, 45.341604537187422, 53.984990428825952, 62.276659123066786,
                  70.624931767476426, 77.79727324562819, 82.881005240210555, 83.82596383699105, 85.775479050115678, 85.647895874781625, 82.121406801790954, 78.787654614781076, 74.308361153830276])

# 20-v2-r3-f
slip5 = np.array([0.0, 0.025312500000000002, 0.059062500000000004, 0.092812500000000006, 0.12656250000000002, 0.28999999999999998, 0.75, 1.34375, 1.9354166666666668, 2.5270833333333336,
                  3.1187499999999999, 3.7104166666666667, 4.3020833333333339, 4.8937500000000007, 5.4854166666666675, 6.0770833333333343, 6.6687500000000011, 7.2604166666666679, 7.8520833333333337])
bond5 = np.array([0.0, 16.497410959806412, 38.493958906214978, 60.490506852623582, 53.730724862691076, 32.368976701926144, 46.320380964466182, 53.362551389229182, 60.895571700733498, 70.421645193646242,
                  81.283769953964168, 88.218274137431948, 92.648027453513293, 97.858499164610748, 99.266642697732422, 91.860288624493535, 78.662911574914844, 64.495097443305497, 53.318329359069025])

# 40-v1g-r3-f
slip6 = np.array([0.0, 0.10312500000000001, 0.24062500000000003, 0.37812500000000004, 0.515625, 1.2,
                  2.0899999999999999, 3.0099999999999998, 3.9299999999999997, 4.8499999999999996, 5.7699999999999996])
bond6 = np.array([0.0, 16.215369080359672, 37.83586118750587, 59.456353294652146, 45.765004806966708,
                  50.556253886415377, 62.050681804949988, 73.987723333968773, 85.329360221465933, 93.882128428932845, 98.569222498938217])

# 40 - v2-r3-f
slip7 = np.array([0.0, 0.097500000000000003, 0.22750000000000001, 0.35750000000000004, 0.90000000000000002,
                  1.6774999999999998, 2.5874999999999995, 3.4974999999999996, 4.4074999999999998, 5.3174999999999999, 6.0])
bond7 = np.array([0.0, 22.114974636426631, 51.601607484995441, 41.579593821881069, 43.18704500283296,
                  53.926330983495362, 64.65156233749542, 76.183761041281713, 84.190526015931965, 85.507210387417871, 79.24773226829106])

# 40-v3-r3-f
slip8 = np.array([0.0, 0.12875, 0.30041666666666667, 0.4720833333333333, 0.94999999999999996,
                  1.6083333333333334, 2.6416666666666671, 3.6750000000000003, 4.7083333333333339, 5.7416666666666671])
bond8 = np.array([0.0, 24.31047070297997, 56.724431640286639, 45.242535526089632, 42.133997120450289,
                  50.630599117045691, 62.734575404684747, 72.601635705654928, 79.220386157161698, 83.157494310531277])

# 20-v3-r3-f
slip9 = np.array([0.0, 0.16620689655172414, 0.48448275862068972, 1.2327586206896552, 1.6810344827586208, 2.1293103448275863, 2.5775862068965516,
                  3.0258620689655173, 3.4741379310344831, 3.9224137931034484, 4.3706896551724137, 4.818965517241379, 5.2672413793103452, 5.7155172413793105, 6.1637931034482758, 6.5])
bond9 = np.array([0.0, 53.035468221615929, 46.946106690853028, 65.101608913634891, 74.37265092804293, 82.344713623823054, 91.684424071878439, 101.9654741591593,
                  111.012646204703, 120.53795013437013, 126.24306086555609, 130.16179466319494, 133.44300562572602, 136.05871770760615, 138.04480490752258, 138.4520655715642])

# 15-v3-r3-f
slip10 = np.array([0.0, 0.09375, 0.505, 0.90172413793103456, 1.2506896551724138, 1.5996551724137933, 1.9486206896551728, 2.2975862068965522, 2.6465517241379315,
                   2.9955172413793107, 3.34448275862069, 3.6934482758620693, 4.0424137931034485, 4.3913793103448278, 4.7403448275862079, 5.0893103448275863, 5.4382758620689664, 5.7000000000000002])
bond10 = np.array([0.0, 43.05618551913318, 40.888629416715574, 49.321970730383285, 56.158143245133338, 62.245706611484323, 68.251000923721875, 73.545464379399633, 79.032738465995692,
                   84.188949455670524, 87.531858162376921, 91.532666285021264, 96.66808302759236, 100.23305856244875, 103.01090365681807, 103.98920712455558, 104.69444418370917, 105.09318577617957])


x = np.linspace(0, 5.5, 30)

# y = (np.interp(x, slip2, bond2) + np.interp(x, slip3, bond3) +
#      np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5)) / 4.

# a = 'D:\\bondlaw4.txt'
# np.savetxt(a, np.vstack((x, y)))

# normalize to bond per yarn
# bond1 = bond1 / 9.
# bond2 = bond2 / 9.
# bond3 = bond3 / 9.
# bond4 = bond4 / 9.
# bond5 = bond5 / 9.
# bond6 = bond6 / 9.
# bond7 = bond7 / 9.
# bond8 = bond8 / 9.

plt.plot(x, np.interp(x, slip4, bond4), 'k--', label='20-v1_r3')
plt.plot(x, np.interp(x, slip5, bond5), 'k--', label='20-v2_r3')
plt.plot(x, (np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5)) /
         2., 'k', label='20-avg', marker='^', markevery=50, lw=2)

plt.plot(x, np.interp(x, slip1, bond1), 'g--', label='30-v1g_r3')
plt.plot(x, np.interp(x, slip2, bond2), 'g--', label='30-v2_r3')
plt.plot(x, np.interp(x, slip3, bond3), 'g--', label='30-v3_r3')
plt.plot(x, (np.interp(x, slip1, bond1) + np.interp(x, slip2, bond2) +
             np.interp(x, slip3, bond3)) / 3., 'g', label='30-avg', marker='.',
         markevery=50, lw=2)

plt.plot(x, np.interp(x, slip6, bond6), 'b--', label='40-v1g_r3')
plt.plot(x, np.interp(x, slip7, bond7), 'b--', label='40-v2_r3')
plt.plot(x, np.interp(x, slip8, bond8), 'b--', label='40-v3_r3')
plt.plot(x, (np.interp(x, slip6, bond6) + np.interp(x, slip7, bond7) +
             np.interp(x, slip8, bond8)) / 3., 'b', label='40-avg', marker='x',
         markevery=50, lw=2)

y = (np.interp(x, slip4, bond4) + np.interp(x, slip5, bond5) + np.interp(x, slip1, bond1) + np.interp(x, slip2, bond2) +
     np.interp(x, slip3, bond3) + np.interp(x, slip6, bond6) + np.interp(x, slip7, bond7) + np.interp(x, slip8, bond8)) / 8.
plt.plot(x, y, '-r', label='avg-all', lw=2)

# plt.plot(x, np.interp(x, slip9, bond9), 'r', lw=2, label='20-v3_r3_unloading')
# plt.plot(x, np.interp(x, slip10, bond10), 'k',
#          lw=2, label='15-v1_r3_unloading')

print([x])
print([y])

plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.legend(loc='best', ncol=2)

plt.show()

ts = TStepper()
n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=5.5)]
tl = TLoop(ts=ts)

# reproduce
plt.figure()


def plt_expri(L_x, slip, bond, fpath, label, color):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             color='k', markevery=5)
    d, f = np.loadtxt(fpath,  delimiter=';')
#     plt.plot(x, np.interp(x, d / 2., f * 1000.),
#              '--', color=color, label=label)
    plt.plot(d[d <= 11.] / 2., f[d <= 11.] * 1000.,
             '--', color=color, label=label)


# plt_expri(100., slip4, bond4,
#           'D:\\data\\pull_out\\all\\DPO-20cm-0-3300SBR-V1_R3_f.asc', '20-v1', 'r')
# plt_expri(100., slip5, bond5,
#           'D:\\data\\pull_out\\all\\DPO-20cm-0-3300SBR-V2_R3_f.asc', '20-v2', 'r')
# plt_expri(150., slip1, bond1,
#           'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V1g_R3_f.asc', '30-v1', 'g')
# plt_expri(150., slip2, bond2,
#           'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V2_R3_f.asc', '30-v2', 'g')
# plt_expri(150., slip3, bond3,
#           'D:\\data\\pull_out\\all\\DPO-30cm-0-3300SBR-V3_R3_f.asc', '30-v3', 'g')
# plt_expri(200., slip6, bond6,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V1g_R3_f.asc', '40-v1', 'b')
# plt_expri(200., slip7, bond7,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V2_R3_f.asc', '40-v2', 'b')
# plt_expri(200., slip8, bond8,
#           'D:\\data\\pull_out\\all\\DPO-40cm-0-3300SBR-V3_R3_f.asc', '40-v3', 'b')
#
# plt_expri(100., slip9, bond9,
# 'D:\\data\\pull_out\\all\\DPO-20cm-0-3300SBR-V3_R3_f.asc', '20-v3', 'r')
# plt_expri(75., slip10, bond10,
# 'D:\\data\\pull_out\\all\\DPO-15cm-0-3300SBR-V1_R3_f.asc', '15-v1', 'b')
# plt.plot(0, 0, marker='.', color='k', label='numerical')
# plt.legend(loc='best', ncol=2)
# plt.xlabel('displacement[mm]')
# plt.ylabel('pull-out force[N]')
# plt.ylim(0, 20000)


# prediction

tl.ts.mats_eval.slip = x.tolist()
tl.ts.mats_eval.bond = y.tolist()
plt.figure()
tl.ts.L_x = 250.
U_record, F_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', markevery=5, label='predicted')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V1g_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='50-v1')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-50cm-0-3300SBR-V3_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='50-v3')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)

plt.figure()
tl.ts.L_x = 300.
U_record, F_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', label='predicted', markevery=5)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-0-3300SBR-V1g_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v1g')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-0-3300SBR-V2_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v2')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-60cm-0-3300SBR-V3_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='60-v3')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)


plt.figure()
tl.ts.L_x = 350.
U_record, F_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof],
         marker='.', color='k', label='predicted', markevery=5)
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-0-3300SBR-V1g_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='70-v1g')
d, f = np.loadtxt(
    'D:\\data\\pull_out\\all\\DPO-70cm-0-3300SBR-V2_R3_f.asc',  delimiter=';')
plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='70-v2')
plt.legend(loc='best', ncol=2)
plt.xlabel('displacement[mm]')
plt.ylabel('pull-out force[kN]')
plt.ylim(0,)
#
#
# plt.figure()
# tl.ts.L_x = 400.
# U_record, F_record = tl.eval()
# n_dof = 2 * ts.domain.n_active_elems + 1
# plt.plot(U_record[:, n_dof], F_record[:, n_dof],
#          marker='.', color='k', label='predicted', markevery=5)
# d, f = np.loadtxt(
#     'D:\\data\\pull_out\\all\\DPO-80cm-0-3300SBR-V1g_R3_f.asc',  delimiter=';')
# plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='80-v1g')
# d, f = np.loadtxt(
#     'D:\\data\\pull_out\\all\\DPO-80cm-0-3300SBR-V2_R3_f.asc',  delimiter=';')
# plt.plot(x, np.interp(x, d / 2., f * 1000.), '--', label='80-v2')
# plt.legend(loc='best', ncol=2)
# plt.xlabel('displacement[mm]')
# plt.ylabel('pull-out force[kN]')
# plt.ylim(0,)

plt.show()

'''
Created on 13.09.2017

@author: Yingxiong
'''
from .plot_all_bond_r4 import x, y, L_arr, predict_max_limited_slip
from matplotlib import pyplot as plt
import numpy as np

dpo_max_s_limit_0 = []
dpo_max_s_limit_1 = []
dpo_max_s_limit_2 = []
dpo_max_s_limit_3 = []

for L in L_arr:
    dpo_max_s_limit_0.append(predict_max_limited_slip(L, x, y, 0.1))
    dpo_max_s_limit_1.append(predict_max_limited_slip(L, x, y, 0.2))
    dpo_max_s_limit_2.append(predict_max_limited_slip(L, x, y, 0.5))
    dpo_max_s_limit_3.append(predict_max_limited_slip(L, x, y, 100.))


figure = plt.figure()
fig = figure.add_subplot(111)
fig.plot(L_arr, np.array(dpo_max_s_limit_0) / 1.83, label='0.1 mm')
fig.plot(L_arr, np.array(dpo_max_s_limit_1) / 1.83, label='0.2 mm')
fig.plot(L_arr, np.array(dpo_max_s_limit_2) / 1.83, label='0.5 mm')
fig.plot(L_arr, np.array(dpo_max_s_limit_3) / 1.83, label='no_limit')

fig.set_ylim(0, )
fig.set_xlim(0, )
fig.set_xlabel('anchorage length [mm]')
fig.set_ylabel('yarn stress[MPa]')
plt.legend(loc='best')
plt.show()

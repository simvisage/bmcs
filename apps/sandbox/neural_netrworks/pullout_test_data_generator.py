'''
Created on Dec 6, 2018

@author: rch
'''

import os
import os.path

from fontTools.cffLib import CFFFontSet

from bmcs.pullout.pullout_multilinear import \
    PullOutModel
import numpy as np
import pylab as p


home_dir = os.path.expanduser('~')
data_dir = os.path.join(home_dir, 'data')
f_name = 'trainng_data.txt'
file_path = os.path.join(data_dir, f_name)
print('data stored in', file_path)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if __name__ == '__main__':
    po = PullOutModel()
    po.tline.step = 0.01
    po.w_max = 0.6
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')

    L_list = [10, 20, 50, 100]
    A_m_list = np.array(
        [10.0 * distance
         for distance in[10., 20., 30., 40]],
        dtype=np.float_
    )
    A_f_list = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float_)
    tau_list = np.array(['3.0, 4.0, 5.0, 6.0, 7.0, 8.0'])
    s_list = np.array(['0.1, 0.2, 0.3, 0.4, 0.5, 0.6'])

    P_max = []
    w_at_max = []

    for L in L_list:
        for A_m in A_m_list:
            for A_f in A_f_list:   # I assumed ring cross-section
                for tau in tau_list:
                    for s in s_list:

                        print('',)
                        d_f = 2 * ((A_f / np.pi)**0.5)    # diameter
                        po.geometry.L_x = L
                        po.cross_section.set(
                            A_f=A_f, P_b=((np.pi) * (d_f)), A_m=A_m)
                        po.mats_eval.set(s_data=s,
                                         tau_data=tau)
                        po.mats_eval.update_bs_law = True
                        po.run()

                        P_t = po.get_P_t()
                        w_0, w_L = po.get_w_t()
                        m_idx = np.argmax(P_t)
                        P_max.append(P_t[m_idx])
                        w_at_max.append(w_L[m_idx])

    # print(P_max)
#    p.plot(L_list, P_max)
    p.show()
print("L_list=", L_list, '\n', "P_max=", P_max)
print(len(P_max))
np.savetxt(file_path, P_max)

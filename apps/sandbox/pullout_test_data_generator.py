'''
Created on Dec 6, 2018

@author: rch
'''

import os
import os.path

from bmcs.pullout.pullout_multilinear import \
    PullOutModel
import numpy as np
import pylab as p


home_dir = os.path.expanduser('~')                     
data_dir = os.path.join(home_dir, 'data')                  
f_name = 'trainng_data.txt'
file_path = os.path.join(data_dir, f_name)
print('data_dir', data_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if __name__ == '__main__':                             
    po = PullOutModel()                                    
    po.tline.step = 0.01                   
    po.w_max = 0.6
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67 / 9.0, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 1.7',
                     tau_data='0, 70, 0, 0')
    po.mats_eval.update_bs_law = True

    P_max = []
    w_at_max = []
    L_list = [10, 20, 50, 100]
    for L in L_list:
        print('',)
        po.geometry.L_x = L
        po.run()

        P_t = po.get_P_t()
        w_0, w_L = po.get_w_t()

#        p.plot(w_L, P_t)
        m_idx = np.argmax(P_t)             
        P_max.append(P_t[m_idx])             
        w_at_max.append(w_L[m_idx])

#         print(P_max, w_at_max)
#         p.plot(w_L, P_t)
    p.plot(L_list, P_max)
#    p.plot(L_list, P_max)
    p.show()

np.savetxt(file_path, P_max)



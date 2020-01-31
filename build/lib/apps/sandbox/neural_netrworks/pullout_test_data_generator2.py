'''
Created on Dec 6, 2018

@author: rch
'''

import os
import os.path

from fontTools.cffLib import CFFFontSet
from mayavi import mlab

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

w_max = 0.6
po = PullOutModel()
po.tline.step = 0.01
po.w_max = w_max
po.geometry.L_x = 200.0
po.loading_scenario.set(loading_type='monotonic')


def run_test_program(test_program):
    P_max = []
    w_at_max = []
    for tau, s, A_f, A_m, L in test_program:
        print('tau: %g, s: %g, A_f: %g, A_m: %g, L: %g' %
              (tau, s, A_f, A_m, L))
        d_f = 2 * ((A_f / np.pi)**0.5)    # diameter
        po.geometry.L_x = L
        po.cross_section.set(
            A_f=A_f, P_b=((np.pi) * (d_f)), A_m=A_m)
        po.mats_eval.set(s_data='0, %g, %g' % (s, w_max),
                         tau_data='0, %g, %g' % (tau, tau))
        po.mats_eval.update_bs_law = True
        po.run()

        P_t = po.get_P_t()
        w_0, w_L = po.get_w_t()
        m_idx = np.argmax(P_t)
        P_max.append(P_t[m_idx])
        w_at_max.append(w_L[m_idx])
    return (np.array(P_max, dtype=np.float_),
            np.array(w_at_max, dtype=np.float_))


if __name__ == '__main__':

    L = [10, 20, 50]  # , 100]
    A_m = [10.0 * distance
           for distance in[10]]  # ., 20., 30., 40]]
    A_f = [0.5, 1.0, 1.5]  # , 2.0]
    tau = [3.0, 4.0, 5.0]  # , 6.0]  # , 7.0, 8.0]
    s = [0.1]  # , 0.2], #0.3, 0.4, 0.5, 0.6]

    params = tau, s, A_f, A_m, L
    test_grid = np.meshgrid(*params)
    test_grid_shape = test_grid[0].shape
    test_program = np.vstack([p.flatten() for p in test_grid]).T
    print('inputs')
    print(test_program)
    output_vars = run_test_program(test_program)
    print('outputs')
    print(output_vars)

    out_grids = [out.reshape(*test_grid_shape) for out in output_vars]

    print(out_grids[0])

    print(test_grid[0].shape)
    print(out_grids[0].shape)
#    mlab.contour3d(test_grid[0], test_grid[2], test_grid[4], out_grids[0])
    mlab.contour3d(out_grids[0][0, :, :, 0, :])
    mlab.show()
#    p.plot(L_list, P_max)
#    p.show()

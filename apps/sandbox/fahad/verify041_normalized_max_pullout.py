'''
Created on 18.06.2019

@author: fseemab
'''
import time

from ibvpy.fets import FETS2D4Q
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid
from view.window.bmcs_window import BMCSWindow

import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import traits.api as tr

from .verify02_bond_slip_2d_cum_damage import PullOut2D, verify02_quasi_pullout


def verify_normalized_pullout_force():

    ds = 16
    r_steel = ds / 2
    L_x = ds * 5
    r_concrete = 75
    n_x = 10
    n_y = 4
    ax = p.subplot(111)

    f_list = [0, -5, -10, -15, -20]
    for f_lateral in f_list:  # [0, -100]

        print('lateral confining pressure', f_lateral)

        s = verify02_quasi_pullout(f_lateral=f_lateral)
        s.xd_steel.trait_set(coord_min=(0, 0),
                             coord_max=(L_x, r_steel),
                             shape=(n_x, 1)
                             )
        s.xd_concrete.trait_set(coord_min=(0, r_steel),
                                coord_max=(r_steel, r_concrete),
                                shape=(n_x, n_y)
                                )
        s.u_max = 0.5
        s.tline.step = 0.05
        s.run()
        print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
        w = s.get_window()
        w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

        print('P_max', np.max(s.record['Pw'].sim.hist.F_t))
        print('P_end', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))

#         if False:
#             s = verify02_quasi_pullout(f_lateral=f_lateral)
#             s.xd_steel.trait_set(coord_min=(0, 0),
#                                  coord_max=(L_x, r_steel),
#                                  shape=(n_x, 1)
#                                  )
#             s.xd_concrete.trait_set(coord_min=(0, r_steel),
#                                     coord_max=(r_steel, r_concrete),
#                                     shape=(n_x, n_y)
#                                     )
#             s.u_max = 0.5
#             s.tline.step = 0.05
        s.m_steel.trait_set(E=200000, nu=0.3)
        s.m_concrete.trait_set(E=29800, nu=0.3)
        s.m_ifc.trait_set(E_T=12900,
                          E_N=1e9,
                          tau_bar=4.2,  # 4.0,
                          K=11.0, gamma=55,  # 10,
                          c=2.6, S=4.8e-4, r=0.51,
                          m=0.175,
                          algorithmic=False)
        s.f_lateral = f_lateral

        w = s.get_window()
        w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

        s.run()
    p.show()


if __name__ == '__main__':
    verify_normalized_pullout_force()

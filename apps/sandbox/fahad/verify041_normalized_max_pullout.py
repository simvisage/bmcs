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
import traits.api as tr
from .verify02_bond_slip_2d_cum_damage import PullOut2D


def verify_normalized_pullout_force():

    ds = 16
    r_steel = ds / 2
    r_concrete = r_steel * 5
    n_x = 10
    n_y = 4

    f_list = [0, -5, -10, -15, -20]

    for f_lateral in f_list:
        s = PullOut2D(u_max=1)
        s.xd_steel.trait_set(coord_min=(0, 0),
                             coord_max=(ds, r_steel),
                             shape=(n_x, 1)
                             )
        s.xd_concrete.trait_set(coord_min=(0, r_steel),
                                coord_max=(ds, r_concrete),
                                shape=(n_x, n_y)
                                )
        s.m_steel.trait_set(E=200000, nu=0.2)
        s.m_concrete.trait_set(E=28000, nu=0.3)
        s.m_ifc.trait_set(E_T=10000,
                          E_N=1000000,
                          m=0,
                          algorithmic=False)
        s.bc_lateral_pressure.trait_set(var='f', value=0)
        s.run()
        p0 = np.max(s.record['Pw'].sim.hist.F_t)
        print('p0 =', p0)

        s.m_ifc.trait_set(E_T=10000,
                          E_N=1000000,
                          m=0.175,
                          algorithmic=False)
        s.bc_lateral_pressure.trait_set(var='f', value=f_lateral)
        s.tloop.verbose = False
        s.tstep.model_structure_changed = True
        s.run()
        p_lateral = np.max(s.record['Pw'].sim.hist.F_t)
        print('p_lateral =', p_lateral)
        N_p = p_lateral / p0
        print("f = %g -> N_p  = %g" % (f_lateral, N_p))
#         w = s.get_window()
#         w.run()
#         w.configure_traits()


if __name__ == '__main__':
    verify_normalized_pullout_force()

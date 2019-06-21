'''
Created on 18.06.2019

@author: fseemab
'''
import time

from ibvpy.fets import FETS2D4Q
from simulator.xdomain.xdomain_fe_grid import XDomainFEGrid

import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr

from .verify02_bond_slip_2d_cum_damage import PullOut2D


def verify_normalized_pullout_force():

    ds = 16  # mm
    dx = 5 * ds
    r_steel = ds / 2
    r_concrete = r_steel * 5
    n_x = 10

    f = [0, -5, -10, -15, -20]

    s0 = PullOut2D()
#     s0._get_xd_steel(XDomainFEGrid(coord_min=(0, 0),
#                                    coord_max=(dx, r_steel),
#                                    shape=(n_x, 1)))
#     s0._get_xd_concrete(XDomainFEGrid(coord_min=(0, r_steel),
#                                       coord_max=(dx, r_concrete),
#                                       shape=(n_x, 1)))
    s0.m_steel.trait_set(E=200000, nu=0.2)
    s0.m_concrete.trait_set(E=28000, nu=0.3)
    s0.m_ifc.trait_set(E_T=10000,
                       E_N=1000,
                       m=0.0,
                       algorithmic=True)

    p0 = np.max(s0.record['Pw'].sim.hist.F_t)
    print("p0  = %f" % p0)

    s1 = PullOut2D()
    s1.m_steel.trait_set(E=200000, nu=0.2)
    s1.m_concrete.trait_set(E=28000, nu=0.3)
    s1.m_ifc.trait_set(E_T=12900,  # 12900,
                       E_N=100000,
                       m=0.175,
                       algorithmic=True)
    s1.bc_lateral_pressure.trait_set(var='f', value=f[0])
    p1 = np.max(s1.record['Pw'].sim.hist.F_t)
    N_p1 = p1 / p0
    print("p1 = %f" % N_p1)

    s2 = PullOut2D()
    s2.m_steel.trait_set(E=200000, nu=0.2)
    s2.m_concrete.trait_set(E=28000, nu=0.3)
    s2.m_ifc.trait_set(E_T=12900,  # 12900,
                       E_N=100000,
                       m=0.175,
                       algorithmic=True)
    s2.bc_lateral_pressure.trait_set(var='f', value=f[1])
    p2 = np.max(s1.record['Pw'].sim.hist.F_t)
    N_p2 = p2 / p0
    print("p2 = %f" % N_p2)

    s3 = PullOut2D()
    s3.m_steel.trait_set(E=200000, nu=0.2)
    s3.m_concrete.trait_set(E=28000, nu=0.3)
    s3.m_ifc.trait_set(E_T=12900,  # 12900,
                       E_N=100000,
                       m=0.175,
                       algorithmic=True)
    s3.bc_lateral_pressure.trait_set(var='f', value=f[2])
    p3 = np.max(s1.record['Pw'].sim.hist.F_t)
    N_p3 = p3 / p0
    print("p3 = %f" % N_p3)

    s4 = PullOut2D()
    s4.m_steel.trait_set(E=200000, nu=0.2)
    s4.m_concrete.trait_set(E=28000, nu=0.3)
    s4.m_ifc.trait_set(E_T=12900,  # 12900,
                       E_N=100000,
                       m=0.175,
                       algorithmic=True)
    s4.bc_lateral_pressure.trait_set(var='f', value=f[3])
    p4 = np.max(s1.record['Pw'].sim.hist.F_t)
    N_p4 = p4 / p0
    print("p4 = %f" % N_p4)

    s5 = PullOut2D()
    s5.m_steel.trait_set(E=200000, nu=0.2)
    s5.m_concrete.trait_set(E=28000, nu=0.3)
    s5.m_ifc.trait_set(E_T=12900,  # 12900,
                       E_N=100000,
                       m=0.175,
                       algorithmic=True)
    s5.bc_lateral_pressure.trait_set(var='f', value=f[4])
    p5 = np.max(s1.record['Pw'].sim.hist.F_t)
    N_p5 = p5 / p0
    print("p5 = %f" % N_p5)

    P_final = [N_p1, N_p2, N_p3, N_p4, N_p5]
    plt.plot(f, P_final)
    plt.xlabel('Lateral pressure')
    plt.ylabel('Normalized pressure')
    plt.show()

#     s0.tloop.k_max = 1000
#     s0.tloop.verbose = True
#     s0.tline.step = 0.005
#     s0.tstep.fe_domain.serialized_subdomains
#     s0.run()
#     w = s0.get_window()
#     w.run()
#     time.sleep(1)
#     w.configure_traits


if __name__ == '__main__':
    verify_normalized_pullout_force()

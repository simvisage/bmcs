''' 
Created on 30.04.2019 
 
@author: fseemab 
'''
import time


import numpy as np


from apps.verify.bond_cum_damage.pullout_axisym import PullOutAxiSym


def verify_bond_slip_axisym_cum_damage():
    p = 1
    ds = p / np.pi

    #from .mlab_decorators import decorate_figure
    u_max = 0.001  # 2
    #f_max = 30
    L_x = 1
    # get the diameter corresponding to the perimeter equal to one
    r_steel = ds / 2.0
    r_concrete = ds * 5
    n_x = 1

    print('r', r_steel)

    s = PullOutAxiSym(u_max=u_max,
                      n_x=n_x, n_y_concrete=1, n_y_steel=1)
    s.cross_section.trait_set(R_f=r_steel, R_m=r_concrete)
    s.geometry.L_x = L_x
    s.cross_section.trait_set(R_m=r_concrete,
                              R_f=r_steel)
    s.m_steel.trait_set(E=200000, nu=0.2)
    s.m_concrete.trait_set(E=28000, nu=0.3)
    s.m_ifc.trait_set(E_T=10000,  # 12900,
                      E_N=100000,
                      tau_bar=1,  # 4.0,
                      K=0, gamma=0,
                      c=1, S=0.0025, r=1)
    s.tloop.k_max = 1000
    s.tloop.verbose = True
    s.tline.step = 0.5
    s.tstep.fe_domain.serialized_subdomains
    w = s.get_window()
    w.run()
    time.sleep(1)
    w.configure_traits()


if __name__ == '__main__':
    verify_bond_slip_axisym_cum_damage()

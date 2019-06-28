''' 
Created on 30.04.2019 
 
@author: fseemab 
'''
import time
import numpy as np
from apps.verify.bond_cum_damage.pullout_axisym import PullOutAxiSym


def verify_pullout_axisym():
    u_max = 0.001  # 2
    #f_max = 30
    L_b = 10
    # get the diameter corresponding to the perimeter equal to one
    ds = 1 / np.pi
    r_steel = ds / 2.0
    r_concrete = ds * 5

    s = PullOutAxiSym(u_max=u_max,
                      n_x=10, n_y_concrete=10, n_y_steel=2)
    s.cross_section.trait_set(R_f=r_steel, R_m=r_concrete)
    s.geometry.L_x = L_b
    s.cross_section.trait_set(R_m=r_concrete,
                              R_f=r_steel)
    s.m_ifc.trait_set(E_T=10000,  # 12900,
                      tau_bar=1,  # 4.0,
                      K=0, gamma=0,
                      c=1, S=0.0025, r=1)
    s.tloop.k_max = 1000
    s.tloop.verbose = True
    s.tline.step = 0.05
    s.tstep.fe_domain.serialized_subdomains
    w = s.get_window()
#     w.run()
#     time.sleep(5)
    w.configure_traits()


if __name__ == '__main__':
    verify_pullout_axisym()

''' 
Created on 30.04.2019 
 
@author: fseemab 

Model calibration of the pressure sensitive cumulative 
damage model using a pull out test and reproducing the results
of the paper
'''
import time

from apps.verify.bond_cum_damage.verify02_pullout_axisym_bc_check import PullOutAxiSym
import numpy as np


def verify_bond_slip_axisym_cum_damage():
    p = 1
    ds = 14  # p / np.pi

    #from .mlab_decorators import decorate_figure
    u_max = 1  # 2
    #f_max = 30
    L_x = 3 * ds
    # get the diameter corresponding to the perimeter equal to one
    r_steel = ds / 2.0
    r_concrete = ds * 5
    n_x = 30

    print('r', r_steel)

    s = PullOutAxiSym(u_max=u_max,
                      n_x=n_x, n_y_concrete=40, n_y_steel=5)
    s.cross_section.trait_set(R_f=r_steel, R_m=r_concrete)
    s.geometry.L_x = L_x
    s.cross_section.trait_set(R_m=r_concrete,
                              R_f=r_steel)
    s.m_steel.trait_set(E=200000, nu=0.2)
    s.m_concrete.trait_set(E=28000, nu=0.3)
    s.m_ifc.trait_set(E_T=12900,  # 12900,
                      E_N=1e+9,
                      tau_bar=4.2,  # 4.0,
                      K=11.0, gamma=55,
                      c=2.8, S=0.00048, r=0.51)
    s.tloop.k_max = 1000
    s.tloop.verbose = True
    s.tline.step = 0.01
    s.tstep.fe_domain.serialized_subdomains
    w = s.get_window()
    w.run()
    time.sleep(1)
    w.configure_traits()


if __name__ == '__main__':
    verify_bond_slip_axisym_cum_damage()

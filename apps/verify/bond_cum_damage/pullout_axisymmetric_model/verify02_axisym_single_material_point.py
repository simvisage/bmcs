''' 
Created on 30.04.2019 
 
@author: fseemab 
Application of axisymmetry on a single material point to 
reproduce results as depicted in the paper

'''
from apps.verify.bond_cum_damage.pullout_axisym_model import PullOutAxiSym
# from apps.verify.bond_cum_damage.pullout_axisym_model_singlematpoint import PullOutAxiSym
import numpy as np


#from .mlab_decorators import decorate_figure
u_max = 2
dx = 1
ds = 1 / np.pi
r_steel = ds / 2.0
np.pi * r_steel**2
r_concrete = ds * 3
tau_bar = 2.0
E_T = 1000
s_0 = tau_bar / E_T
n_x = 10
L_x = 1

s = PullOutAxiSym(u_max=u_max,
                  n_x=n_x, n_y_concrete=40, n_y_steel=5)
s.cross_section.trait_set(R_f=r_steel, R_m=r_concrete)
s.geometry.L_x = L_x
s.cross_section.trait_set(R_m=r_concrete,
                          R_f=r_steel)
s.m_steel.trait_set(E=200000, nu=0.3)
s.m_concrete.trait_set(E=30000, nu=0.2)
s.m_ifc.trait_set(E_T=12900,
                  tau_bar=4,
                  K=0, gamma=10,
                  c=1, S=0.0025, r=1)
s.u_max = 2
s.tloop.k_max = 1000
s.tloop.verbose = True
s.tline.step = 0.005
s.tstep.fe_domain.serialized_subdomains
w = s.get_window()
w.configure_traits()

'''
Created on 11.10.2019

@author: fseemab
'''
from apps.verify.bond_cum_damage.pullout_axisymmetric_model.pullout_axisym_model import PullOutAxiSym, Geometry, CrossSection
import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import traits.api as tr

f_lateral = 0
ax = p.subplot(111)
ds = 16  # mm
g = Geometry(L_x=ds * 2.5)
c = CrossSection(R_m=56, R_f=ds / 2)
s = PullOutAxiSym(geometry=g,
                  cross_section=c,
                  n_x=30,
                  n_y_concrete=1,
                  n_y_steel=1)
s.f_lateral = f_lateral
s.xd_steel.trait_set(coord_min=(0, 0),
                     coord_max=(g.L_x, c.R_f),
                     shape=(s.n_x, s.n_y_steel)
                     )
s.xd_concrete.trait_set(coord_min=(0, c.R_f),
                        coord_max=(g.L_x,
                                   c.R_m),
                        shape=(s.n_x, s.n_y_concrete)
                        )
s.m_steel.trait_set(E=200000, nu=0.3)
s.m_concrete.trait_set(E=29800, nu=0.3)
s.m_ifc.trait_set(E_T=20958,
                  E_N=1e5,
                  tau_bar=15,
                  K=30, gamma=200,
                  c=2.5, S=0.007, r=0.9,
                  m=0.175,
                  algorithmic=False)

s.u_max = 10
s.tline.step = 0.005
s.tloop.verbose = True
s.run()

print('P_max', np.max(s.record['Pw'].sim.hist.F_t))
print('P_end', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))

s.f_lateral = f_lateral

w = s.get_window()
w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

p.show()

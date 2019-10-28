'''
Created on 30.09.2019

@author: fseemab
'''
from apps.verify.bond_cum_damage.pullout_axisymmetric_model.pullout_axisym_model import PullOutAxiSym, Geometry, CrossSection
import numpy as np
import pylab as p


def verify_pullout_axisym(f_lateral=0.0):
    d_s = 14
    L_x = 5 * d_s
    g = Geometry(L_x=L_x)
    c = CrossSection(R_m=d_s * 10,
                     R_f=d_s / 2,
                     )
    s = PullOutAxiSym(geometry=g,
                      cross_section=c,
                      n_x=30,
                      u_max=0.5
                      )

    s.m_ifc.trait_set(E_T=12900,
                      E_N=1e9,
                      tau_bar=4.2,  # 4.0,
                      K=11.0, gamma=55,  # 10,
                      c=2.6, S=4.8e-4, r=0.51,
                      m=0.3,
                      algorithmic=False)
    s.f_lateral = f_lateral
    s.u_max = 0.3
    s.tloop.k_max = 10000
    s.tloop.verbose = True
    s.tline.step = 0.1  # 0.005
    s.tstep.fe_domain.serialized_subdomains
    return s


if __name__ == '__main__':
    ax = p.subplot(111)
    s = verify_pullout_axisym(f_lateral=-200)
    s.run()
    print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
    print('f_lateral =', s.f_lateral)
    w = s.get_window()
    w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

    # s = verify02_quasi_pullout(f_lateral=-100)
#     s.f_lateral = -10
#     s.tline.step = 0.1
#     s.run()
#     print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
#     print('f_lateral =', s.f_lateral)
#     #w = s.get_window()
#     w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)
    p.show()
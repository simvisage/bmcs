'''
Created on 23.07.2019

@author: fseemab
'''
from apps.verify.bond_cum_damage.pullout2d_model import PullOut2D
import numpy as np
import pylab as p


def verify01_unit_length_test(f_lateral=0.0):
    s = PullOut2D(n_x=10, L_x=1, perimeter=1,
                  r_steel=(1 / 2 * np.pi))  # L_x = 100
    s.m_ifc.trait_set(E_T=10000,
                      E_N=1e9,
                      tau_bar=1,  # 4.0,
                      K=0, gamma=10,  # 10,
                      c=1, S=0.0025, r=1,
                      m=0.0,
                      algorithmic=False)
    s.f_lateral = f_lateral
    s.u_max = 0.01
    s.tloop.k_max = 1000
    s.tloop.verbose = True
    s.tline.step = 0.0005  # 0.005
    s.tline.step = 0.01
    s.tstep.fe_domain.serialized_subdomains
    s.run()
    return s


if __name__ == '__main__':
    ax = p.subplot(111)
    s = verify01_unit_length_test(f_lateral=-0)
    s.run()
    print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
    w = s.get_window()
    w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

    # s = verify02_quasi_pullout(f_lateral=-100)
    s.f_lateral = -0.2
    s.tline.step = 0.005
    s.run()
    print('F', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
    #w = s.get_window()
    w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)
    p.show()

'''
Created on 01.10.2019

@author: fseemab
'''
import numpy as np
import pylab as p

from .pullout_axisym_model import PullOutAxiSym, Geometry, CrossSection


def verify_normalized_pullout_force():

    ax = p.subplot(111)

    f_list = [0]  # -40000, -50000, -60000
    for f_lateral in f_list:  # [0, -100]
        ds = 16
        print('lateral confining pressure', f_lateral)
        g = Geometry(L_x=ds * 5)
        c = CrossSection(R_m=75,
                         R_f=ds / 2)
        s = PullOutAxiSym(geometry=g,
                          cross_section=c,
                          n_x=10,
                          n_y_concrete=1,
                          n_y_steel=1)
        s.tloop.k_max = 1000
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
        s.m_ifc.trait_set(E_T=12900,
                          E_N=1e5,
                          tau_bar=4.2,  # 4.0,
                          K=11, gamma=55,  # 10,
                          c=2.8, S=0.00048, r=0.51,
                          m=0.175,
                          algorithmic=False)

        s.u_max = 0.005
        s.tline.step = 0.1
        s.tloop.verbose = True
        s.run()

        print('P_max', np.max(s.record['Pw'].sim.hist.F_t))
        print('P_end', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))

        s.f_lateral = f_lateral

        w = s.get_window()
        w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

#         result = np.array(s.record['Pw'].sim.hist.F_t,
#                           s.record['slip'].sim.hist.F_t).transpose()
#         np.savetxt("Pullout%s.txt" % f_lateral, result)
    p.show()


if __name__ == '__main__':
    #     abc = open('sigNnew1.txt', 'w')
    #     abc.close()
    verify_normalized_pullout_force()

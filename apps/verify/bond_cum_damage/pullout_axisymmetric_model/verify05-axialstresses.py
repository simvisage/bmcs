'''
Created on 5 Feb 2020

@author: fseemab
'''
# from apps.verify.bond_cum_damage.pullout_axisymmetric_model.pullout_axisym_model import PullOutAxiSym, Geometry, CrossSection
from apps.sandbox.fahad.pulloutaxisymdemomodel import PullOutAxiSym, Geometry, CrossSection
import numpy as np
import pylab as p


def verify_normalized_pullout_force():

    ax = p.subplot(111)

    f_list = [0]
    u_max = 0.1
    # dt_list = [0.01]
    for f_lateral in f_list:  # [0, -100]
        ds = 16
        print('lateral confining pressure', f_lateral)
        g = Geometry(L_x=ds * 5)
        c = CrossSection(R_m=75,
                         R_f=ds / 2)
        s = PullOutAxiSym(geometry=g,
                          cross_section=c,
                          n_x=5,
                          n_y_concrete=2,
                          n_y_steel=2)
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
        s.m_ifc.trait_set(sigma_o=10.0,
                          E_N=50000,
                          E_T=20000,
                          sig_t=5.0,
                          S_N=0.00000001,
                          S_T=0.0000005,
                          c_N=1.2,
                          c_T=2.2,
                          K=0,
                          gamma=0,
                          m=0.2,
                          b=0.2,
                          #=====================================================
                          # E_T=10000,
                          # E_N=1000000,
                          # gamma=55.0,
                          # K=11.0,
                          # tau_bar=4.2,
                          # S=0.005,
                          # r=1.0,
                          # c=2.8,
                          # m=0.175,
                          algorithmic=False)

        s.u_max = u_max
        s.tline.step = 0.1
        s.tloop.verbose = True
        #s.run()

        #=======================================================================
        # print('P_max', np.max(s.record['Pw'].sim.hist.F_t))
        # print('P_end', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
        # print('S_shape', np.array(s.record['stress'].ug.point_data.tensors).shape)
        # print('S_max', np.max(s.record['stress'].ug.point_data.tensors))
        # print('S_min', np.min(s.record['stress'].ug.point_data.tensors))
        # print('S1', np.array(s.record['stress'].ug.point_data.tensors)[0, :])
        #=======================================================================

        s.f_lateral = f_lateral

        w = s.get_window()
        w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)

#         result = np.array(s.record['Pw'].sim.hist.F_t,
#                           s.record['slip'].sim.hist.F_t).transpose()
#         np.savetxt("Pullout%s.txt" % f_lateral, result)
    return w


if __name__ == '__main__':
    # abc = open('sigNm0lp-100tan.txt', 'w')
    # abc.close()
    w = verify_normalized_pullout_force()
    w.configure_traits()
    p.show()

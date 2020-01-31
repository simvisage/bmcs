'''
Created on 01.10.2019

@author: fseemab
'''
import os.path

from ibvpy.mats.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.mats.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from mayavi import mlab
from simulator.demo.mlab_decorators import decorate_figure

import numpy as np
import pylab as p

from .pulloutaxisymdemomodel import \
    PullOutAxiSym, Geometry, CrossSection


dir = os.path.expanduser('~')


def verify_normalized_pullout_force():

    ax = p.subplot(111)

    f_lateral = 0
    dt_list = [0.1]
    for dt in dt_list:  # [0, -100]
        ds = 16
        print('lateral confining pressure', f_lateral)
        g = Geometry(L_x=ds * 5)
        c = CrossSection(R_m=75,
                         R_f=ds / 2)
        s = PullOutAxiSym(geometry=g,
                          cross_section=c,
                          n_x=5,
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
                          m=0.4,  # 0.175,
                          #                           algorithmic=False)
                          )

        s.u_max = 0.5
        s.tline.step = dt
        s.tloop.verbose = True
        s.run()

        print('P_max', np.max(s.record['Pw'].sim.hist.F_t))
        print('P_end', np.sum(s.hist.F_t[-1, s.right_x_s.dofs]))
<<<<<<< Updated upstream
        
        #=======================================================================
        # mlab.options.backend = 'envisage'
        # print(s.hist['stress'])
        # f_stress = mlab.figure()
        # scene = mlab.get_engine().scenes[-1]
        # scene.name = 'stress'
        # stress_viz = Viz3DTensorField(vis3d=s.hist['stress'])
        # stress_viz.setup()
        # stress_viz.warp_vector.filter.scale_factor = 100.0
        # stress_viz.plot(s.tstep.t_n)
        # lut_manager = mlab.colorbar(title='sig_ab', orientation='horizontal', nb_labels=5)
        # # fix the range
        #=======================================================================
=======

        mlab.options.backend = 'envisage'
        print(s.hist['stress'])
        f_stress = mlab.figure()
        scene = mlab.get_engine().scenes[-1]
        scene.name = 'stress'
        stress_viz = Viz3DTensorField(vis3d=s.hist['stress'])
        stress_viz.setup()
        stress_viz.warp_vector.filter.scale_factor = 100.0
        stress_viz.plot(s.tstep.t_n)
        lut_manager = mlab.colorbar(
            title='sig_ab', orientation='horizontal', nb_labels=5)
        # fix the range
>>>>>>> Stashed changes
        # print(s.record['stress'])
        # scalar_lut_manager.data_range = array([0., 1.])
       # lut_manager.data_range = np.array([np.min(s.record['stress']), np.max(s.record['stress'])])

        #======================================================================
        # f_strain = mlab.figure()
        # scene = mlab.get_engine().scenes[-1]
        # scene.name = 'strain'
        # strain_viz = Viz3DTensorField(vis3d=s.hist['strain'])
        # strain_vis = Vis3DTensorField(vis=s.hist['strain'])
        # strain_viz.setup()
        # strain_viz.warp_vector.filter.scale_factor = 100.0
        # strain_viz.plot(s.tstep.t_n)
        # lut_manager = mlab.colorbar(title='eps_ab', orientation='horizontal', nb_labels=5)
        # # fix the range
        # lut_manager.data_range = (np.min(s.record['strain']), np.max(s.record['strain']))
        #======================================================================
        #======================================================================
        #
        # f_damage = mlab.figure()
        # scene = mlab.get_engine().scenes[-1]
        # scene.name = 'damage'
        # damage_viz = Viz3DScalarField(vis3d=s.hist['damage'])
        # damage_viz.setup()
        # damage_viz.lut_manager.use_default_range = True
        # damage_viz.warp_vector.filter.scale_factor = 10.0
        # damage_viz.plot(s.tstep.t_n)
        # damage_viz.plot(0.0)
        #======================================================================

        # decorate_figure(f_stress, stress_viz)  # , 800, [300, 40, 0])
        # decorate_figure(f_strain, strain_viz)  # , 800, [300, 40, 0])
        #======================================================================
        # decorate_figure(f_damage, damage_viz, 800, [300, 40, 0])
        #======================================================================

        s.f_lateral = f_lateral

        w = s.get_window()
        w.viz_sheet.viz2d_dict['Pw'].plot(ax, 1)
    return w


if __name__ == '__main__':
    #     abc = open('sigNnew1.txt', 'w')
    #     abc.close()
    w = verify_normalized_pullout_force()
    # w.configure_traits()
    p.show()
    mlab.show()

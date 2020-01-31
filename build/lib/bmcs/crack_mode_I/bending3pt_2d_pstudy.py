'''
Created on Jun 14, 2018

@author: rch
'''
from ibvpy.mats.mats3D.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from ibvpy.mats.mats3D.viz3d_stress_field import \
    Vis3DStressField, Viz3DStressField
from view.window import BMCSModel, BMCSWindow

from .bending3pt_2d import BendingTestModel, \
    Vis2DCrackBand, Viz2DTA, Viz2DStrainInCrack
import matplotlib.pyplot as plt
from .viz3d_energy import Viz2DEnergy, Vis2DEnergy, Viz2DEnergyReleasePlot


if __name__ == '__main__':
    bt = BendingTestModel(n_e_x=10, n_e_y=30, k_max=1000,
                          mats_eval_type='scalar damage'
                          #mats_eval_type='microplane damage (eeq)'
                          #mats_eval_type='microplane CSD (eeq)'
                          #mats_eval_type='microplane CSD (odf)'
                          )
    L_c = 2.0
    E = 20000.0
    eps_0 = 120.e-6
    #eps_f = 7.0e-3
    f_t = 2.5
    G_f = 0.1
    bt.mats_eval.trait_set(
        stiffness='secant',
        E=E,
        nu=0.2
    )
    bt.mats_eval.omega_fn.trait_set(
        f_t=f_t,
        G_f=G_f,
        L_s=L_c
    )

    bt.w_max = 2.5
    bt.tline.step = 0.02
    bt.cross_section.b = 100
    bt.geometry.trait_set(
        L=900,
        H=110,
        a=10,
        L_c=L_c
    )
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
#    bt.add_viz2d('load function', 'load-time')
    bt.add_viz2d('F-w', 'load-displacement')

    viz2d_Fw = w.viz_sheet.viz2d_dict['load-displacement']
    vis2d_energy = bt.response_traces['energy']
    viz2d_energy_rates = Viz2DEnergyReleasePlot(name='dissipated energy',
                                                vis2d=vis2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_energy_rates)
    w.viz_sheet.monitor_chunk_size = 1

    L_c_list = [0.1]
    G_f_list = [0.1, 0.2]
    n_e_x_li = [15, 20, 25]
    colors = ['blue', 'green', 'orange', 'yellow', 'black']
    bt.mats_eval.omega_fn.trait_set(
        L_s=L_c_list[0]
    )
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for L_c in L_c_list:
        for G_f_i in G_f_list:
            c = colors.pop()
            #    for n_e_x, c in zip(n_e_x_li, colors):
            bt.trait_set(n_e_x=20)
            bt.geometry.trait_set(
                L_c=L_c
            )
            bt.mats_eval.omega_fn.trait_set(
                G_f=G_f_i,
                L_s=L_c
            )
            w.run()
            w.offline = True
            w.finish_event = True
            w.join()
            viz2d_Fw.plot(ax1, 1, color=c,
                          label='L_c_x=%g, G_f=%g' % (L_c, G_f_i))
            viz2d_energy_rates.plot(ax2, 1, color=c,
                                    label='L_c_x=%g, G_f=%g' % (L_c, G_f_i))
    plt.show()

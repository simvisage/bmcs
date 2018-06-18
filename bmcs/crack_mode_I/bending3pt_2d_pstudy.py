'''
Created on Jun 14, 2018

@author: rch
'''
from view.window import BMCSModel, BMCSWindow

from bending3pt_2d import BendingTestModel, \
    Vis2DCrackBand, Viz2DTA, Viz2DStrainInCrack
from ibvpy.mats.mats3D.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from ibvpy.mats.mats3D.viz3d_stress_field import \
    Vis3DStressField, Viz3DStressField
import matplotlib.pyplot as plt
from viz3d_energy import Viz2DEnergy, Vis2DEnergy, Viz2DEnergyRatesPlot


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
    G_f = 0.01
    bt.mats_eval.trait_set(
        stiffness='secant',
        E=E,
        nu=0.2
    )
    bt.mats_eval.omega_fn.trait_set(
        E=E,
        f_t=f_t,
        G_f=G_f,
        L_s=L_c
    )

    bt.w_max = 1.0
    bt.tline.step = 0.05
    bt.cross_section.b = 100
    bt.geometry.trait_set(
        L=900,
        H=100,
        a=45,
        L_c=L_c
    )
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
#    bt.add_viz2d('load function', 'load-time')
    bt.add_viz2d('F-w', 'load-displacement')

#     vis2d_energy = Vis2DEnergy(model=bt)
#     viz2d_energy = Viz2DEnergy(name='dissipation', vis2d=vis2d_energy)
#     viz2d_energy_rates = Viz2DEnergyRatesPlot(
#         name='dissipation rate', vis2d=vis2d_energy)
#     vis2d_crack_band = Vis2DCrackBand(model=bt)
#     viz2d_cb_strain = Viz2DStrainInCrack(name='strain in crack',
#                                          vis2d=vis2d_crack_band)
#     viz2d_cb_a = Viz2DTA(name='crack length',
#                          vis2d=vis2d_crack_band)
#     viz2d_cb_dGda = Viz2DdGdA(name='energy release per crack extension',
#                               vis2d=vis2d_energy,
#                               vis2d_cb=vis2d_crack_band)
#    w.viz_sheet.viz2d_list.append(viz2d_energy)
#    w.viz_sheet.viz2d_list.append(viz2d_energy_rates)
#    w.viz_sheet.viz2d_list.append(viz2d_cb_strain)
#    w.viz_sheet.viz2d_list.append(viz2d_cb_a)
#    w.viz_sheet.viz2d_list.append(viz2d_cb_dGda)
    viz2d_Fw = w.viz_sheet.viz2d_dict['load-displacement']
#    vis3d = Vis3DStressField()
#    bt.tloop.response_traces.append(vis3d)
#    bt.tloop.response_traces.append(vis2d_energy)
#    bt.tloop.response_traces.append(vis2d_crack_band)
#    viz3d = Viz3DStressField(vis3d=vis3d)
#    w.viz_sheet.add_viz3d(viz3d)
    w.viz_sheet.monitor_chunk_size = 1

    L_c_list = [0.1, 5]
    n_e_x_li = [15, 20, 25]
    colors = ['blue', 'green', 'orange', 'yellow', 'black']
    ax = plt.subplot(111)
    for L_c, c in zip(L_c_list, colors):
        #    for n_e_x, c in zip(n_e_x_li, colors):
        bt.trait_set(n_e_x=20)
        bt.geometry.trait_set(
            L_c=L_c
        )
        bt.mats_eval.omega_fn.trait_set(
            L_s=L_c
        )
        w.run()
        w.offline = True
        w.finish_event = True
        w.join()
        viz2d_Fw.plot(ax, 1, color=c, label='L_c_x=%g' % L_c)
    plt.show()

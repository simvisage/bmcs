'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from reporter import Reporter
from view.window import BMCSWindow

from .pullout_sim import Viz2DPullOutFW, Viz2DPullOutField, \
    Viz2DEnergyPlot, Viz2DEnergyReleasePlot, \
    PullOutModel, PulloutRecord


def run_pullout_fatigue(*args, **kw):
    po = PullOutModel(n_e_x=200, k_max=500, control_variable='f', w_max=1)
    po.tline.step = 0.001
    po.tloop.k_max = 1000
    po.geometry.L_x = 82.0
    po.loading_scenario.set(loading_type='cyclic')
    po.loading_scenario.set(number_of_cycles=20)
    po.loading_scenario.set(maximum_loading=10000)  # 19200)
    po.loading_scenario.set(unloading_ratio=0.1)
    po.loading_scenario.set(amplitude_type="constant")
    po.loading_scenario.set(loading_range="non-symmetric")
    po.cross_section.set(A_f=153.9, P_b=44, A_m=15400.0)

    po.mats_eval_type = 'cumulative fatigue'
    po.mats_eval.trait_set(
        E_b=12900,
        tau_pi_bar=4.2,
        K=11.0,
        gamma=55.0,
        S=4.8e-4,
        r=0.51,
        c=8.8
    )

    po.record['Pw'] = PulloutRecord()
    fw = Viz2DPullOutFW(name='Pw', vis2d=po.hist['Pw'])
    u_p = Viz2DPullOutField(plot_fn='u_p', vis2d=po)
    eps_p = Viz2DPullOutField(plot_fn='eps_p', vis2d=po)
    sig_p = Viz2DPullOutField(plot_fn='sig_p', vis2d=po)
    s = Viz2DPullOutField(plot_fn='s', vis2d=po)
    sf = Viz2DPullOutField(plot_fn='sf', vis2d=po)
    energy = Viz2DEnergyPlot(vis2d=po.hist['Pw'])
    dissipation = Viz2DEnergyReleasePlot(vis2d=po.hist['Pw'])
    w = BMCSWindow(sim=po)
    w.viz_sheet.viz2d_list.append(fw)
    w.viz_sheet.viz2d_list.append(u_p)
    w.viz_sheet.viz2d_list.append(eps_p)
    w.viz_sheet.viz2d_list.append(sig_p)
    w.viz_sheet.viz2d_list.append(s)
    w.viz_sheet.viz2d_list.append(sf)
    w.viz_sheet.viz2d_list.append(energy)
    w.viz_sheet.viz2d_list.append(dissipation)
    w.viz_sheet.monitor_chunk_size = 10

    w.configure_traits(*args, **kw)


def test_reporter():
    po = PullOutModel(n_e_x=100, k_max=500, w_max=1.0)
    po.tline.step = 0.01
    po.geometry.L_x = 500.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function', 'Load-time')
    po.add_viz2d('F-w', 'Load-displacement')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'omega', plot_fn='omega')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    r = Reporter(report_items=[po, w.viz_sheet])
    r.write()
    r.show_tex()
    r.run_pdflatex()
    r.show_pdf()


if __name__ == '__main__':
    run_pullout_fatigue()
    # test_reporter()

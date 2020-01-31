'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from view.window import BMCSWindow

from .pullout_sim import Viz2DPullOutFW, Viz2DPullOutField, \
    PullOutModel, PulloutRecord, Viz2DEnergyPlot, Viz2DEnergyReleasePlot


def run_pullout_frp_damage(*args, **kw):
    po = PullOutModel(n_e_x=100, w_max=0.5)
    po.tline.step = 0.01
    po.geometry.L_x = 200.0
    po.mats_eval_type = 'damage'
    po.mats_eval.omega_fn_type = 'FRP'
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)

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
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_frp_damage()

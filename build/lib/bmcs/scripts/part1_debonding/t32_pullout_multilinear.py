'''
Created on Jul 12, 2017

@author: rch
'''
from bmcs.pullout.pullout_multilinear import PullOutModel
from view.window import BMCSWindow


def construct_pullout_study():
    po = PullOutModel(name='t32_pullout_multilinear',
                      n_e_x=100, k_max=1000, w_max=1.84)
    po.title = 'Pullout with piecewise linear bond-slip law'
    po.desc = '''The pull-out response is calculated for specifying the values 
    explicitly specified bond-slip law.
'''
    po.tline.step = 0.01
    po.geometry.L_x = 200.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.set(s_data='0, 0.1, 0.4, 4.0',
                     tau_data='0, 800, 0, 0')
    po.mats_eval.update_bs_law = True
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    return w


def run_pullout_multilinear(*args, **kw):
    w = construct_pullout_study()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_multilinear()

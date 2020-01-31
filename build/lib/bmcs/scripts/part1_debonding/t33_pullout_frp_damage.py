'''
Created on Jul 12, 2017

@author: rch
'''
from bmcs.pullout.pullout_frp_damage import PullOutModel
from view.window import BMCSWindow


def construct_pullout_study(*args, **kw):
    po = PullOutModel(name='t33_pullout_frp_damage',
                      n_e_x=100, k_max=500, w_max=1.0)
    po.title = 'Pullout with bond damage-softening'
    po.desc = '''This example demonstrates the local nature of debonding
    with propagating damage process zone. Notice the constant 
    energy release rate.
'''
    po.tline.step = 0.01
    po.geometry.L_x = 500.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=16.67, P_b=1.0, A_m=1540.0)
    # po.mats_eval.set()
    po.mats_eval.omega_fn.set(b=10.4, Gf=1.19, plot_max=0.5)
    po.run()

    w = BMCSWindow(model=po)
    po.add_viz2d('load function')
    po.add_viz2d('F-w')
    po.add_viz2d('field', 'u_C', plot_fn='u_C')
    po.add_viz2d('field', 'w', plot_fn='w')
    po.add_viz2d('field', 'eps_C', plot_fn='eps_C')
    po.add_viz2d('field', 's', plot_fn='s')
    po.add_viz2d('field', 'sig_C', plot_fn='sig_C')
    po.add_viz2d('field', 'sf', plot_fn='sf')
    po.add_viz2d('dissipation', 'dissipation')
    po.add_viz2d('dissipation rate', 'dissipation rate')

    return w


def run_pullout_frp_damage(*args, **kw):
    w = construct_pullout_study()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_frp_damage()

'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from .pullout_sim import PullOutModel


def run_pullout_dp(*args, **kw):
    po = PullOutModel(n_e_x=100, k_max=500, w_max=1.5)
    po.sim.tline.step = 0.01
    po.geometry.L_x = 200.0
    po.loading_scenario.trait_set(loading_type='monotonic')
    po.cross_section.trait_set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval_type = 'damage-plasticity'
    po.mats_eval.trait_set(gamma=0.0, K=15.0, tau_bar=45.0)
    po.mats_eval.omega_fn_type = 'li'
    po.mats_eval.omega_fn.trait_set(alpha_2=1.0, plot_max=10.0)
    w = po.get_window()
    w.run()
    w.offline = False
    w.finish_event = True
    w.configure_traits(*args, **kw)


def run_pullout_dp_cyclic():
    po = PullOutModel(n_e_x=200, k_max=500, w_max=2.5,
                      mats_eval_type='damage-plasticity')
    po.sim.tline.step = 0.01
    po.loading_scenario.trait_set(loading_type='cyclic',
                                  amplitude_type='constant',
                                  loading_range='non-symmetric'
                                  )
    po.loading_scenario.trait_set(number_of_cycles=2,
                                  unloading_ratio=0.98,
                                  )
    po.cross_section.trait_set(A_f=16.67, P_b=1.0, A_m=1540.0)
    po.mats_eval.trait_set(gamma=25.0, K=0.0, tau_bar=2.5 * 9.0)
    po.mats_eval.omega_fn_type = 'li'
    po.mats_eval.omega_fn.trait_set(alpha_1=1.0, alpha_2=2, plot_max=2.8)
    w = po.get_window()
    # w.run()

    w.configure_traits()


if __name__ == '__main__':
    # run_pullout_dp()
    run_pullout_dp_cyclic()

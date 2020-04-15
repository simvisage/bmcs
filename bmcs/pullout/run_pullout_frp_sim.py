'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from .pullout_sim import PullOutModel


def run_pullout_frp_damage(*args, **kw):
    po = PullOutModel(n_e_x=500, w_max=2.5)
    po.sim.tline.step = 0.01
    po.geometry.L_x = 3500.0
    po.mats_eval_type = 'damage'
    po.mats_eval.omega_fn_type = 'FRP'
    po.mats_eval.omega_fn.trait_set(B=10.4, Gf=1.19)
    po.loading_scenario.trait_set(loading_type='monotonic')
    po.cross_section.trait_set(A_f=16.67, P_b=1.0, A_m=1540.0)
    w = po.get_window()
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_frp_damage()

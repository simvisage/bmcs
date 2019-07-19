'''
Created on 12.01.2016

@author: ABaktheer, RChudoba, Yingxiong

Test the bond-slip law on a unit surface area using the 
1D pullout model.
'''

from bmcs.api import PullOutModel


def run_pullout_fatigue(*args, **kw):
    po = PullOutModel(n_e_x=1, k_max=500, control_variable='u', w_max=1)
    po.tline.step = 0.5
    po.tloop.k_max = 1000
    po.geometry.L_x = 1.0
    po.loading_scenario.set(loading_type='monotonic')
    po.cross_section.set(A_f=116.0, P_b=1, A_m=400000.0)
    po.mats_eval_type = 'cumulative fatigue'
    po.mats_eval.trait_set(
        E_m=30000,
        E_f=200000,
        E_b=10000,
        tau_pi_bar=1,
        K=0,
        gamma=0,
        c=1,
        S=0.0025,
        r=1
    )
    w = po.get_window()
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_pullout_fatigue()

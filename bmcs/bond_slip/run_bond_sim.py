'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

import time

from .bond_slip_model import BondSlipModel


def run_bond_sim_elasto_plasticity(*args, **kw):
    po = BondSlipModel()
    po.mats_eval_type = 'elasto-plasticity'
    po.mats_eval.trait_set(gamma=0.0, K=15.0, tau_bar=10.0)
    po.tline.step = 0.005
    po.control_variable = 'f'
    po.w_max = 40.0
    po.loading_scenario.loading_type = 'cyclic'
    po.loading_scenario.trait_set(number_of_cycles=3)
    w = po.get_window()
    # w.run()
    w.offline = False
    w.configure_traits(*args, **kw)


def run_bond_sim_damage(*args, **kw):
    po = BondSlipModel()
    po.mats_eval_type = 'damage'
    po.loading_scenario.trait_set(loading_type='cyclic')
    po.loading_scenario.number_of_cycles = 3
    po.mats_eval.omega_fn_type = 'abaqus'
    po.mats_eval.omega_fn.trait_set(s_0=0.1, s_u=30, alpha=0.1, plot_max=1)
    w = po.get_window()
    # w.run()
    w.offline = False
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_sim_damage()
    # run_bond_sim_elasto_plasticity()

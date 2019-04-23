'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''


from bmcs.api import PullOutModel


class BondSlipModel(PullOutModel):

    n_e_x = 1
    w_max = 1

    def __init__(self, *args, **kw):
        super(BondSlipModel, self).__init__(*args, **kw)
        self.tline.step = 0.01
        self.geometry.L_x = 1.0
        self.loading_scenario.trait_set(loading_type='cyclic')
        self.loading_scenario.number_of_cycles = 3
        self.cross_section.trait_set(A_f=1e+5, P_b=1.0, A_m=1e+5)
        self.mats_eval_type = 'damage-plasticity'
        self.mats_eval.trait_set(gamma=0.0, K=15.0, tau_bar=10.0)
        self.mats_eval.omega_fn_type = 'jirasek'
        self.mats_eval.omega_fn.trait_set(s_f=0.1, plot_max=10.0)


def run_bond_sim(*args, **kw):
    po = BondSlipModel()
    w = po.get_window()
    w.run()
    w.offline = False
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_sim()

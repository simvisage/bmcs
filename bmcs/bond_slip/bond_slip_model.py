
from bmcs.pullout.pullout_sim import PullOutModel
from traits.api import on_trait_change


class BondSlipModel(PullOutModel):

    n_e_x = 1
    w_max = 1

    def __init__(self, *args, **kw):
        super(BondSlipModel, self).__init__(*args, **kw)
        self.tline.step = 0.01
        self.geometry.L_x = 1.0
        self.cross_section.trait_set(A_f=1e+5, P_b=1.0, A_m=1e+5)

    @on_trait_change('mats_eval_type')
    def _reset_mats_eval(self):
        self.mats_eval.E_m = 1e+5
        self.mats_eval.E_f = 1e+5

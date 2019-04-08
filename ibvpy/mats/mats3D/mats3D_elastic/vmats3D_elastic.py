
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from simulator.model import Model
from traits.api import \
    Trait, Dict
from traitsui.api import \
    Item, View, VSplit, Group

import numpy as np


class MATS3DElastic(Model, MATS3DEval, MATS3D):
    '''
    Elastic Model.
    Material time-step-evaluator for Scalar-Damage-Model
    '''

    state_var_shapes = {}

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),),
                              ),
                       resizable=True
                       )

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab, tn1):
        '''
        Corrector predictor computation.
        @param eps_Emab input variable - strain tensor
        '''
        sigma_Emab = np.einsum(
            'abcd,...cd->...ab', self.D_abef, eps_Emab
        )
        Em_len = len(eps_Emab.shape) - 2
        new_shape = tuple([1 for _ in range(Em_len)]) + self.D_abef.shape
        D_abef = self.D_abef.reshape(*new_shape)
        return sigma_Emab, D_abef

    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'max_principle_sig': self.get_max_principle_sig,
                'strain_energy': self.get_strain_energy}

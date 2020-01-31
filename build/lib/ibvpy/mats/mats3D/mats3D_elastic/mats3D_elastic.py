
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from traits.api import \
    Trait, Dict
from traitsui.api import \
    Item, View, VSplit, Group

import numpy as np


class MATS3DElastic(MATS3DEval, MATS3D):
    '''
    Elastic Model.
    Material time-step-evaluator for Scalar-Damage-Model
    '''

    state_array_shapes = {}

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

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, algorithmic):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        Em_len = len(eps_Emab_n1.shape) - 2
        new_shape = tuple([1 for i in range(Em_len)]) + self.D_abef.shape
        D_abef = self.D_abef.reshape(*new_shape)
        sigma_Emab = np.einsum(
            '...abcd,...cd->...ab', D_abef, eps_Emab_n1
        )
        return D_abef, sigma_Emab

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #

    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'max_principle_sig': self.get_max_principle_sig,
                'strain_energy': self.get_strain_energy}

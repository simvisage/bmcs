'''
Created on Apr 19, 2019

@author: rch
'''
'''
Created on Apr 19, 2019

@author: rch
'''

import sys

from ibvpy.api import IMATSEval
from ibvpy.mats.mats3D import \
    MATS3DMplDamageODF, MATS3DMplDamageEEQ, MATS3DElastic, \
    MATS3DScalarDamage
from traits.api import \
    Instance, \
    Trait, on_trait_change, Interface, \
    HasStrictTraits


class ICL(Interface):
    pass


class CL1(HasStrictTraits):
    pass


class CL2(HasStrictTraits):
    pass


class TestTrait(HasStrictTraits):
    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type1 = Trait('cl1',
                            {'cl1': CL1,
                             'cl2': CL2,
                             },
                            MAT=True
                            )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval1(self):
        self.mats_eval = self.mats_eval_type1_()

    mats_eval1 = Instance(ICL,
                          MAT=True)
    '''Material model'''

    def _mats_eval1_default(self):
        return self.mats_eval_type1_()

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('microplane damage (eeq)',
                           {'elastic': MATS3DElastic,
                            'microplane damage (eeq)': MATS3DMplDamageEEQ,
                            'microplane damage (odf)': MATS3DMplDamageODF,
                            'scalar damage': MATS3DScalarDamage,
                            },
                           MAT=True
                           )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         MAT=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()


if __name__ == '__main__':
    tt = TestTrait()
    print(tt.mats_eval_type1)
    print(tt.mats_eval_type1_)
    tt.mats_eval_type1 = 'cl2'
    print(tt.mats_eval1)
    print(tt.mats_eval_type)
    print(tt.mats_eval_type_)

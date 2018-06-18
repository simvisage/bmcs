'''
Created on Sep 3, 2009

@author: jakub
'''

from ibvpy.mats.mats2D.mats2D_tensor import \
    map2d_sig_eng_to_mtx, map2d_eps_mtx_to_eng, map2d_sig_mtx_to_eng, \
    map2d_tns4_to_tns2, compliance_mapping2d
from ibvpy.mats.mats2D.mats2D_tensor import map2d_eps_eng_to_mtx
from ibvpy.mats.mats_eval import MATSEval
from traits.api import Callable, Constant
from traitsui.api import View

import numpy as np


class MATS2DEval(MATSEval):

    n_dims = Constant(2)

    # dimension-dependent mappings
    #
    map_tns4_to_tns2 = Callable(map2d_tns4_to_tns2, transient=True)
    map_eps_eng_to_mtx = Callable(map2d_eps_eng_to_mtx, transient=True)
    map_sig_eng_to_mtx = Callable(map2d_sig_eng_to_mtx, transient=True)
    compliance_mapping = Callable(compliance_mapping2d, transient=True)
    map_sig_mtx_to_eng = Callable(map2d_sig_mtx_to_eng, transient=True)
    map_eps_mtx_to_eng = Callable(map2d_eps_mtx_to_eng, transient=True)

    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATS2DEval, self)._get_explorer_config()

        from ibvpy.api import TLine
        from ibvpy.mats.mats2D.mats2D_explorer_bcond import BCDofProportional

        # overload the default configuration
        c['bcond_list'] = [
            BCDofProportional(max_strain=0.00016, alpha_rad=np.pi / 8.0)]
        c['tline'] = TLine(step=0.05, max=1)
        return c

    trait_view = View()

'''
Created on Sep 3, 2009

'''

import copy
from math import pi

from ibvpy.mats.mats3D.mats3D_tensor import \
    map3d_eps_eng_to_mtx, map3d_sig_eng_to_mtx, map3d_eps_mtx_to_eng, map3d_sig_mtx_to_eng, \
    map3d_ijkl2mn, map3d_tns2_to_tns4, map3d_tns4_to_tns2, compliance_mapping3d
from traits.api import Callable, Constant, Property

from ibvpy.mats.matsXD.vmatsXD_eval import MATSXDEval


class MATS3DEval(MATSXDEval):
    '''Base class for 3D models
    '''

    # number of spatial dimensions of an integration cell for the material model
    #
    n_dims = Constant(3)


class NotUsedAnyMore:
    # dimension dependent tensor mappings
    #
    map_tns4_to_tns2 = Callable(map3d_tns4_to_tns2, transient=True)
    map_eps_eng_to_mtx = Callable(map3d_eps_eng_to_mtx, transient=True)
    map_sig_eng_to_mtx = Callable(map3d_sig_eng_to_mtx, transient=True)
    compliance_mapping = Callable(compliance_mapping3d, transient=True)
    map_sig_mtx_to_eng = Callable(map3d_sig_mtx_to_eng, transient=True)
    map_eps_mtx_to_eng = Callable(map3d_eps_mtx_to_eng, transient=True)

    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATS3DEval, self)._get_explorer_config()

        from ibvpy.api import TLine
        from ibvpy.mats.mats3D.mats3D_explorer_bcond import BCDofProportional3D

        # overload the default configuration
        c['bcond_list'] = [
            BCDofProportional3D(max_strain=0.005, phi=0., theta=pi / 2.)]
        c['tline'] = TLine(step=0.05, max=1)
        return c

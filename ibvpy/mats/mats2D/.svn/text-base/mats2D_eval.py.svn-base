'''
Created on Sep 3, 2009

@author: jakub
'''

from enthought.traits.api import Callable, Constant
from ibvpy.mats.mats_eval import MATSEval
from ibvpy.mats.mats2D.mats2D_tensor import map2d_eps_eng_to_mtx

from ibvpy.mats.mats2D.mats2D_tensor import \
    map2d_eps_eng_to_mtx, map2d_sig_eng_to_mtx, map2d_eps_mtx_to_eng, map2d_sig_mtx_to_eng, \
    map2d_ijkl2mn, map2d_tns2_to_tns4, map2d_tns4_to_tns2, compliance_mapping2d

class MATS2DEval( MATSEval ):

    n_dims = Constant( 2 )

    # dimension-dependent mappings
    #
    map_tns4_to_tns2 = Callable( map2d_tns4_to_tns2, transient = True )
    map_eps_eng_to_mtx = Callable( map2d_eps_eng_to_mtx, transient = True )
    map_sig_eng_to_mtx = Callable( map2d_sig_eng_to_mtx, transient = True )
    compliance_mapping = Callable( compliance_mapping2d, transient = True )
    map_sig_mtx_to_eng = Callable( map2d_sig_mtx_to_eng, transient = True )
    map_eps_mtx_to_eng = Callable( map2d_eps_mtx_to_eng, transient = True )

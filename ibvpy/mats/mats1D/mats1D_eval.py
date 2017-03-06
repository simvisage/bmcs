#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 3, 2009 by jakub

from traits.api import Callable, Constant
from ibvpy.mats.mats_eval import MATSEval

def identity_mapping( var ):
    return var

def reshape( var ):
    return var.reshape( ( 1, 1 ) )

def flatten( var ):
    return var.flatten()

class MATS1DEval( MATSEval ):

    n_dims = Constant( 1 )

    # dimension-dependent mappings
    #
    map_tns4_to_tns2 = Callable( identity_mapping, transient = True )
    map_eps_eng_to_mtx = Callable( reshape, transient = True )
    map_sig_eng_to_mtx = Callable( reshape, transient = True )
    compliance_mapping = Callable( identity_mapping, transient = True )
    map_sig_mtx_to_eng = Callable( flatten, transient = True )
    map_eps_mtx_to_eng = Callable( flatten, transient = True )

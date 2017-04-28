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
# Created on Sep 4, 2009 by: rch


from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasStrictTraits, on_trait_change, Event, \
     implements, Dict, Property, cached_property, Delegate, List, WeakRef

from util.traits.either_type import \
    EitherType

from traitsui.api import \
     Item, View, HSplit, VSplit, VGroup, Group, Spring

from ibvpy.api import BCDof, RTrace, TStepper
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.core.sdomain import SDomain
from ibvpy.api import BCDof
from ibvpy.core.ibv_model import IBVModel
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.mats.mats1D.mats1D_damage.mats1D_damage import MATS1DDamage
from ibvpy.mats.mats1D.mats1D_plastic.mats1D_plastic import MATS1DPlastic

from mathkit.mfn import MFnLineArray

from numpy import pi as Pi
from math import cos, sin
from ibvpy.mats.matsXD.matsXD_explore import MATSXDExplore

class MATS1DExplore( MATSXDExplore ):
    '''
    Simulate the loading histories of a material point in 1D space.
    
    Specify the loading scenarios.
    '''
    mats_eval = EitherType( klasses = [ MATS1DElastic, 
                                        MATS1DDamage,
                                        MATS1DPlastic ] )
    def _mats_eval_default(self):
        return MATS1DElastic()

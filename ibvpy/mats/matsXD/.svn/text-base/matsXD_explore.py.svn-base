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

from enthought.traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, \
     Instance, Int, Trait, Range, HasStrictTraits, on_trait_change, Event, \
     implements, Dict, Property, cached_property, Delegate, List, WeakRef, \
     PrototypedFrom, DelegatesTo

from util.traits.either_type import \
    EitherType

from enthought.traits.ui.api import \
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

class MATSXDExplore( HasStrictTraits ):
    '''
    Base class for MATSExplorer dimensional to manage dimensionally 
    dependent presentation of the material models.
    Simulate the loading histories of a material point in 1D space.
        '''
    explorer = WeakRef

    mats_eval = Instance( IMATSEval )
    
    explorer_config = Dict( {} )
    
    def _mats_eval_changed(self):
        if self.explorer_config:
            ec = self.explorer_config
        else:
            ec = self.mats_eval.explorer_config
        mats_eval = ec.get('mats_eval', self.mats_eval )
        if self.explorer == None:
            return

        self.explorer.tloop.tstepper.tse = mats_eval
        self.explorer.tloop.tstepper.sdomain.mats_eval = mats_eval
        
        tl = self.explorer.tloop 
        tl.bcond_list = ec['bcond_list']
        tl.rtrace_list = ec['rtrace_list']
        if ec.has_key('tline'):
            tl.tline = ec['tline']
        tl.reset()
    
    traits_view = View( Item('mats_eval', show_label = False ),
                              resizable = True,
                              width = 1.0,
                              height = 1.0
                              )


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
# Created on Jan 19, 2010 by: rch

from traits.api import Array, Bool, Enum, Float, HasTraits, \
                                 Instance, Int, Trait, Str, Enum, \
                                 Callable, List, TraitDict, Any, Range, \
                                 Delegate, Event, on_trait_change, Button, \
                                 Interface, implements, Property, cached_property
                                 
class ISimModel( Interface ):
    '''
    Interface of SimModel.
    '''
    def get_sim_outputs( self ):
        '''
        '''

    def peval( self ):
        '''
        '''

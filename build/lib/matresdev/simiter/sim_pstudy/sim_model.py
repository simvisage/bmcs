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
                                 Interface, implements, Property, cached_property, \
                                 Self, This

from .i_sim_model import ISimModel
from numpy import array
from .sim_output import SimOut
import time

class SimModel( HasTraits ):
    '''
    The SimModel defines an idealization with a input parameters and
    the ranges of validity. There are three types of parameters:

    numerical floating point
    numerical integer
    categorical (enumerators)
    
    The sim_model only names the parameters of the model without specifying
    the possible levels of their evaluation. This is done using two procedures:
    
    get_input_dict
    get_output_dict
    '''
    implements( ISimModel )
    
    param_1 = Float( 3.6, 
                     unit = 'm',
                     ps_levels = ( 3, 5, 4 ) )
    
    param_2 = Float( 8.6, 
                     unit = 'N',
                     ps_levels = ( 2.4, 8.9, 3 ) )
    
    index_1 = Int( 10,
                   ps_levels = ( 2, 4, 2 ) )
    
    material_model = Callable( pstudy = True, transient = True,
                               ps_levels = [ 'matmod_1', 'matmod_2' ] )
    def _material_model_default( self ):
        return self.matmod_1
    
    def matmod_1( self, param_1, param_2 ):
        return param_1 / 2 + param_2

    def matmod_2( self, param_1, param_2 ):
        return param_1 / 2 - param_2
    
    def get_output_1(self):
        return self.param_1**2 + self.param_2 * self.param_1 / self.index_1 \
                * self.material_model( self.param_1, self.param_2 )

    def get_output_2(self):
        return self.param_1 + self.param_2 * self.param_1**2 / self.index_1**3 \
                * self.material_model( self.param_1, self.param_2 )
    
    def get_sim_outputs( self ):
        return [ SimOut( name = '$\sigma_1$', unit = 'kN' ),
                 SimOut( name = 'output_2', unit = 'kNm' ) ]
    
    def peval(self):
        ''' Return the set of outputs for the current setting of parameters.
        '''
        time.sleep(1)
        return array( [self.get_output_1(), 
                       self.get_output_2() ], 
                       dtype = 'float_' )
        
def run():
    from .sim_pstudy import SimPStudy
    import pickle

    model = SimModel()
    model.index_1 = 8
    tp = open( 'test.pickle', 'w' )
    pickle.dump( model, tp )
    tp.close()
    
    tp = open( 'test.pickle', 'r' )
    model = pickle.load( tp )
    tp.close()
    print('model.index_1', model.index_1)
    

    yse = SimArray( model = model )

    
    yse.configure_traits()
    
if __name__ == '__main__':
    run()                     
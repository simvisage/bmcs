
from enthought.traits.api import \
     Array, Bool, Enum, Float, HasTraits, \
     HasStrictTraits, \
     Instance, Int, Trait, Str, Enum, \
     Callable, List, TraitDict, Any, Range, \
     Delegate, Event, on_trait_change, Button, \
     Interface, Property, cached_property, WeakRef, Dict

from enthought.traits.ui.api import \
     Item, View, HGroup, ListEditor, VGroup, \
     HSplit, Group, Handler, VSplit

from enthought.traits.ui.menu import \
     NoButtons, OKButton, CancelButton, \
     Action
     
from enthought.traits.api import \
    implements

from numpy import zeros, float_

from i_tstepper_eval import ITStepperEval

from rtrace_eval import RTraceEval

class TStepperEval( HasTraits ):
    """
    Base class for time stepper evaluators (TSE).

    Each time stepper classes implement the methods evaluating the
    governing equations of the simulated problem at a discrete time
    instance t. The time-steppers can be chained in order to reflect
    the hierarchical structure of spatial integration starting from
    the domain and using embedded loops over the finite elements,
    layers and material points.

    Furthermore, the time-step-evals specify the variables that can be
    evaluated for the provided spatial context and state array
    (rte_dict) attribute.
    """
    implements( ITStepperEval )

    tstepper = WeakRef( 'ibvpy.core.tstepper.TStepper' )

    def get_state_array_size( self ):
        '''
        Get the array representing the physical state of an object.
        '''
        return 0

    def setup( self, sctx ):
        '''
        Setup the spatial context and state array  
        '''
        pass
    
    def apply_constraints( self, K ):
        '''
        Apply implicity constraints associated with internal mappings
        within the time stepper evaluator.
        
        This method is used by hierarchical representation of the domain
        with local refinements and subdomains.
        '''
        pass
    
    def get_corr_pred( self, sctx, u, tn, tn1 ):
        '''
        Get the corrector and predictor.
        '''
        raise NotImplementedError
    
 
    def identify_parameters(self):
        '''
        Extract the traits that are floating points and can be associated 
        with a statistical distribution.
        '''
        params = {}
        for name, trait in self.traits().items():
            if trait.trait_type.__class__ is Float:
                params[name] = trait
        return params

    def register_mv_pipelines(self,e):
        ''' Register the visualization pipelines in mayavi engine
            (empty by default)
        '''
        pass
        
    # Resp-trace-evals
    #
    rte_dict = Trait( Dict )
    def _rte_dict_default(self):
        return {}


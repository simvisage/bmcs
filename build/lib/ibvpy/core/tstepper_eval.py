
from traits.api import \
    Float, HasTraits, \
    Trait, WeakRef, Dict
from traits.api import \
    provides

from .i_tstepper_eval import ITStepperEval


@provides(ITStepperEval)
class TStepperEval(HasTraits):

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

    tstepper = WeakRef('ibvpy.core.tstepper.TStepper')

    def get_state_array_size(self):
        '''
        Get the array representing the physical state of an object.
        '''
        return 0

    def setup(self, sctx):
        '''
        Setup the spatial context and state array  
        '''
        pass

    def apply_constraints(self, K):
        '''
        Apply implicity constraints associated with internal mappings
        within the time stepper evaluator.

        This method is used by hierarchical representation of the domain
        with local refinements and subdomains.
        '''
        pass

    def get_corr_pred(self, sctx, u, tn, tn1):
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
        for name, trait in list(self.traits().items()):
            if trait.trait_type.__class__ is Float:
                params[name] = trait
        return params

    def register_mv_pipelines(self, e):
        ''' Register the visualization pipelines in mayavi engine
            (empty by default)
        '''
        pass

    # Resp-trace-evals
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {}

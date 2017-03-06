
from traits.api import \
    Array, Bool, Enum, Float, HasTraits, \
    HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, Property, cached_property, WeakRef, Dict

from traitsui.api import \
    Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit

from traitsui.menu import \
    NoButtons, OKButton, CancelButton, \
    Action

from numpy import zeros, float_


class ITStepperEval(Interface):

    """
    Interface for time step evaluators (ITStepperEvalE).

    Each time stepper classes implement the methods evaluating the
    governing equations of the simulated problem at a discrete time
    instance t.
    """

    def get_state_array_size(self):
        """
        Get the size of the state array.

        The state array is really an array of floating point numbers.
        Anything else should be used to represent the physical
        object's state.
        """

    def setup(self, sctx):
        '''
        Setup the state array and spatial context to be operated on.
        '''

    def get_corr_pred(self, sctx, u, tn, tn1):
        '''
        Return the corrector and predictor for supplied control variable.
        '''

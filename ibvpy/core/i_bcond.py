
from traits.api import Array, Bool, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, Property, cached_property


class IBCond(Interface):
    '''
    Interface of the boundary condition.
    '''

    def is_essential(self):
        '''
        Distinguish the essential and natural boundary conditions.

        This is needed to reorganize the system matrices and vectors
        to reflect the prescribed primary variables (displacements).
        '''

    def is_natural(self):
        '''
        Distinguish the essential and natural boundary conditions.

        This is needed to reorganize the system matrices and vectors
        to reflect the prescribed primary variables (displacements).
        '''

#    def get_dofs( self ):
#        '''
#        Return the list of affected DOFs.
#
#        This is needed to reorganize the system matrices and vectors
#        to reflect the prescribed primary variables (displacements).
#        '''

    def setup(self, sctx):
        '''
        Locate the spatial context.
        '''

    def apply_essential(self, K):
        '''
        Locate the spatial context.
        '''

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):
        '''
        Locate the spatial context.
        '''

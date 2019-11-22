from numpy import repeat, arange
from traits.api import \
    HasTraits, provides, \
    List, \
    Property, cached_property

from .i_fe_parent_domain import IFEParentDomain


@provides(IFEParentDomain)
class FEGridActivationMap(HasTraits):

    #-------------------------------------------------------------------------
    # Implement the parent interface
    #-------------------------------------------------------------------------

    # @todo: separate piece of functionality - does not use any parent-child link
    # should be captured in a separate piece of code.
    inactive_elems = List(changed_structure=True)

    def deactivate(self, idx_tuple):
        '''Exclude the specified element from the integration.
        '''
        offset = self.get_cell_offset(idx_tuple)
        self.inactive_elems.append(offset)

    def reactivate(self, idx_tuple):
        '''Include the element in the computation agaoin
        '''
        raise NotImplementedError

    # get boolean array with inactive elements indicated by False
    activation_map = Property(depends_on='inactive_elems,shape')

    @cached_property
    def _get_activation_map(self):
        amap = repeat(True, self.n_grid_elems)
        amap[self.inactive_elems] = False
        return amap

    # get indices of all active elements
    idx_active_elems = Property(depends_on='inactive_elems,shape')

    @cached_property
    def _get_idx_active_elems(self):
        return arange(self.n_grid_elems)[self.activation_map]

    n_active_elems = Property

    def _get_n_active_elems(self):
        if self.inactive_elems != None:
            return self.idx_active_elems.shape[0]
        else:
            return self.shape

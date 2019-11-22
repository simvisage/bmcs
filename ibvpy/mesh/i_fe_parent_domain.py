

from traits.api import Interface, List, Int, Array, Property

class IFEParentDomain( Interface ):
    '''
    Interface of the spatial domain.
    '''

        
    n_active_elems = Property
    
    def deactivate(self, idx_tuple ):
        '''Deactivate an element with the tuple index idx.
        '''
    
    def reactivate(self, idx_tuple ):
        '''Reactivate an element that has previously been deactivated.
        '''
    

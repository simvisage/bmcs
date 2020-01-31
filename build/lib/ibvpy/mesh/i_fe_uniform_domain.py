

from traits.api import \
    Interface, Int, Array, Property


class IFEUniformDomain(Interface):
    '''
    Interface of the spatial domain.
    '''
    n_active_elems = Property(Int)

    elem_dof_map = Array

    elem_X_map = Array

    elem_x_map = Array

    def apply_on_ip_grid(self, fn, ip_mask):
        '''
        Apply the function fn over the first dimension of the array.
        @param fn: function to apply for each ip from ip_mask and each element. 
        @param ip_mask: specifies the local coordinates within the element.     
        '''

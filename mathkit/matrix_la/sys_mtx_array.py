
from numpy import array, where, append
from traits.api import \
    HasTraits, Int, Array, Property, cached_property, List, Trait, Dict, \
    Any


class SysMtxArray(HasTraits):
    '''Class managing an array of equally sized matrices with 
    enumerated equations.

    It is used as an intermediate result of finite element integration
    of the system matrix. From this format, it is converted to a sparse
    matrix representation. For example, the coord format, the conversion 
    is possible by flattening the value array using the ravel.
    (see e.g. coo_mtx) 
    '''
    dof_map_arr = Array('int_')
    mtx_arr = Array('float_')

    n_dofs = Property(depends_on='dof_map_arr')

    @cached_property
    def _get_n_dofs(self):
        if self.dof_map_arr.shape[0] == 0:
            return 0
        return self.dof_map_arr.max() + 1

    def get_dof_ix_array(self, dof):
        '''Return the element number and index of the dof within the element
        '''
        return where(self.dof_map_arr == dof)

    def _zero_rc(self, dof_ix_array):
        '''Set row column values associated with dof a
        to zero.  
        '''
        el_arr, row_arr = dof_ix_array
        for el, i_dof in zip(el_arr, row_arr):
            k_diag = self.mtx_arr[el, i_dof, i_dof]
            self.mtx_arr[el, i_dof, :] = 0.0
            self.mtx_arr[el, :, i_dof] = 0.0

    def _add_col_to_vector(self, dof_ix_array, F, factor):
        '''Get the slice of the a-th column.
        (used for the implementation of the essential boundary conditions)
        '''
        el_arr, row_arr = dof_ix_array
        for el, i_dof in zip(el_arr, row_arr):
            rows = self.dof_map_arr[el]
            F[rows] += factor * self.mtx_arr[el, :, i_dof]

    def _get_diag_elem(self, dof_ix_array):
        '''Get the value of diagonal element at a-ths dof. 
        '''
        K_aa = 0.
        el_arr, row_arr = dof_ix_array
        for el, i_dof in zip(el_arr, row_arr):
            K_aa += self.mtx_arr[el, i_dof, i_dof]
        return K_aa

    def _add_diag_elem(self, dof_ix_array, K_aa):
        '''Get the value of diagonal element at a-ths dof. 
        '''
        el_arr, row_arr = dof_ix_array
        el = el_arr[0]
        i_dof = row_arr[0]
        self.mtx_arr[el, i_dof, i_dof] = K_aa

    def _get_col_subvector(self, dof_ix_array):
        idx_arr = array([], dtype=int)
        val_arr = array([], dtype=float)
        el_arr, row_arr = dof_ix_array
        for el, i_dof in zip(el_arr, row_arr):
            r_dofs = self.dof_map_arr[el]
            idx_arr = append(idx_arr, r_dofs)
            val_arr = append(val_arr, self.mtx_arr[el, :, i_dof])

        return idx_arr, val_arr

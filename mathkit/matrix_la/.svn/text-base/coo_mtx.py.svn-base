
from enthought.traits.api import HasTraits, Array, Property, cached_property, Instance, \
    Delegate, Any
from numpy import allclose, arange, eye, linalg, ones, ix_, array, zeros, \
                hstack, meshgrid, vstack, dot, newaxis, c_, r_, copy, where, \
                ones, concatenate
#from sys_mtx_assembly import SysMtxAssembly
from scipy import sparse
from scipy.sparse.linalg.dsolve import linsolve
from time import time


class COOSparseMtx( HasTraits ):

    assemb = Any

    ij_map = Property( depends_on = 'assemb.+' )
    @cached_property
    def _get_ij_map( self ):
        '''
        Derive the row and column indices of individual values 
        in every element matrix.
        '''

        ij_dof_map_list = []
        # loop over the list of matrix arrays
        for sys_mtx_arr in self.assemb.get_sys_mtx_arrays():

            el_dof_map = sys_mtx_arr.dof_map_arr
            ij_dof_map = zeros( ( el_dof_map.shape[0],
                                 2,
                                 el_dof_map.shape[1] ** 2,
                                 ), dtype = 'int_' )
            for el, dof_map in enumerate( el_dof_map ):
                row_dof_map, col_dof_map = meshgrid( dof_map, dof_map )
                ij_dof_map[el, ...] = vstack( [row_dof_map.flatten(),
                                              col_dof_map.flatten()] )
            ij_dof_map_list.append( ij_dof_map )

        return ij_dof_map_list

    x_l = Property( depends_on = 'el_dof_map' )
    @cached_property
    def _get_x_l( self ):
        '''Helper property to get an array of all row indices'''
        return hstack( [ ij_map[:, 0, :].flatten()
                         for ij_map in self.ij_map ] )

    y_l = Property( depends_on = 'el_dof_map' )
    @cached_property
    def _get_y_l( self ):
        '''Helper property to get an array of all column indices'''
        return hstack( [ ij_map[:, 1, :].flatten()
                         for ij_map in self.ij_map ] )

    data_l = Property
    def _get_data_l( self ):

        return hstack( [ sm_arr.mtx_arr.ravel()
                         for sm_arr in self.assemb.get_sys_mtx_arrays() ] )

    def solve( self, rhs ):
        '''Construct the matrix and use the solver to get 
        the solution for the supplied rhs. 
        '''
        ij = vstack( ( self.x_l, self.y_l ) )

        # Assemble the system matrix from the flattened data and 
        # sparsity map containing two rows - first one are the row
        # indices and second one are the column indices.
        mtx = sparse.coo_matrix( ( self.data_l, ij ) )

        u_vct = linsolve.spsolve( mtx, rhs )
        return u_vct

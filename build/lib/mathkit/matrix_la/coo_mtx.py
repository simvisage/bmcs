
from numpy import zeros, hstack, meshgrid, vstack
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.dsolve import linsolve
from traits.api import HasTraits, Property, cached_property, Any

import numpy as np


class COOSparseMtx(HasTraits):

    assemb = Any

    ij_map = Property(depends_on='assemb.+')

    @cached_property
    def _get_ij_map(self):
        '''
        Derive the row and column indices of individual values 
        in every element matrix.
        '''

        ij_dof_map_list = []
        # loop over the list of matrix arrays
        for sys_mtx_arr in self.assemb.get_sys_mtx_arrays():

            el_dof_map = sys_mtx_arr.dof_map_arr
            ij_dof_map = zeros((el_dof_map.shape[0],
                                2,
                                el_dof_map.shape[1] ** 2,
                                ), dtype='int_')
            for el, dof_map in enumerate(el_dof_map):
                row_dof_map, col_dof_map = meshgrid(dof_map, dof_map)
                ij_dof_map[el, ...] = vstack([row_dof_map.flatten(),
                                              col_dof_map.flatten()])
            ij_dof_map_list.append(ij_dof_map)

        return ij_dof_map_list

    x_l = Property(depends_on='el_dof_map')

    @cached_property
    def _get_x_l(self):
        '''Helper property to get an array of all row indices'''
        return hstack([ij_map[:, 0, :].flatten()
                       for ij_map in self.ij_map])

    y_l = Property(depends_on='el_dof_map')

    @cached_property
    def _get_y_l(self):
        '''Helper property to get an array of all column indices'''
        return hstack([ij_map[:, 1, :].flatten()
                       for ij_map in self.ij_map])

    data_l = Property

    def _get_data_l(self):

        return hstack([sm_arr.mtx_arr.ravel()
                       for sm_arr in self.assemb.get_sys_mtx_arrays()])

    def solve(self, rhs, check_pos_dev=False):
        '''Construct the matrix and use the solver to get 
        the solution for the supplied rhs
        pos_dev - test the positive definiteness of the matrix. 
        '''
        ij = vstack((self.x_l, self.y_l))

        # Assemble the system matrix from the flattened data and
        # sparsity map containing two rows - first one are the row
        # indices and second one are the column indices.
        mtx = sparse.coo_matrix((self.data_l, ij))
        mtx_csr = mtx.tocsr()

        pos_def = True
        if check_pos_dev:
            evals_small, evecs_small = eigsh(mtx_csr, 3, sigma=0, which='LM')
            min_eval = np.min(evals_small)
            pos_def = min_eval > 1e-10

        u_vct = linsolve.spsolve(mtx_csr, rhs)
        return u_vct, pos_def

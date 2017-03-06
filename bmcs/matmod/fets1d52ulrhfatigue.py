'''
Created on 12.01.2016

@author: Yingxiong
'''

import numpy as np
from ibvpy.fets.fets_eval import FETSEval , IFETSEval
from traits.api import implements, Int, Array, \
    Property, cached_property, Float


class FETS1D52ULRHFatigue(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

#     A_m = Float(100 * 8 - 9 * 1.85, desc='matrix area [mm2]')
#     A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
#     P_b = Float(9 * np.sqrt(np.pi * 4 * 1.85),
#                 desc='perimeter of the bond interface [mm]')

    A_m = Float(1)
    A_f = Float(1)
    P_b = Float(1)

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    n_dof_r = Property
    '''Number of node positions associated with degrees of freedom. 
    '''
    @cached_property
    def _get_n_dof_r(self):
        return len(self.dof_r)

    n_e_dofs = Property
    '''Number of element degrees
    '''
    @cached_property
    def _get_n_e_dofs(self):
        return self.n_nodal_dofs * self.n_dof_r

    def _get_ip_coords(self):
        offset = 1e-6
        return np.array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return np.array([1., 1.], dtype=float)

    # Integration parameters
    #
    ngp_r = 2

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = np.array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return np.array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)

'''
Created on 15.02.2018

@author: abaktheer
'''

from ibvpy.fets.fets_eval import FETSEval
import numpy as np
import sympy as sp
import traits.api as tr


#=================================================
# 8 nodes isoparametric volume element (3D)
#=================================================
# generate shape functions with sympy
xi_1 = sp.symbols('xi_1')
xi_2 = sp.symbols('xi_2')
xi_3 = sp.symbols('xi_3')


#=========================================================================
# Finite element specification
#=========================================================================

N_xi_i = sp.Matrix([(1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                    (1.0 + xi_1) * (-1.0 + xi_2) * (-1.0 + xi_3) / 8.0,  # 2
                    (-1.0 + xi_1) * (1.0 + xi_2) * (-1.0 + xi_3) / 8.0,
                    (-1.0 - xi_1) * (-1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                    (-1.0 + xi_1) * (-1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                    (-1.0 - xi_1) * (1.0 - xi_2) * (-1.0 - xi_3) / 8.0,  # 6
                    (1.0 - xi_1) * (-1.0 - xi_2) * (-1.0 - xi_3) / 8.0,
                    (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0, ], dtype=np.float_)

dN_xi_ir = sp.Matrix((((-1.0 / 8.0) * (-1.0 + xi_2) * (-1.0 + xi_3),
                       (-1.0 / 8.0) * (-1.0 + xi_1) * (-1.0 + xi_3),
                       (-1.0 / 8.0) * (-1.0 + xi_1) * (-1.0 + xi_2)),

                      ((1.0 / 8.0) * (-1.0 + xi_2) * (-1.0 + xi_3),  # 2
                       (1.0 / 8.0) * (1.0 + xi_1) * (-1.0 + xi_3),
                       (1.0 / 8.0) * (1.0 + xi_1) * (-1.0 + xi_2)),

                      ((1.0 / 8.0) * (1.0 + xi_2) * (-1.0 + xi_3),
                       (1.0 / 8.0) * (-1.0 + xi_1) * (-1.0 + xi_3),
                       (1.0 / 8.0) * (-1.0 + xi_1) * (1.0 + xi_2)),

                      ((-1.0 / 8.0) * (1.0 + xi_2) * (-1.0 + xi_3),
                       (-1.0 / 8.0) * (1.0 + xi_1) * (-1.0 + xi_3),
                       (-1.0 / 8.0) * (1.0 + xi_1) * (1.0 + xi_2)),

                      ((1.0 / 8.0) * (-1.0 + xi_2) * (1.0 + xi_3),
                       (1.0 / 8.0) * (-1.0 + xi_1) * (1.0 + xi_3),
                       (1.0 / 8.0) * (-1.0 + xi_1) * (-1.0 + xi_2)),

                      ((-1.0 / 8.0) * (-1.0 + xi_2) * (1.0 + xi_3),  # 6
                       (-1.0 / 8.0) * (1.0 + xi_1) * (1.0 + xi_3),
                       (-1.0 / 8.0) * (1.0 + xi_1) * (-1.0 + xi_2)),

                      ((-1.0 / 8.0) * (1.0 + xi_2) * (1.0 + xi_3),
                       (-1.0 / 8.0) * (-1.0 + xi_1) * (1.0 + xi_3),
                       (-1.0 / 8.0) * (-1.0 + xi_1) * (1.0 + xi_2)),

                      ((1.0 / 8.0) * (1.0 + xi_2) * (1.0 + xi_3),
                       (1.0 / 8.0) * (1.0 + xi_1) * (1.0 + xi_3),
                       (1.0 / 8.0) * (1.0 + xi_1) * (1.0 + xi_2))), dtype=np.float_)

# symetrization operator
delta = np.identity(3)


class FETS3D8H(FETSEval):
    dof_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])

    geo_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])
    vtk_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])
    n_nodal_dofs = 3

    vtk_cells = [[0, 1, 3, 2, 4, 5, 7, 6]]
    vtk_cell_types = 'Hexahedron'
    vtk_cell = [0, 1, 3, 2, 4, 5, 7, 6]
    vtk_cell_type = 'Hexahedron'

    vtk_expand_operator = tr.Array(np.float_, value=np.identity(3))

    # numerical integration points (IP) and weights
    xi_m = tr.Array(np.float_,
                    value=[[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0),
                            1.0 / np.sqrt(3.0)],
                           ])

    w_m = tr.Array(value=[1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float_)

    n_m = tr.Property

    def _get_n_m(self):
        return len(self.w_m)

    shape_function_values = tr.Property(tr.Tuple)
    '''The values of the shape functions and their derivatives at the IPs
    '''
    @tr.cached_property
    def _get_shape_function_values(self):
        N_mi = np.array([N_xi_i.subs(list(zip([xi_1, xi_2, xi_3], xi)))
                         for xi in self.xi_m], dtype=np.float_)
        N_im = np.einsum('mi->im', N_mi)
        dN_mir = np.array(
            [dN_xi_ir.subs(list(zip([xi_1, xi_2, xi_3], xi)))
             for xi in self.xi_m], dtype=np.float_
        ).reshape(8, 8, 3)
        dN_nir = np.array(
            [dN_xi_ir.subs(list(zip([xi_1, xi_2, xi_3], xi)))
             for xi in self.vtk_r], dtype=np.float_
        ).reshape(8, 8, 3)
        dN_imr = np.einsum('mir->imr', dN_mir)
        dN_inr = np.einsum('nir->inr', dN_nir)
        return (N_im, dN_imr, dN_inr)

    N_im = tr.Property()
    '''Shape function values in integration poindots.
    '''

    def _get_N_im(self):
        return self.shape_function_values[0]

    dN_imr = tr.Property()
    '''Shape function derivatives in integration poindots.
    '''

    def _get_dN_imr(self):
        return self.shape_function_values[1]

    dN_inr = tr.Property()
    '''Shape function derivatives in visualization poindots.
    '''

    def _get_dN_inr(self):
        return self.shape_function_values[2]

    I_sym_abcd = tr.Array(np.float)

    def _I_sym_abcd_default(self):
        return 0.5 * \
            (np.einsum('ac,bd->abcd', delta, delta) +
             np.einsum('ad,bc->abcd', delta, delta))

'''
Created on 12.01.2016

@author: Yingxiong
'''

from traits.api import Int, Array, \
    Property, cached_property, Float, List, provides, Type

from ibvpy.fets.fets_eval import FETSEval, IFETSEval
import numpy as np
import sympy as sp
import traits.api as tr


r_ = sp.symbols('r')

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

N_xi_i = sp.Matrix([0.5 - xi_1 / 2., 0.5 + xi_1 / 2.], dtype=np.float_)

dN_xi_ir = sp.Matrix([[-1. / 2], [1. / 2]], dtype=np.float_)


@provides(IFETSEval)
class FETS1D52ULRHFatigue(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

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

    dots_class = Type

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

    xi_m = Array(value=[[-1], [1]], dtype=np.float)
    r_m = Array(value=[[-1], [1]], dtype=np.float_)
    w_m = Array(value=[1, 1], dtype=np.float_)

    Nr_i_geo = List([(1 - r_) / 2.0,
                     (1 + r_) / 2.0, ])

    dNr_i_geo = List([- 1.0 / 2.0,
                      1.0 / 2.0, ])

    Nr_i = Nr_i_geo
    dNr_i = dNr_i_geo

    N_mi_geo = Property()

    @cached_property
    def _get_N_mi_geo(self):
        return self.get_N_mi(sp.Matrix(self.Nr_i_geo, dtype=np.float_))

    dN_mid_geo = Property()

    @cached_property
    def _get_dN_mid_geo(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i_geo, dtype=np.float_))

    N_mi = Property()

    @cached_property
    def _get_N_mi(self):
        return self.get_N_mi(sp.Matrix(self.Nr_i, dtype=np.float_))

    dN_mid = Property()

    @cached_property
    def _get_dN_mid(self):
        return self.get_dN_mid(sp.Matrix(self.dNr_i, dtype=np.float_))

    def get_N_mi(self, Nr_i):
        return np.array([Nr_i.subs(r_, r)
                         for r in self.r_m], dtype=np.float_)

    def get_dN_mid(self, dNr_i):
        dN_mdi = np.array([[dNr_i.subs(r_, r)]
                           for r in self.r_m], dtype=np.float_)
        return np.einsum('mdi->mid', dN_mdi)

    n_m = Property(depends_on='w_m')

    @cached_property
    def _get_n_m(self):
        return len(self.w_m)

    A_C = Property(depends_on='A_m,A_f')

    @cached_property
    def _get_A_C(self):
        return np.array((self.A_f, self.P_b, self.A_m), dtype=np.float_)

    def get_B_EmisC(self, J_inv_Emdd):
        fets_eval = self

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_m = fets_eval.n_gp
        n_E = J_inv_Emdd.shape[0]
        n_s = 3
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None, :])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv_Emdd, dNr)

        B = np.zeros((n_E, n_m, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :, :, B_N_n_rows, B_N_n_cols] = (B_factors[None, None, :] *
                                              Nx[:, :, N_idx])
        B[:, :, :, B_dN_n_rows, B_dN_n_cols] = dNx[:, :, :, dN_idx]

        return B

    def _get_B_EimsC(self, dN_Eimd, sN_Cim):

        n_E, _, _, _ = dN_Eimd.shape
        n_C, n_i, n_m = sN_Cim.shape
        n_s = 3
        B_EimsC = np.zeros(
            (n_E, n_i, n_m, n_s, n_C), dtype='f')
        B_EimsC[..., [1, 1], [0, 1]] = sN_Cim[[0, 1], :, :]
        B_EimsC[..., [0, 2], [0, 1]] = dN_Eimd[:, [0, 1], :, :]

        return B_EimsC

    shape_function_values = tr.Property(tr.Tuple)
    '''The values of the shape functions and their derivatives at the IPs
    '''
    @tr.cached_property
    def _get_shape_function_values(self):
        N_mi = np.array([N_xi_i.subs(list(zip([xi_1, xi_2, xi_3], xi)))
                         for xi in self.xi_m], dtype=np.float_)
        N_im = np.einsum('mi->im', N_mi)
        dN_mir_arr = [np.array(dN_xi_ir.subs(list(zip([xi_1], xi)))).astype(np.float_)
                      for xi in self.xi_m]
        dN_mir = np.array(dN_mir_arr, dtype=np.float)
        dN_nir_arr = [np.array(dN_xi_ir.subs(list(zip([xi_1], xi)))).astype(np.float_)
                      for xi in self.vtk_r]
        dN_nir = np.array(dN_nir_arr, dtype=np.float)
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
        delta = np.identity(3)
        return 0.5 * \
            (np.einsum('ac,bd->abcd', delta, delta) +
             np.einsum('ad,bc->abcd', delta, delta))


if __name__ == '__main__':
    fe = FETS1D52ULRHFatigue()
    print('dN_imr', fe.dN_imr)

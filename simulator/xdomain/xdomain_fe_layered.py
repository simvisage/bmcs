'''
'''
from traits.api import HasStrictTraits, Instance, \
    Property, cached_property, Float, provides, Int, Type, Tuple

from ibvpy.fets.fets1D5.fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from ibvpy.fets.i_fets_eval import IFETSEval
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_array import SysMtxArray
import numpy as np
from simulator.i_xdomain import IXDomain


n_C = 2

ONE = np.ones((1,), dtype=np.float_)
DELTA_cd = np.identity(n_C)
c1 = np.arange(n_C) + 1
SWITCH_C = np.power(-1.0, c1)
SWITCH_CD = np.power(-1.0, c1[np.newaxis, :] + c1[:, np.newaxis])


@provides(IXDomain)
class XDomainFEGridLayered(HasStrictTraits):
    '''Discretization grid of a layered medium.
    '''
    # Number of elements
    n_e_x = Int(20)
    # length
    L_x = Float(200)

    #=========================================================================
    # Type and shape specification of state variables representing the domain
    #=========================================================================
    U_var_shape = Property(Int)

    vtk_expand_operator = Property

    def _get_vtk_expand_operator(self):
        return self.fets.vtk_expand_operator

    K_type = Type(SysMtxArray)

    def _get_U_var_shape(self):
        return self.mesh.n_dofs

    state_var_shape = Property(Tuple)

    def _get_state_var_shape(self):
        return (self.mesh.n_active_elems, self.fets.n_m,)

    #=========================================================================
    # Conversion between linear algebra objects and field variables
    #=========================================================================
    def map_U_to_field(self, U):
        n_c = self.fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[self.I_Ei]
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            self.B_Eimabc, U_Eia
        )
        return eps_Emab

    def map_field_to_F(self, sig_Emab):
        _, n_i, _, _, _, n_c = self.B_Eimabc.shape
        f_Eic = self.integ_factor * np.einsum(
            'm,Eimabc,Emab,Em->Eic',
            self.fets.w_m, self.B_Eimabc, sig_Emab, self.det_J_Em
        )
        f_Ei = f_Eic.reshape(-1, n_i * n_c)
        dof_E = self.dof_Eia.reshape(-1, n_i * n_c)
        F_int = np.bincount(dof_E.flatten(), weights=f_Ei.flatten())
        return F_int

    def map_field_to_K(self, D_Emabef):
        K_Eicjd = self.integ_factor * np.einsum(
            'Emicjdabef,Emabef->Eicjd', self.BB_Emicjdabef, D_Emabef
        )
        _, n_i, n_c, n_j, n_d = K_Eicjd.shape
        K_Eij = K_Eicjd.reshape(-1, n_i * n_c, n_j * n_d)
        dof_Ei = self.dof_Eia.reshape(-1, n_i * n_c)
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=dof_Ei)

    fets = Instance(IFETSEval)
    '''Finite element formulation object.
    '''

    def _fets_default(self):
        return FETS1D52ULRHFatigue()

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''

    def _get_A(self):
        return np.array([self.fets.A_m, self.fets.P_b,
                         self.fets.A_f])

    sdomain = Property(Instance(FEGrid), depends_on='L_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_sdomain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets)
        return domain

    #=========================================================================
    # index maps
    #=========================================================================

    dof_ECid = Property(depends_on='+input')
    '''For a given element, layer, node number and dimension
    return the dof number
    '''
    @cached_property
    def _get_dof_ECid(self):
        dof_EiCd = self.sdomain.dof_grid.cell_dof_map[..., np.newaxis]
        return np.einsum('EiCd->ECid', dof_EiCd)

    I_Ei = Property(depends_on='+input')
    '''For a given element and its node number return the global index
    of the node'''
    @cached_property
    def _get_I_Ei(self):
        return self.sdomain.geo_grid.cell_grid.cell_node_map

    dof_E = Property(depends_on='+input')
    '''Get ordered array of degrees of freedom corresponding to each element.
    '''
    @cached_property
    def _get_dof_E(self):
        return self.dof_ECid.reshape(-1, self.fets.n_e_dofs)

    dof_ICd = Property(depends_on='+input')
    '''Get degrees of freedom
    '''
    @cached_property
    def _get_dof_ICd(self):
        return self.sdomain.dof_grid.dofs

    dofs = Property(depends_on='_input')
    '''Get degrees of freedom flat'''
    @cached_property
    def _get_dofs(self):
        return self.dof_ICd.flatten()
    #=========================================================================
    # Coordinate arrays
    #=========================================================================

    X_Id = Property(depends_on='+input')
    'Coordinate of the node `I` in dimension `d`'
    @cached_property
    def _get_X_Id(self):
        return self.sdomain.geo_grid.cell_grid.point_x_arr

    X_Eid = Property(depends_on='+input')
    'Coordinate of the node `i` in  element `E` in dimension `d`'
    @cached_property
    def _get_X_Eid(self):
        return self.X_Id[self.I_Ei, :]

    X_Emd = Property(depends_on='+input')
    'Coordinate of the integration point `m` of an element `E` in dimension `d`'
    @cached_property
    def _get_X_Emd(self):
        N_mi_geo = self.fets.N_mi_geo
        return np.einsum('mi,Eid->Emd', N_mi_geo, self.X_Eid)

    X_J = Property(depends_on='+input')
    '''Return ordered vector of nodal coordinates respecting the the order
    of the flattened array of elements, nodes and spatial dimensions.'''
    @cached_property
    def _get_X_J(self):
        return self.X_Eid.flatten()

    X_M = Property(depends_on='+input')
    '''Return ordered vector of global coordinates of integration points
    respecting the the order of the flattened array of elements, 
    nodes and spatial dimensions. Can be used for point-value visualization
    of response variables.'''
    @cached_property
    def _get_X_M(self):
        return self.X_Emd.flatten()

    #=========================================================================
    # cached time-independent terms
    #=========================================================================
    dN_Eimd = Property
    '''Shape function derivatives in every integration point
    '''

    def _get_dN_Eimd(self):
        return self.constant_terms[0]

    sN_Cim = Property
    '''Slip operator between the layers C = 0,1
    '''

    def _get_sN_Cim(self):
        return self.constant_terms[1]

    constant_terms = Property(depends_on='+input')
    '''Procedure calculating all constant terms of the finite element
    algorithm including the geometry mapping (Jacobi), shape 
    functions and the kinematics needed
    for the integration of stresses and stifnesses in every material point.
    '''
    @cached_property
    def _get_constant_terms(self):
        fets = self.fets
        dN_mid_geo = fets.dN_mid_geo
        N_mi = fets.N_mi
        dN_mid = fets.dN_mid
        # Geometry approximation / Jacobi transformation
        J_Emde = np.einsum('mid,Eie->Emde', dN_mid_geo, self.X_Eid)
        J_inv_Emed = np.linalg.inv(J_Emde)

        # Quadratic forms
        dN_Eimd = np.einsum('mid,Eied->Eime', dN_mid, J_inv_Emed)
        sN_Cim = np.einsum('C,mi->Cim', SWITCH_C, N_mi)
        return dN_Eimd, sN_Cim

    J_mtx = Property(depends_on='L_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets = self.fets
        domain = self.sdomain
        # [ d, n ]
        geo_r = fets.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property(depends_on='L_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='L_x')
    '''The B matrix
    '''
    @cached_property
    def _get_B(self):
        '''Calculate and assemble the system stiffness matrix.
        '''
        fets = self.fets
        domain = self.sdomain

        n_s = 3

        n_dof_r = fets.n_dof_r
        n_nodal_dofs = fets.n_nodal_dofs

        n_ip = fets.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None, :])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :, :, B_N_n_rows, B_N_n_cols] = (B_factors[None, None, :] *
                                              Nx[:, :, N_idx])
        B[:, :, :, B_dN_n_rows, B_dN_n_cols] = dNx[:, :, :, dN_idx]

        return B


if __name__ == '__main__':
    xd = XDomainFEGridLayered(n_e_x=2)
    print(xd.B.shape)

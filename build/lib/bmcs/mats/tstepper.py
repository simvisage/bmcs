'''
Created on 12.01.2016
@author: Yingxiong
'''
from ibvpy.api import \
    BCDof, IMATSEval, IFETSEval, FEGrid
from ibvpy.core.bcond_mngr import BCondMngr
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from traits.api import HasTraits, Instance, \
    Property, cached_property, List

from bmcs.mats.fets1d52ulrhfatigue_delete_moved_to_fets import FETS1D52ULRHFatigue
from .mats_bondslip import MATSBondSlipFatigue
import numpy as np

n_C = 2

ONE = np.ones((1,), dtype=np.float_)
DELTA_cd = np.identity(n_C)
c1 = np.arange(n_C) + 1
SWITCH_C = np.power(-1.0, c1)
SWITCH_CD = np.power(-1.0, c1[np.newaxis, :] + c1[:, np.newaxis])


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Instance(IMATSEval)
    '''Finite element formulation object.
    '''

    def _mats_eval_default(self):
        return MATSBondSlipFatigue()

    fets_eval = Instance(IFETSEval)
    '''Finite element formulation object.
    '''

    def _fets_eval_default(self):
        return FETS1D52ULRHFatigue()

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''

    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.P_b, self.fets_eval.A_f])

    sdomain = Instance(FEGrid)
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
        return self.dof_ECid.reshape(-1, self.fets_eval.n_e_dofs)

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
        N_mi_geo = self.fets_eval.N_mi_geo
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
        fet = self.fets_eval
        dN_mid_geo = fet.dN_mid_geo
        N_mi = fet.N_mi
        dN_mid = fet.dN_mid
        # Geometry approximation / Jacobi transformation
        J_Emde = np.einsum('mid,Eie->Emde', dN_mid_geo, self.X_Eid)
        J_inv_Emed = np.linalg.inv(J_Emde)
        # Quadratic forms
        dN_Eimd = np.einsum('mid,Eied->Eime', dN_mid, J_inv_Emed)
        sN_Cim = np.einsum('C,mi->Cim', SWITCH_C, N_mi)
        return dN_Eimd, sN_Cim

    # Boundary condition manager
    #
    bcond_mngr = Instance(BCondMngr)

    def _bcond_mngr_default(self):
        return BCondMngr()

    bcond_list = Property(List(BCDof))
    '''Convenience constructor
    This property provides the possibility to write
    tstepper.bcond_list = [BCDof(var='u',dof=5,value=0, ... ]
    The result gets propagated to the BCondMngr
    '''

    def _get_bcond_list(self):
        return self.bcond_mngr.bcond_list

    def _set_bcond_list(self, bcond_list):
        self.bcond_mngr.bcond_list = bcond_list

    J_mtx = Property(depends_on='L_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.sdomain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
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
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.sdomain

        n_s = mats_eval.n_s - 1

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
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

    def apply_essential_bc(self):
        '''Insert initial boundary conditions at the start up of the calculation.. 
        '''
        self.K = SysMtxAssembly()
        for bc in self.bcond_list:
            bc.apply_essential(self.K)

    def apply_bc(self, step_flag, K_mtx, F_ext, t_n, t_n1):
        '''Apply boundary conditions for the current load increement
        '''
        for bc in self.bcond_list:
            bc.apply(step_flag, None, K_mtx, F_ext, t_n, t_n1)

    def get_corr_pred(self, step_flag, U, d_U, eps, sig,
                      t_n, t_n1, xs_pi, alpha, z, kappa, w):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.sdomain
        elem_dof_map = domain.elem_dof_map

        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_nodal_dofs = self.fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_nodal_dofs)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('Einsd,End->Eis', self.B, d_u_n)
        # print'B =', self.B
        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D, xs_pi, alpha, z, w = mats_eval.get_corr_pred(
            eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, kappa, w)
        # print'D=',D
        # system matrix
        self.K.reset_mtx()
        Ke = np.einsum('i,s,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.A, self.B, D, self.B, self.J_det)

        self.K.add_mtx_array(
            Ke.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,s,eis,einsd,ei->end',
                           w_ip, self.A, sig, self.B, self.J_det)
        F_int = np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())
        F_ext = np.zeros_like(F_int, dtype=np.float_)
        self.apply_bc(step_flag, self.K, F_ext, t_n, t_n1)

        return F_ext - F_int, F_int, self.K, eps, sig, xs_pi, alpha, z, w, D

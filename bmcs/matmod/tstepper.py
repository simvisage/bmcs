'''
Created on 12.01.2016
@author: Yingxiong
'''
from ibvpy.api import BCDof
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from traits.api import HasTraits, Instance, \
    Property, cached_property, Float, List

from fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from mats_bondslip import MATSEvalFatigue
import numpy as np


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Property(Instance(MATSEvalFatigue))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_mats_eval(self):
        return MATSEvalFatigue()

    fets_eval = Property(Instance(FETS1D52ULRHFatigue))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRHFatigue()

    A = Property()
    '''array containing the A_m, L_b, A_f
    '''

    def _get_A(self):
        return np.array([self.fets_eval.A_m, self.fets_eval.P_b, self.fets_eval.A_f])

    # Number of elements
    n_e_x = 20
    # length
    L_x = Float(200)

    domain = Property(Instance(FEGrid), depends_on='L_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

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
        domain = self.domain
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
        domain = self.domain

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
                      t_n, t_n1, xs_pi, alpha, z, w):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
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
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)
        # print'B =', self.B
        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D, xs_pi, alpha, z, w = mats_eval.get_corr_pred(
            eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, w)
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

'''
Created on Mar 4, 2017

@author: rch

'''
from ibvpy.api import \
    TStepperEval, IFETSEval
from ibvpy.core.i_sdomain import ISDomain
from mathkit.matrix_la import \
    SysMtxArray
from traits.api import \
    Array, \
    Property, cached_property, \
    Instance, Float

import numpy as np


# todo -- move this into the elementdefinition
n_C = 2

ONE = np.ones((1,), dtype=np.float_)
DELTA_cd = np.identity(n_C)
c1 = np.arange(n_C) + 1
SWITCH_C = np.power(-1.0, c1)
SWITCH_CD = np.power(-1.0, c1[np.newaxis, :] + c1[:, np.newaxis])


class DOTSGridEval(TStepperEval):

    #=========================================================================
    # Original code, might be obsolete
    #=========================================================================
    sdomain = Instance(ISDomain)

    def new_cntl_var(self):
        return np.zeros(self.sdomain.n_dofs, np.float_)

    def new_resp_var(self):
        return np.zeros(self.sdomain.n_dofs, np.float_)

    def new_tangent_operator(self):
        '''
        Return the tangent operator used for the time stepping
        '''
        return SysMtxArray()

    state_array = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_array(self):

        # The overall size is just a n_elem times the size of a single element
        #
        n_elems = self.sdomain.n_elems
        n_ip = self.fets_eval.n_ip
        state_arr_size = self.fets_eval.state_arr_size
        return np.zeros((n_elems, n_ip, state_arr_size), dtype=np.float_)

    fets_eval = Instance(IFETSEval)

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
    BB_ECidDjf = Property(Array)
    '''Product of shape function derivatives  mappings 
    in every integration point
    '''

    def _get_BB_ECidDjf(self):
        return self.constant_terms[0]

    NN_ECidDjf = Property(Array)
    '''Product of shape functions in every integration point
    '''

    def _get_NN_ECidDjf(self):
        return self.constant_terms[1]

    dN_Eimd = Property
    '''Shape function derivatives in every integration point
    '''

    def _get_dN_Eimd(self):
        return self.constant_terms[2]

    sN_Cim = Property
    '''Slip operator between the layers C = 0,1
    '''

    def _get_sN_Cim(self):
        return self.constant_terms[3]

    eta = Float(1.0, label='ratio of stiffnesses E_0/E_1',
                enter_set=True, auto_set=False)

    G = Float(1.0, label='Shear stiffness',
              enter_set=True, auto_set=False)

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
        w_m = fet.w_m

        A_C = np.ones((n_C,), dtype=np.float_)
        A_C[0] *= self.eta
        # Geometry approximation / Jacobi transformation
        J_Emde = np.einsum('mid,Eie->Emde', dN_mid_geo, self.X_Eid)
        J_det_Em = np.linalg.det(J_Emde)
        J_inv_Emed = np.linalg.inv(J_Emde)
        # Quadratic forms
        dN_Eimd = np.einsum('mid,Eied->Eime', dN_mid, J_inv_Emed)
        sN_Cim = np.einsum('C,mi->Cim', SWITCH_C, N_mi)
        BB_ECidDjf = np.einsum('m, CD, C, Eimd,Ejmf,Em->ECidDjf',
                               w_m, DELTA_cd, A_C, dN_Eimd, dN_Eimd, J_det_Em)
        NN_ECidDjf = np.einsum('m, d,f,CD,mi,mj,Em->ECidDjf',
                               w_m, ONE, ONE, SWITCH_CD, N_mi, N_mi, J_det_Em)
        return BB_ECidDjf, NN_ECidDjf, dN_Eimd, sN_Cim

    def get_corr_pred(self, sctx, U, dU, tn, tn1, F_int, *args, **kw):

        n_e_dofs = self.fets_eval.n_e_dofs
        #
        K_ECidDjf = self.BB_ECidDjf + self.NN_ECidDjf * self.G
        K_Eij = K_ECidDjf.reshape(-1, n_e_dofs, n_e_dofs)
        #
        dU_ECid = dU[self.dof_ECid]
        f_ECid = np.einsum('ECidDjf,EDjf->ECid', K_ECidDjf, dU_ECid)
        f_Ei = f_ECid.reshape(-1, self.fets_eval.n_e_dofs)
        F_I = np.bincount(self.dof_E.flatten(), weights=f_Ei.flatten())
        F_int[self.dofs] += F_I
        #
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=self.dof_E)

    def get_corr_pred2(self, sctx, U, dU, tn, tn1, F_int, *args, **kw):

        fets = self.fets_eval
        sa = self.state_array
        U_ECid = U[self.dof_ECid]
        dU_ECid = dU[self.dof_ECid]
        eps_ECmdf, deps_ECmdf = fets.get_eps(self.dN_Eimd, U_ECid, dU_ECid)
        sig_ECmdf, D_ECmdfgh = fets.get_sig_D(eps_ECmdf, deps_ECmdf, sa)

        n_e_dofs = self.fets_eval.n_e_dofs
        #
        K_ECidDjf = self.BB_ECidDjf + self.NN_ECidDjf * self.G
        K_Eij = K_ECidDjf.reshape(-1, n_e_dofs, n_e_dofs)
        #
        f_ECid = np.einsum('ECidDjf,EDjf->ECid', K_ECidDjf, dU_ECid)
        f_Ei = f_ECid.reshape(-1, n_e_dofs)
        F_I = np.bincount(self.dof_E.flatten(), weights=f_Ei.flatten())
        F_int[self.dofs] += F_I
        #
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=self.dof_E)

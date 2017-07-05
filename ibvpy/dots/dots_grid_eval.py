'''
Created on Mar 4, 2017

@author: rch

'''
from ibvpy.api import \
    TStepperEval, IFETSEval, IMATSEval
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

    state_array_n = Property(depends_on='sdomain.changed_structure')

    @cached_property
    def _get_state_array_n(self):

        # The overall size is just a n_elem times the size of a single element
        #
        n_E = self.sdomain.n_elems
        n_m = self.fets_eval.n_m
        state_arr_shape = self.mats_eval.state_arr_shape
        return np.zeros((n_E, n_m) + state_arr_shape, dtype=np.float_)

    state_array_n1 = Array
    '''State variable with trial values
    '''

    fets_eval = Property(Instance(IFETSEval))

    def _get_fets_eval(self):
        return self.sdomain.fets_eval

    mats_eval = Property(Instance(IMATSEval))

    def _get_mats_eval(self):
        return self.fets_eval.mats_eval

    #=========================================================================
    # index maps
    #=========================================================================

    dof_EiCd = Property(depends_on='+input')
    '''For a given element, layer, node number and dimension
    return the dof number
    '''
    @cached_property
    def _get_dof_EiCd(self):
        dof_EiCd = self.sdomain.dof_Eid[..., np.newaxis]
        return dof_EiCd

    I_Ei = Property(depends_on='+input')
    '''For a given element and its node number return the global index
    of the geometric node'''
    @cached_property
    def _get_I_Ei(self):
        return self.sdomain.I_Ei

    dof_E = Property(depends_on='+input')
    '''Get ordered array of degrees of freedom corresponding to each element.
    '''
    @cached_property
    def _get_dof_E(self):
        return self.dof_EiCd.reshape(-1, self.fets_eval.n_e_dofs)

    dof_ICd = Property(depends_on='+input')
    '''Get degrees of freedom
    '''
    @cached_property
    def _get_dof_ICd(self):
        return self.sdomain.dofs

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
        return self.sdomain.X_Id

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

    B_EmisC = Property(depends_on='+input')
    '''Array of cached B-matices.'''
    @cached_property
    def _get_B_EmisC(self):
        return self.constant_terms[2]

    J_det_Em = Property(depends_on='+input')
    '''Array of Jacobi determinants.'''
    @cached_property
    def _get_J_det_Em(self):
        return self.constant_terms[3]

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

    def get_K(self):
        w_m = self.fets_eval.w_m
        A_C = self.fets_eval.A_C
        B_EmisCd = self.B_EmisC[..., np.newaxis]
        K_ECidDjf = np.einsum('m,s,EmisCd,EmjrDf,Em->ECidDjf',
                              w_m, A_C, B_EmisCd, B_EmisCd, self.J_det_Em)
        return K_ECidDjf

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
        J_det_Em = np.linalg.det(J_Emde)
        J_inv_Emed = np.linalg.inv(J_Emde)
        # Quadratic forms
        dN_Eimd = np.einsum('mid,Eied->Eime', dN_mid, J_inv_Emed)
        sN_Cim = np.einsum('C,mi->Cim', SWITCH_C, N_mi)
        B_EmisC = self.fets_eval.get_B_EmisC(J_inv_Emed)
        return dN_Eimd, sN_Cim, B_EmisC, J_det_Em

    def get_corr_pred(self, U, dU, tn, tn1, F_int,
                      step_flag='predictor', update_state=False,
                      *args, **kw):

        if update_state:
            self.state_array_n[...] = self.state_array_n1[...]

        mats = self.mats_eval
        w_m = self.fets_eval.w_m
        A_C = self.fets_eval.A_C

        sa_n = self.state_array_n
        U_EiCd = U[self.dof_EiCd]
        dU_EiCd = dU[self.dof_EiCd]

        n_e_dofs = self.fets_eval.n_e_dofs
        B_EmisCd = self.B_EmisC[..., np.newaxis]
        eps_Ems = np.einsum('EmisCd,EiCd->Ems', B_EmisCd, U_EiCd)
        deps_Ems = np.einsum('EmisCd,EiCd->Ems', B_EmisCd, dU_EiCd)
        sig_Ems, D_Emsr, sa_n1 = mats.get_corr_pred2(eps_Ems, deps_Ems,
                                                     tn, tn1, sa_n)
        self.state_array_n1 = sa_n1
        K_EiCdjDf = np.einsum('m,s,EmisCd,Emsr,EmjrDf,Em->EiCdjDf',
                              w_m, A_C, B_EmisCd, D_Emsr, B_EmisCd,
                              self.J_det_Em)
        K_Eij = K_EiCdjDf.reshape(-1, n_e_dofs, n_e_dofs)
        f_EiCd = np.einsum('m,s,EmisCd,Ems,Em->EiCd',
                           w_m, A_C, B_EmisCd, sig_Ems, self.J_det_Em)
        f_Ei = f_EiCd.reshape(-1, n_e_dofs)
        F_dof = np.bincount(self.dof_E.flatten(), weights=f_Ei.flatten())
        F_int[self.dofs] += F_dof
        return SysMtxArray(mtx_arr=K_Eij, dof_map_arr=self.dof_E)

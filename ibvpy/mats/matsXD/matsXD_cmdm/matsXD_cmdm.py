#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Aug 19, 2009 by: rch, ascholzen

from math import sqrt as scalar_sqrt
from numpy import \
    array, zeros, outer, inner, transpose, dot, \
    fabs, identity, tensordot, \
    float_, \
    sqrt as arr_sqrt, copy
from scipy.linalg import \
    eigh, inv
from traits.api import \
    Bool, Callable, Enum, \
    Int, Trait, on_trait_change, \
    Dict, Property, cached_property
from traitsui.api import \
    Item, View, Group, Spring, Include

from ibvpy.core.rtrace_eval import \
    RTraceEval
from .matsXD_cmdm_polar_discr import \
    PolarDiscr
import numpy as np


# @todo parameterize for 2D and 2D5 - does not make sense for 3D
#---------------------------------------------------------------------------
# Material time-step-evaluator for Microplane-Damage-Model
#---------------------------------------------------------------------------
class MATSXDMicroplaneDamage(PolarDiscr):

    '''
    Microplane Damage Model.
    '''

    #-------------------------------------------------------------------------
    # Classification traits (supplied by the dimensional subclasses)
    #-------------------------------------------------------------------------

    # specification of the model dimension (2D, 3D)
    n_dim = Int

    # specification of number of engineering strain and stress components
    n_eng = Int

    #-------------------------------------------------------------------------
    # Configuration parameters
    #-------------------------------------------------------------------------

    model_version = Enum("compliance", "stiffness")

    symmetrization = Enum("product-type", "sum-type")

    regularization = Bool(False,
                          desc='Flag to use the element length projection'
                          ' in the direction of principle strains',
                          enter_set=True,
                          auto_set=False)

    elastic_debug = Bool(False,
                         desc='Switch to elastic behavior - used for debugging',
                         auto_set=False)

    double_constraint = Bool(False,
                             desc='Use double constraint to evaluate microplane elastic and fracture energy (Option effects only the response tracers)',
                             auto_set=False)

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    config_param_vgroup = Group(Item('model_version', style='custom'),
                                #     Item('stress_state', style='custom'),
                                Item('symmetrization', style='custom'),
                                Item('elastic_debug@'),
                                Item('double_constraint@'),
                                Spring(resizable=True),
                                label='Configuration parameters',
                                show_border=True,
                                dock='tab',
                                id='ibvpy.mats.matsXD.MATSXD_cmdm.config',
                                )

    traits_view = View(Include('polar_fn_group'),
                       dock='tab',
                       id='ibvpy.mats.matsXD.MATSXD_cmdm',
                       kind='modal',
                       resizable=True,
                       scrollable=True,
                       width=0.6, height=0.8,
                       buttons=['OK', 'Cancel']
                       )

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def get_state_array_size(self):
        # In the state array the largest equivalent microplane
        # strains reached in the loading history are saved
        return self.n_mp

    D4_e = Property

    def _get_D4_e(self):
        '''
        Return the elasticity tensor
        '''
        return self.elasticity_tensors[0]

    C4_e = Property

    def _get_C4_e(self):
        '''
        Return the elastic compliance tensor
        '''
        return self.elasticity_tensors[1]

    # -----------------------------------------------------------------------------------------------------
    # Get the 3x3-elasticity and compliance matrix (used only for elastic debug)
    # -----------------------------------------------------------------------------------------------------
    D2_e = Property(depends_on='stress_state, E, nu')

    @cached_property
    def _get_D2_e(self):
        return self.elasticity_tensors[2]

    identity_tns = Property

    @cached_property
    def _get_identity_tns(self):
        '''
        Get the identity matrix (used only in formula for sum-type symmetrization)
        ''' 
        return identity(self.n_dim)

    #-------------------------------------------------------------------------
    # MICROPLANE-DISCRETIZATION RELATED METHOD
    #-------------------------------------------------------------------------
    # get the dyadic product of the microplane normals
    # the array of microplane normals is implemented
    # in the dimension-specific subclasses.
    #
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals
        return array([outer(mpn, mpn) for mpn in self._MPN])

    #-------------------------------------------------------------------------
    # Prepare evaluation - get the 4th order damage tensor beta4
    # (or damage effect tensor M4)
    #-------------------------------------------------------------------------
    def _get_e_vct_arr(self, eps_eng):
        '''
        Projects the strain tensor onto the microplanes and returns a list of
        microplane strain vectors. Method is used both by stiffness and
        compliance version to derive the list 'phi_arr' or 'psi_arr'!
        In case of the compliance version the kinematic constraint is not
        assumed in the derivation of the formula for the damage compliance
        tensor 'C_mdm', e.g. the construction of the damage effect tensor 'M4'.
        '''
        # Switch from engineering notation to tensor notation for the apparent
        # strains
        eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        # Projection of apparent strain onto the individual microplanes
        # slower: e_vct_arr = array( [ dot( eps_mtx, mpn ) for mpn in self._MPN ] )
        # slower: e_vct_arr = transpose( dot( eps_mtx, transpose(self._MPN) ))
        # due to the symmetry of the strain tensor eps_mtx = transpose(eps_mtx)
        # and so this is equal
        e_vct_arr = dot(self._MPN, eps_mtx)
        return e_vct_arr

    def _get_e_equiv_arr(self, e_vct_arr):
        '''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_arr = array([dot(e_vct, mpn)
                         for e_vct, mpn in zip(e_vct_arr, self._MPN)])
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_arr = (fabs(e_N_arr) + e_N_arr) / 2
        # normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_vct_arr = array([self._MPN[i, :] * e_N_arr[i]
                             for i in range(0, self.n_mp)])
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_vct_arr = e_vct_arr - e_N_vct_arr
        # squared tangential strain vector for each microplane
        e_TT_arr = array([inner(e_T_vct, e_T_vct) for e_T_vct in e_T_vct_arr])
        # equivalent strain for each microplane
        e_equiv_arr = arr_sqrt(e_N_pos_arr * e_N_pos_arr + c_T * e_TT_arr)
        return e_equiv_arr

    def _get_e_max(self, e_equiv_arr, e_max_arr):
        '''
        Compares the equivalent microplane strain of a single microplane with the
        maximum strain reached in the loading history for the entire array
        '''
        bool_e_max = e_equiv_arr >= e_max_arr

        # [rch] fixed a bug here - this call was modifying the state array
        # at any invocation.
        #
        # The new value must be created, otherwise side-effect could occur
        # by writing into a state array.
        #
        new_e_max_arr = copy(e_max_arr)
        new_e_max_arr[bool_e_max] = e_equiv_arr[bool_e_max]
        return new_e_max_arr

    def _get_state_variables(self, sctx, eps_app_eng):
        '''
        Compares the list of current equivalent microplane strains with
        the values in the state array and returns the higher values
        '''
        e_vct_arr = self._get_e_vct_arr(eps_app_eng)
        e_equiv_arr = self._get_e_equiv_arr(e_vct_arr)
        e_max_arr_old = sctx.mats_state_array
        e_max_arr_new = self._get_e_max(e_equiv_arr, e_max_arr_old)
        return e_max_arr_new

    def _get_phi_arr(self, sctx, eps_app_eng):
        '''
        Returns a list of the integrity factors for all microplanes.
        '''
        e_max_arr = self._get_state_variables(sctx, eps_app_eng)

        # @todo: is this a possible position to add in the if-case evaluation
        # for compression in case of a z-direction "damage broadcasting"?!
        #
        # get the sign of the normal component of each microplane
        # i.e. check if the microplane strain is negative (compression)
        # if yes ignore the state variable and explicitly set the phi value to 1.0
        # which corresponds to the initial (undamaged) value of e_max = 0.
        # magnitude of the normal strain vector for each microplane
        # @todo: faster numpy functionality possible? Note that the calculation of 'e_N_arr' is calculated twice!
        #
#        e_vct_arr = self._get_e_vct_arr(eps_app_eng)
#        e_N_arr = array([ dot(e_vct, mpn) for e_vct, mpn in zip(e_vct_arr, self._MPN) ])
#        bool_arr = e_N_arr < 0.
#        e_max_arr[bool_arr] = 0.

        return self.get_phi_arr(sctx, e_max_arr)

    def _get_phi_mtx(self, sctx, eps_app_eng):
        '''
        Returns the 2nd order damage tensor 'phi_mtx'
        '''
        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)
        # integration terms for each microplanes
        # @todo: faster numpy functionality possible?
        phi_mtx_arr = array([phi_arr[i] * self._MPNN[i, :, :] * self._MPW[i]
                             for i in range(0, self.n_mp)])
        # sum of contributions from all microplanes
        # sum over the first dimension (over the microplanes)
        phi_mtx = phi_mtx_arr.sum(0)
        return phi_mtx

    def _get_psi_mtx(self, sctx, eps_app_eng):
        '''
        Returns the 2nd order damage effect tensor 'psi_mtx'
        '''
        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)
        # integration terms for each microplanes

        psi_mtx_arr = array([1. / phi_arr[i] * self._MPNN[i, :, :] * self._MPW[i]
                             for i in range(0, self.n_mp)])

        # sum of contributions from all microplanes
        # sum over the first dimension (over the microplanes)
        psi_mtx = psi_mtx_arr.sum(0)
        return psi_mtx

    def _get_beta_tns_product_type(self, phi_mtx):
        '''
        Returns the 4th order damage tensor 'beta4' using product-type symmetrization
        (cf. [Baz97], Eq.(87))
        '''
        n_dim = self.n_dim
        # Get the direction of the principle damage coordinates (pdc):
        phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
        phi_eig_value_real = array([pe.real for pe in phi_eig_value])
        # transform phi_mtx to PDC:
        # (assure that besides the diagonal the entries are exactly zero)
        phi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
        for i in range(n_dim):
            phi_pdc_mtx[i, i] = phi_eig_value_real[i]
        # w_mtx = tensorial square root of the second order damage tensor:
        w_pdc_mtx = arr_sqrt(phi_pdc_mtx)
#        print "w_pdc_mtx", w_pdc_mtx
        # transform the matrix w back to x-y-coordinates:
        w_mtx = dot(dot(phi_eig_mtx, w_pdc_mtx), transpose(phi_eig_mtx))
        # beta_ijkl = w_ik * w_jl (cf. [Baz 97])
        # exploiting numpy-functionality (faster).
        # Method 'outer' returns beta_ijkl = w_ij * w_kl,
        # therefore the axis j and k need to be swapped
        beta4_ = outer(w_mtx, w_mtx).reshape(n_dim, n_dim, n_dim, n_dim)
        beta4 = beta4_.swapaxes(1, 2)
        return beta4

    def _get_beta_tns_sum_type(self, phi_mtx):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        n_dim = self.n_dim
        delta = self.identity_tns

        # The following line correspond to the tensorial expression:
        #
        #        beta4 = zeros((n_dim,n_dim,n_dim,n_dim),dtype=float)
        #        for i in range(0,n_dim):
        #            for j in range(0,n_dim):
        #                for k in range(0,n_dim):
        #                    for l in range(0,n_dim):
        #                        beta4[i,j,k,l] = 0.25 * ( phi_mtx[i,k] * delta[j,l] + phi_mtx[i,l] * delta[j,k] +\
        #                                                  phi_mtx[j,k] * delta[i,l] + phi_mtx[j,l] * delta[i,k] )
        #

        # use numpy functionality to evaluate [Jir99], Eq.(21)
        beta_ijkl = outer(phi_mtx, delta).reshape(n_dim, n_dim, n_dim, n_dim)
        beta_ikjl = beta_ijkl.swapaxes(1, 2)
        beta_iljk = beta_ikjl.swapaxes(2, 3)
        beta_jlik = beta_iljk.swapaxes(0, 1)
        beta_jkil = beta_jlik.swapaxes(2, 3)
        beta4 = 0.25 * (beta_ikjl + beta_iljk + beta_jkil + beta_jlik)

        return beta4

    def _get_M_tns_product_type(self, psi_mtx):
        '''
        Returns the 4th order damage effect tensor 'M4' using product-type symmetrization
        '''
        n_dim = self.n_dim
        # Get the direction orthogonal to the principle damage coordinates (pdc):
        # @todo: is this direction orthogonal? Which one do we want?
        psi_eig_value, psi_eig_mtx = eigh(psi_mtx)
        psi_eig_value_real = array([pe.real for pe in psi_eig_value])
        # transform phi_mtx to PDC:
        # (assure that besides the diagonal the entries are exactly zero)
        psi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
        for i in range(n_dim):
            psi_pdc_mtx[i, i] = psi_eig_value_real[i]
        # second order damage effect tensor:
        w_hat_pdc_mtx = arr_sqrt(psi_pdc_mtx)
#        print "w_hat_pdc_mtx", w_hat_pdc_mtx
        # transform the matrix w back to x-y-coordinates:
        w_hat_mtx = dot(
            dot(psi_eig_mtx, w_hat_pdc_mtx), transpose(psi_eig_mtx))
#        print "w_hat_mtx", w_hat_mtx
        # M_ijkl = w_hat_ik * w_hat_lj (cf. Eq.(5.62) Script Prag Jirasek (2007))
        #        = w_hat_ik * w_hat_jl (w is a symmetric tensor)
        # Exploiting numpy-functionality using the
        # method 'outer' (returns M_ijkl = w_hat_ij * w_hat_kl),
        # therefore the axis j and k need to be swapped
        M4_ = outer(w_hat_mtx, w_hat_mtx).reshape(n_dim, n_dim, n_dim, n_dim)
        M4 = M4_.swapaxes(1, 2)
        return M4

    def _get_M_tns_sum_type(self, psi_mtx):
        '''
        Returns the 4th order damage effect tensor 'M4' using sum-type symmetrization
        '''
        n_dim = self.n_dim
        delta = self.identity_tns

        # The line below corresponds to the tensorial expression
        #
        # (cf. [Jir99], Eq.(30))
        #        M4 = zeros((n_dim,n_dim,n_dim,n_dim),dtype=float)
        #        for i in range(0,n_dim):
        #            for j in range(0,n_dim):
        #                for k in range(0,n_dim):
        #                    for l in range(0,n_dim):
        #                        M4[i,j,k,l] = 0.25 * ( psi_mtx[i,k] * delta[j,l] + psi_mtx[i,l] * delta[j,k] +\
        # psi_mtx[j,k] * delta[i,l] + psi_mtx[j,l] * delta[i,k] )

        # use numpy functionality to evaluate [Jir99], Eq.(21)
        M_ijkl = outer(psi_mtx, delta).reshape(n_dim, n_dim, n_dim, n_dim)
        M_ikjl = M_ijkl.swapaxes(1, 2)
        M_iljk = M_ikjl.swapaxes(2, 3)
        M_jlik = M_iljk.swapaxes(0, 1)
        M_jkil = M_jlik.swapaxes(2, 3)
        M4 = 0.25 * (M_ikjl + M_iljk + M_jkil + M_jlik)

        return M4

    #-------------------------------------------------------------------------
    # Dynamic configuration of the damage tensor evaluation
    #-------------------------------------------------------------------------
    # the traits _get_beta_tns and _get_M_tns are are reset
    # depending on the selected type of symmetrization.

    _get_beta_tns = Callable(transient=True)

    def __get_beta_tns_default(self):
        return self._get_damage_eval_methods()[0]

    _get_M_tns = Callable(transient=True)

    def __get_M_tns_default(self):
        return self._get_damage_eval_methods()[1]

    @on_trait_change('symmetrization')
    def _reset_damage_eval_methods(self):
        self._get_beta_tns, self._get_M_tns = self._get_damage_eval_methods()

    def _get_damage_eval_methods(self):
        if self.symmetrization == 'product-type':
            return (self._get_beta_tns_product_type, self._get_M_tns_product_type)
        elif self.symmetrization == 'sum-type':
            return (self._get_beta_tns_sum_type, self._get_M_tns_sum_type)
        else:
            raise ValueError('Bad symmetrization tag')

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------
    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        # -----------------------------------------------------------------------------------------------
        # check if average strain is to be used for damage evaluation
        # -----------------------------------------------------------------------------------------------
        if eps_avg != None:
            pass
        else:
            eps_avg = eps_app_eng

        # -----------------------------------------------------------------------------------------------
        # for debugging purposes only: if elastic_debug is switched on, linear elastic material is used
        # -----------------------------------------------------------------------------------------------
        if self.elastic_debug:
            # NOTE: This must be copied otherwise self.D2_e gets modified when
            # essential boundary conditions are inserted
            D2_e = copy(self.D2_e)
            sig_eng = tensordot(D2_e, eps_app_eng, [[1], [0]])
            return sig_eng, D2_e

        # -----------------------------------------------------------------------------------------------
        # update state variables
        # -----------------------------------------------------------------------------------------------
        if sctx.update_state_on:
            eps_n = eps_avg - d_eps
            e_max = self._get_state_variables(sctx, eps_n)
            sctx.mats_state_array[:] = e_max

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        if self.regularization:
            h = self.get_regularizing_length(sctx, eps_app_eng)
            self.phi_fn.h = h

        #----------------------------------------------------------------------
        # stiffness version:
        #----------------------------------------------------------------------
        if self.model_version == 'stiffness':

            #------------------------------------------------------------------
            # Damage tensor (2th order):
            #------------------------------------------------------------------

            phi_mtx = self._get_phi_mtx(sctx, eps_avg)

            #------------------------------------------------------------------
            # Damage tensor (4th order) using product- or sum-type symmetrization:
            #------------------------------------------------------------------
            beta4 = self._get_beta_tns(phi_mtx)

            #------------------------------------------------------------------
            # Damaged stiffness tensor calculated based on the damage tensor beta4:
            #------------------------------------------------------------------
            # (cf. [Jir99] Eq.(7): C = beta * D_e * beta^T),
            # minor symmetry is tacitly assumed ! (i.e. beta_ijkl = beta_jilk)
            D4_mdm = tensordot(
                beta4, tensordot(self.D4_e, beta4, [[2, 3], [2, 3]]), [[2, 3], [0, 1]])

            #------------------------------------------------------------------
            # Reduction of the fourth order tensor to a matrix assuming minor and major symmetry:
            #------------------------------------------------------------------
            D2_mdm = self.map_tns4_to_tns2(D4_mdm)

        #----------------------------------------------------------------------
        # compliance version:
        #----------------------------------------------------------------------
        elif self.model_version == 'compliance':

            #------------------------------------------------------------------
            # Damage effect tensor (2th order):
            #------------------------------------------------------------------

            psi_mtx = self._get_psi_mtx(sctx, eps_avg)

            #------------------------------------------------------------------
            # Damage effect tensor (4th order) using product- or sum-type-symmetrization:
            #------------------------------------------------------------------
            M4 = self._get_M_tns(psi_mtx)

            #------------------------------------------------------------------
            # Damaged compliance tensor calculated based on the damage effect tensor M4:
            #------------------------------------------------------------------
            # (cf. [Jir99] Eq.(8): C = M^T * C_e * M,
            # minor symmetry is tacitly assumed ! (i.e. M_ijkl = M_jilk)
            C4_mdm = tensordot(
                M4, tensordot(self.C4_e, M4, [[2, 3], [0, 1]]), [[0, 1], [0, 1]])

            #------------------------------------------------------------------
            # Reduction of the fourth order tensor to a matrix assuming minor and major symmetry:
            #------------------------------------------------------------------
            C2_mdm = self.map_tns4_to_tns2(C4_mdm)
            D2_mdm = inv(self.compliance_mapping(C2_mdm))

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------

        sig_eng = tensordot(D2_mdm, eps_app_eng, [[1], [0]])
        return sig_eng, D2_mdm

    #-------------------------------------------------------------------------
    # Control variables and update state method
    #-------------------------------------------------------------------------
    def new_cntl_var(self):
        return zeros(self.n_eng, float_)

    def new_resp_var(self):
        return zeros(self.n_eng, float_)

    def update_state(self, sctx, eps_app_eng):
        '''
        Update state method is called upon an accepted time-step.
        Here just set the flag on to make the update afterwards in the method itself.
        '''
        print('in update-state')
        # self.update_state_on = True

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------
    def get_eps_app_v3d(self, sctx, eps_app_eng):
        return eps_app_eng

    def get_sig_app_v3d(self, sctx, eps_app_eng, *args, **kw):
        # @TODO
        # the stress calculation is performed twice - it might be
        # cached. But not in the spatial integration scheme.
        sig_app_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return sig_app_eng

    def get_eps_app(self, sctx, eps_app_eng):
        return self.map_eps_eng_to_mtx(eps_app_eng)

    def get_sig_app(self, sctx, eps_app_eng, *args, **kw):
        # @TODO
        # the stress calculation is performed twice - it might be
        # cached. But not in the spatial integration scheme.
        sig_app_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        # return sig_app_eng
        return self.map_sig_eng_to_mtx(sig_app_eng)

    def get_microplane_integrity(self, sctx, eps_app_eng):
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)
        return phi_arr

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2)])

    def get_phi_mtx(self, sctx, eps_app_eng):
        return self._get_phi_mtx(sctx, eps_app_eng)

    def get_phi_pdc(self, sctx, eps_app_eng):
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng)
        # Get the direction of the principle damage coordinates (pdc):
        phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
        phi_eig_value_real = array([pe.real for pe in phi_eig_value])
        # return the minimum value of the eigenvalues of the integrity tensor
        # (= values of integrity in the principle direction)
        #
        min_phi = np.min(phi_eig_value_real)
        max_omega = (1.0 - min_phi)
        return array([max_omega])

    def get_phi_pdc2(self, sctx, eps_app_eng):
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng)
        # Get the direction of the principle damage coordinates (pdc):
        phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
        phi_eig_value_real = array([pe.real for pe in phi_eig_value])
        # return the minimum value of the eigenvalues of the integrity tensor
        # (= values of integrity in the principle direction)
        #
        min_phi = np.min(phi_eig_value_real)
        max_omega = (1.0 - min_phi ** 2)
        return array([max_omega])

    # ------------------------------------------
    # SUBSIDARY METHODS used only for the response tracer:
    # ------------------------------------------
    # Projection: Methods used for projection:

    #  '_get_e_vct_arr'  -  has been defined above (see 'get_corr_pred')
    #   (also used as subsidary method for '_get_e_s_vct_arr'.)

    def _get_s_vct_arr(self, sig_eng):
        '''
        Projects the stress tensor onto the microplanes and
        returns a list of microplane strain vectors.
        (Subsidary method for '_get_e_s_vct_arr'.)
        '''
        # Switch from engineering notation to tensor notation for the apparent
        # strains
        sig_mtx = self.map_sig_eng_to_mtx(sig_eng)
        # Projection of apparent strain onto the individual microplanes
        # slower: s_vct_arr = array( [ dot( sig_mtx, mpn ) for mpn in self._MPN
        # ] )
        s_vct_arr = dot(self._MPN, sig_mtx)
        return s_vct_arr

    # Equiv: Equivalent microplane parts:

    #  '_get_e_equiv_arr'  -  has been defined above (see 'get_corr_pred')

    def _get_s_equiv_arr(self, s_vct_arr):
        '''
        Get microplane equivalent stress.
        (Subsidary method for '_get_e_s_equiv_arr'.)
        '''
        # The same method is used to calculate the equivalent stresses
        # and the equivalent strains
        s_equiv_arr = self._get_e_equiv_arr(s_vct_arr)
        return s_equiv_arr

    # N: normal microplane parts

    def _get_e_N_arr(self, e_vct_arr):
        '''
        Returns a list of the microplane normal strains (scalar)
        based on the list of microplane strain vectors
        (Subsidary method for '_get_e_s_N_arr'.) '''
        # magnitude of the normal strain vector for each microplane
        e_N_arr = array([dot(e_vct, mpn)
                         for e_vct, mpn in zip(e_vct_arr, self._MPN)])
        return e_N_arr
        # @todo: check if the direct calculation of e_N using MPNN makes sense.
        #        Note that e_T needs the calculation of e_N_vct !
        #        s_N_arr = array( [ s_vct_arr[i] * self._MPNN[i,:,:] for i in range(0,self.n_mp) ] )
        #        return s_N_arr[0,:].flatten()

    def _get_s_N_arr(self, s_vct_arr):
        # the same method is used for the calculation of the
        # normal parts of the microplane strain and stress vector
        s_N_arr = self._get_e_N_arr(s_vct_arr)
        return s_N_arr

    # T: tangential microplane parts

    def _get_e_T_arr(self, e_vct_arr):
        '''
        Returns a list of the microplane shear strains (scalar)
        based on the list of microplane strain vectors
        (Subsidary method for '_get_e_s_T_arr'.)
        '''
        # magnitude of the normal strain vector for each microplane
        e_N_arr = self._get_e_N_arr(e_vct_arr)
        # normal strain vector for each microplane
        e_N_vct_arr = array([self._MPN[i, :] * e_N_arr[i]
                             for i in range(0, self.n_mp)])
        # tangential strain vector for each microplane
        e_T_vct_arr = e_vct_arr - e_N_vct_arr
        # squared tangential strain vector for each microplane
        e_TT_arr = array([inner(e_T_vct, e_T_vct) for e_T_vct in e_T_vct_arr])
        # equivalent strain for each microplane
        e_T_arr = arr_sqrt(e_TT_arr)
        return e_T_arr

    def _get_s_T_arr(self, s_vct_arr):
        # the same method is used for the calculation of the
        # tangential parts of the microplane strain and stress vector
        s_T_arr = self._get_e_T_arr(s_vct_arr)
        return s_T_arr

    # -------------------------------------------------------------
    # Get the microplane strain and stresses either based on a
    # consistently derived model-version or the double constraint
    # -------------------------------------------------------------
    def _get_e_s_vct_arr(self, sctx, eps_app_eng):
        '''
        Returns two arrays containing the microplane strain and stress vectors
        either assuming a double constraint or consistently derived based on the specified model version
        '''
        #----------------------------------------------------------------------
        # DOUBLE CONSTRAINT
        #----------------------------------------------------------------------
        if self.double_constraint:
            '''
            Return a pair of microplane stress and strain vectors based on the
            double constrain, i.e kinematic AND static constraint !
            The connection between the apparent strain and stress tensor is established
            with D_mdm based on the chosen model version (e.g. stiffness or compliance)
            '''
            # microplane strain vectors obtained by projection (kinematic
            # constraint):
            e_app_vct_arr = self._get_e_vct_arr(eps_app_eng)
            # get the corresponding macroscopic stresses
            sig_app_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
            # microplane stress vectors obtained by projection (static
            # constraint)
            s_app_vct_arr = self._get_s_vct_arr(sig_app_eng)
            return e_app_vct_arr, s_app_vct_arr

        #----------------------------------------------------------------------
        # CONSISTENTLY DERIVED pair of microplane strain and stress vectors
        #----------------------------------------------------------------------
        '''
        Returns two arrays containing the microplane strain and stress vectors
        consistently derived based on the specified model version, i.e. either 'stiffness' or 'compliance'
        '''
        #-------------------
        # stiffness version:
        #-------------------
        if self.model_version == 'stiffness':
            # microplane equivalent strains obtained by projection (kinematic
            # constraint)
            e_app_vct_arr = self._get_e_vct_arr(eps_app_eng)

            # microplane equivalent stresses calculated based on corresponding 'beta' and 'phi_mtx'
            # 2nd order damage tensor:
            phi_mtx = self._get_phi_mtx(sctx, eps_app_eng)
            # 4th order damage tensor:
            if self.symmetrization == 'product-type':
                beta4 = self._get_beta_tns_product_type(phi_mtx)
            elif self.symmetrization == 'sum-type':
                beta4 = self._get_beta_tns_sum_type(phi_mtx)

            # apparent strain tensor:
            eps_app_mtx = self.map_eps_eng_to_mtx(eps_app_eng)
            # effective strain tensor:
            eps_eff_mtx = tensordot(beta4, eps_app_mtx, [[0, 1], [0, 1]])
            # effective stress tensor:
            sig_eff_mtx = tensordot(self.D4_e, eps_eff_mtx, [[2, 3], [0, 1]])
            # effective microplane stresses obtained by projection (static
            # constraint)
            s_eff_vct_arr = array([dot(sig_eff_mtx, mpn) for mpn in self._MPN])
            # microplane scalar damage variable (integrity):
            phi_arr = self._get_phi_arr(sctx, eps_app_eng)
            # apparent microplane stresses
            s_app_vct_arr = array(
                [dot(phi, s_eff_vct) for phi, s_eff_vct in zip(phi_arr, s_eff_vct_arr)])

        #--------------------
        # compliance version:
        #--------------------
        if self.model_version == 'compliance':
            # get the corresponding macroscopic stresses
            sig_app_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)

            # microplane equivalent stress obtained by projection (static
            # constraint)
            s_app_vct_arr = self._get_s_vct_arr(sig_app_eng)

            # microplane equivalent strains calculated based on corresponding 'M' and 'psi_mtx'
            # 2nd order damage effect tensor:
            psi_mtx = self._get_psi_mtx(sctx, eps_app_eng)
            # 4th order damage effect tensor:
            if self.symmetrization == 'product-type':
                M4 = self._get_M_tns_product_type(psi_mtx)
            elif self.symmetrization == 'sum-type':
                M4 = self._get_M_tns_sum_type(psi_mtx)

            # apparent stress tensor:
            sig_app_mtx = self.map_sig_eng_to_mtx(sig_app_eng)
            # effective stress tensor:
            sig_eff_mtx = tensordot(M4, sig_app_mtx, [[2, 3], [0, 1]])
            # effective strain tensor:
            eps_eff_mtx = tensordot(self.C4_e, sig_eff_mtx, [[2, 3], [0, 1]])
            # effective microplane strains obtained by projection (kinematic
            # constraint)
            e_eff_vct_arr = array([dot(eps_eff_mtx, mpn) for mpn in self._MPN])
            # microplane scalar damage variable (integrity):
            phi_arr = self._get_phi_arr(sctx, eps_app_eng)
            psi_arr = 1. / phi_arr
            # apparent microplane strains
            e_app_vct_arr = array(
                [dot(psi, e_eff_vct) for psi, e_eff_vct in zip(psi_arr, e_eff_vct_arr)])

        return e_app_vct_arr, s_app_vct_arr

    def _get_psi_arr_cv(self, sctx, e_max_arr):
        '''
        Returns a list of the compliance integrity factors for all microplanes, i.e. psi_i(e_max) = 1/phi_i(e_max)
        '''
        return 1. / self.get_phi_arr(sctx, e_max_arr)

    def _get_psi_mtx_cv(self, sctx, e_max_arr):
        '''
        Returns the 2nd order damage effect tensor 'psi_mtx'
        '''
        # scalar integrity factor for each microplane
        psi_arr = self._get_psi_arr_cv(sctx, e_max_arr)
        # integration terms for each microplanes
        psi_mtx_arr = array([psi_arr[i] * self._MPNN[i, :, :] * self._MPW[i]
                             for i in range(0, self.n_mp)])
        # sum of contributions from all microplanes
        # sum over the first dimension (over the microplanes)
        psi_mtx = psi_mtx_arr.sum(0)
        return psi_mtx

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # @todo: is an implicit formulation of CMDM possible? First outline see below
    #-------------------------------------------------------------------------
    #
    def get_lack_of_fit_psi_arr(self, sctx, e_max_arr_new, eps_app_eng):
        # state variables of last converged step at time t_i
        e_max_arr_old = sctx.mats_state_array

        # compare current values with state variables take max value in history
        e_max_arr = max(e_max_arr_old, e_max_arr_new)

        # get the corresponding macroscopic stresses for D_mtx at time t_i+1
        # for strain
        sig_app_eng, D_mtx_cv = self.get_corr_pred(
            sctx, e_max_arr, eps_app_eng, 0, 0, 0)
        psi_arr = self._get_psi_arr_cv(sctx, e_max_arr)
        psi_mtx_cv = self._get_psi_mtx(sctx, e_max_arr)
        if self.symmetrization == 'product-type':
            M4 = self._get_M_tns_product_type(psi_mtx_cv)
        elif self.symmetrization == 'sum-type':
            M4 = self._get_M_tns_sum_type(psi_mtx_cv)
        # apparent stress tensor:
        sig_app_mtx = self.map_sig_eng_to_mtx(sig_app_eng)
        # effective stress tensor:
        sig_eff_mtx = tensordot(M4, sig_app_mtx, [[2, 3], [0, 1]])
        # effective strain tensor:
        eps_eff_mtx = tensordot(self.C4_e, sig_eff_mtx, [[2, 3], [0, 1]])
        # effective microplane strains obtained by projection (kinematic
        # constraint)
        e_eff_vct_arr = array([dot(eps_eff_mtx, mpn) for mpn in self._MPN])
        # microplane scalar damage variable (integrity):
        # apparent microplane strains
        e_app_vct_arr = array([dot(psi, e_eff_vct)
                               for psi, e_eff_vct in zip(psi_arr, e_eff_vct_arr)])
        # equivalent strain for each microplane
        e_equiv_arr = self._get_e_equiv_arr(e_app_vct_arr)
        # lack of fit in the state variables
        psi_arr_trial = self._get_psi_arr_cv(sctx, e_equiv_arr)
        lof = psi_arr_trial - psi_arr
        return lof

    # -------------------------------------------------------------
    # Get the values of the maximum microplane strains (state variables) iteratively
    # calculated within each iteration step of the global time loop algorithm fullfilling
    # also in the trial steps the consistently derived damage tensor based on the compliance
    # version (implicit formulation)
    # -------------------------------------------------------------
    def get_corr_pred_cv(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Returns two arrays containing the microplane strain and stress vectors
        consistently derived based on the specified model version, i.e. either 'stiffness' or 'compliance'
        '''
        # for compliance version only 
        #
        if self.model_version != 'compliance':
            raise ValueError('only valid for compliance version')

        raise ImportError('If this is needed include the import of the scipy package')
        #e_max_arr_new = brentq(e_max_arr_new, self.get_lack_of_fit_psi_arr(
        #    self, sctx, e_msax_arr_new, eps_app_eng))

        #self.update_state_variables(e_max_arr_new)

        # get the corresponding macroscopic stresses for D_mtx at time t_i+1
        # for strain
        #sig_app_eng, D_mtx_cv = self.get_corr_pred(
        #    sctx, e_max_arr_new, eps_app_eng, 0, 0, 0)

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    # Equiv:
    def _get_e_s_equiv_arr(self, sctx, eps_app_eng):
        '''
        Returns two arrays containing the corresponding equivalent
        microplane strains and stresses.
        '''
        e_app_vct_arr, s_app_vct_arr = self._get_e_s_vct_arr(sctx, eps_app_eng)

        # equivalent strain for each microplane
        e_equiv_arr = self._get_e_equiv_arr(e_app_vct_arr)
        # equivalent strain for each microplane
        s_equiv_arr = self._get_s_equiv_arr(s_app_vct_arr)
        return e_equiv_arr, s_equiv_arr

    def get_e_equiv_arr(self, sctx, eps_app_eng):
        '''
        Return a list of equivalent microplane strains.
        '''
        e_equiv_arr, s_equiv_arr = self._get_e_s_equiv_arr(sctx, eps_app_eng)
        return e_equiv_arr

    def get_s_equiv_arr(self, sctx, eps_app_eng):
        '''
        Return a list of equivalent microplane stresses.
        '''
        e_equiv_arr, s_equiv_arr = self._get_e_s_equiv_arr(sctx, eps_app_eng)
        return s_equiv_arr

    # N: normal microplane strain and stresses:
    def _get_e_s_N_arr(self, sctx, eps_app_eng):
        '''
        Returns two arrays containing the corresponding normal
        microplane strains and stress components.
        '''
        e_app_vct_arr, s_app_vct_arr = self._get_e_s_vct_arr(sctx, eps_app_eng)
        # normal strain for each microplane
        e_N_arr = self._get_e_N_arr(e_app_vct_arr)
        # normal strain for each microplane
        s_N_arr = self._get_s_N_arr(s_app_vct_arr)
        return e_N_arr, s_N_arr

    def get_e_N_arr(self, sctx, eps_app_eng):
        '''
        Return a list of normal microplane strains components.
        '''
        e_N_arr, s_N_arr = self._get_e_s_N_arr(sctx, eps_app_eng)
        return e_N_arr

    def get_s_N_arr(self, sctx, eps_app_eng):
        '''
        Return a list of normal microplane stresses components.
        '''
        e_N_arr, s_N_arr = self._get_e_s_N_arr(sctx, eps_app_eng)
        return s_N_arr

    # T: tangential microplane strain and stresses:
    def _get_e_s_T_arr(self, sctx, eps_app_eng):
        '''
        Returns two arrays containing the corresponding tangential
        microplane strains and stress components.
        '''
        e_app_vct_arr, s_app_vct_arr = self._get_e_s_vct_arr(sctx, eps_app_eng)
        # shear strain for each microplane
        e_T_arr = self._get_e_T_arr(e_app_vct_arr)
        # shear strain for each microplane
        s_T_arr = self._get_s_T_arr(s_app_vct_arr)
        return e_T_arr, s_T_arr

    def get_e_T_arr(self, sctx, eps_app_eng):
        '''
        Return a list of tangential microplane strains components.
        '''
        e_T_arr, s_T_arr = self._get_e_s_T_arr(sctx, eps_app_eng)
        return e_T_arr

    def get_s_T_arr(self, sctx, eps_app_eng):
        '''
        Return a list of tangential microplane stresses components.
        '''
        e_T_arr, s_T_arr = self._get_e_s_T_arr(sctx, eps_app_eng)
        return s_T_arr

# @todo: remove / old

# @todo: remove: this method and method 'get_integ' in 'polar_fn'
# old implementation: this assumes a decoupled reaction of all microplanes
    def get_fracture_energy_arr(self, sctx, e_max_arr):
        '''
        Get the microplane contributions to the fracture energy
        '''
        fracture_energy_arr = self.get_polar_fn_fracture_energy_arr(
            sctx, e_max_arr)
        return fracture_energy_arr

    def get_fracture_energy(self, sctx, eps_app_eng, *args, **kw):
        '''
        Get the macroscopic fracture energy as a weighted sum of all mircoplane contributions
        '''
        e_max_arr = self._get_state_variables(sctx, eps_app_eng)
        fracture_energy_arr = self.get_fracture_energy_arr(sctx, e_max_arr)
        fracture_energy = array([dot(self._MPW, fracture_energy_arr)], float)
        return fracture_energy

    def get_e_equiv_projection(self, sctx, eps_app_eng):
        '''
        Return a list of equivalent microplane strains as the projection of the strain tensor
        (kinematic constraint)
        '''
        e_vct_arr = self._get_e_vct_arr(eps_app_eng)
        e_equiv_arr = self._get_e_equiv_arr(e_vct_arr)
#        print 'e_equiv_arr:  ', e_equiv_arr
        return e_equiv_arr

    def get_max_omega_i2(self, sctx, eps_app_eng):
        '''
        Get maximum damage at all microplanes.
        '''
        min_phi = np.min(self._get_phi_arr(sctx, eps_app_eng))
        max_omega = 1. - min_phi ** 2
        if max_omega == 1.:
            print('max_omega_i2', max_omega)
#            print 'eps_app_eng', eps_app_eng
        return np.array([max_omega])

    def get_max_omega_i(self, sctx, eps_app_eng):
        '''
        Get maximum damage at all microplanes.
        '''
        min_phi = np.min(self._get_phi_arr(sctx, eps_app_eng))
        max_omega = 1. - min_phi
        if max_omega == 1.:
            print('max_omega_i', max_omega)
#            print 'eps_app_eng', eps_app_eng
        return np.array([max_omega])

    def get_omega_mtx(self, sctx, eps_app_eng, *args, **kw):
        '''
        Returns the 2nd order damage tensor 'phi_mtx'
        '''
        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)
        omega_arr = 1 - phi_arr ** 2
        # integration terms for each microplanes
        # @todo: faster numpy functionality possible?
        omega_mtx_arr = array([omega_arr[i] * self._MPNN[i, :, :] * self._MPW[i]
                               for i in range(0, self.n_mp)])
        # sum of contributions from all microplanes
        # sum over the first dimension (over the microplanes)
        omega_mtx = omega_mtx_arr.sum(0)

        return omega_mtx
# ##

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'eps_app_v3d': self.get_eps_app_v3d,
                'sig_app_v3d': self.get_sig_app_v3d,
                'eps_app': self.get_eps_app,
                'strain_energy': self.get_strain_energy,
                'sig_app': self.get_sig_app,
                'sig_norm': self.get_sig_norm,
                'phi_mtx': self.get_phi_mtx,
                'omega_mtx': self.get_omega_mtx,
                'phi_pdc': self.get_phi_pdc,
                'phi_pdc2': self.get_phi_pdc2,
                'microplane_damage': RTraceEval(eval=self.get_microplane_integrity,
                                                ts=self),

                'e_equiv_arr': RTraceEval(eval=self.get_e_equiv_arr,
                                          ts=self),
                's_equiv_arr': RTraceEval(eval=self.get_s_equiv_arr,
                                          ts=self),
                'e_N_arr': RTraceEval(eval=self.get_e_N_arr,
                                      ts=self),
                's_N_arr': RTraceEval(eval=self.get_s_N_arr,
                                      ts=self),
                'e_T_arr': RTraceEval(eval=self.get_e_T_arr,
                                      ts=self),
                's_T_arr': RTraceEval(eval=self.get_s_T_arr,
                                      ts=self),

                'equiv_projection': RTraceEval(eval=self.get_e_equiv_projection,
                                               ts=self),
                'fracture_energy_arr': self.get_fracture_energy_arr,
                'fracture_energy': self.get_fracture_energy,
                'max_omega_i': self.get_max_omega_i,
                'max_omega_i2': self.get_max_omega_i2
                }

    #-------------------------------------------------------------------------
    # List of response tracers to be constructed within the mats_explorer
    #-------------------------------------------------------------------------
    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATSXDMicroplaneDamage, self)._get_explorer_config()

        from ibvpy.rtrace.rt_dof import RTDofGraph
        from ibvpy.rtrace.rt_dof import RTraceArraySnapshot
        from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm_rtrace_Gf_mic import \
            MATS2DMicroplaneDamageTraceGfmic, \
            MATS2DMicroplaneDamageTraceEtmic, MATS2DMicroplaneDamageTraceUtmic
        from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm_rtrace_Gf_mac import \
            MATS2DMicroplaneDamageTraceGfmac, \
            MATS2DMicroplaneDamageTraceEtmac, MATS2DMicroplaneDamageTraceUtmac

        c['rtrace_list'] += [
            RTDofGraph(name='time - microplane damage',
                        var_x='time', idx_x=0,
                        var_y='microplane_damage', idx_y=0,
                        record_on='update'),

            # e_equiv, s_equiv
            RTDofGraph(name='e_equiv - s_equiv',
                        var_x='e_equiv_arr', idx_x=0,
                        var_y='s_equiv_arr', idx_y=0,
                        record_on='update'),

            # e_N, s_N:
            RTDofGraph(name='e_N - s_N',
                        var_x='e_N_arr', idx_x=0,
                        var_y='s_N_arr', idx_y=0,
                        record_on='update'),

            # e_T, s_T:
            RTDofGraph(name='e_T - s_T',
                        var_x='e_T_arr', idx_x=0,
                        var_y='s_T_arr', idx_y=0,
                        record_on='update'),

            RTraceArraySnapshot(name='equiv_projection',
                                var='equiv_projection',
                                record_on='update'),

            RTraceArraySnapshot(name='microplane damage',
                                var='microplane_damage',
                                record_on='update'),

            RTraceArraySnapshot(name='e_equiv',
                                var='e_equiv_arr',
                                record_on='update'),
            RTraceArraySnapshot(name='s_equiv',
                                var='s_equiv_arr',
                                record_on='update'),
            RTraceArraySnapshot(name='e_N',
                                var='e_N_arr',
                                record_on='update'),
            RTraceArraySnapshot(name='s_N',
                                var='s_N_arr',
                                record_on='update'),
            RTraceArraySnapshot(name='e_T',
                                var='e_T_arr',
                                record_on='update'),
            RTraceArraySnapshot(name='s_T',
                                var='s_T_arr',
                                record_on='update'),

            # G_f_mic: microplane fracture energy:
            #                                       MATS2DMicroplaneDamageTraceGfmic(name = 'G_f_mic_equiv',
            #                                                                        var_x = 'e_equiv_arr', idx_x = 0,
            #                                                                        var_y = 's_equiv_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceGfmic(name = 'G_f_mic_N',
            #                                                                        var_x = 'e_N_arr', idx_x = 0,
            #                                                                        var_y = 's_N_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceGfmic(name = 'G_f_mic_T',
            #                                                                        var_x = 'e_T_arr', idx_x = 0,
            #                                                                        var_y = 's_T_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            # E_t_mic: microplane total energy
            #                                       MATS2DMicroplaneDamageTraceEtmic(name = 'E_t_mic_equiv',
            #                                                                        var_x = 'e_equiv_arr', idx_x = 0,
            #                                                                        var_y = 's_equiv_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceEtmic(name = 'E_t_mic_N',
            #                                                                        var_x = 'e_N_arr', idx_x = 0,
            #                                                                        var_y = 's_N_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceEtmic(name = 'E_t_mic_T',
            #                                                                        var_x = 'e_T_arr', idx_x = 0,
            #                                                                        var_y = 's_T_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            # U_t_mic: microplane elastic energy
            #                                       MATS2DMicroplaneDamageTraceUtmic(name = 'U_t_mic_equiv',
            #                                                                        var_x = 'e_equiv_arr', idx_x = 0,
            #                                                                        var_y = 's_equiv_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceUtmic(name = 'U_t_mic_N',
            #                                                                        var_x = 'e_N_arr', idx_x = 0,
            #                                                                        var_y = 's_N_arr', idx_y = 0,
            #                                                                        record_on = 'update' ),
            #                                       MATS2DMicroplaneDamageTraceUtmic(name = 'U_t_mic_T',
            #                                                                        var_x = 'e_T_arr', idx_x = 0,
            #                                                                        var_y = 's_T_arr', idx_y = 0,
            # record_on = 'update' ),

            # direction 11:
            # G_f_mac: macroscopic fracture energy:
            MATS2DMicroplaneDamageTraceGfmac(name='G_f_mac_11',

                                             var_x='eps_app', idx_x=0,
                                             var_y='sig_app', idx_y=0,
                                             record_on='update'),
            # E_t_mac: macroscopic total energy:
            MATS2DMicroplaneDamageTraceEtmac(name='E_t_mac_11',
                                             var_x='eps_app', idx_x=0,
                                             var_y='sig_app', idx_y=0,
                                             record_on='update'),
            # U_t_mac: macroscopic elastic energy:
            MATS2DMicroplaneDamageTraceUtmac(name='U_t_mac_11',
                                             var_x='eps_app', idx_x=0,
                                             var_y='sig_app', idx_y=0,
                                             record_on='update'),

            # direction 22:
            # G_f_mac: macroscopic fracture energy:
            #                                       MATS2DMicroplaneDamageTraceGfmac(name = 'G_f_mac_22',
            #                                                                        var_x = 'eps_app', idx_x = 1,
            #                                                                        var_y = 'sig_app', idx_y = 1,
            #                                                                        record_on = 'update' ),
            # E_t_mac: macroscopic total energy:
            #                                       MATS2DMicroplaneDamageTraceEtmac(name = 'E_t_mac_22',
            #                                                                        var_x = 'eps_app', idx_x = 1,
            #                                                                        var_y = 'sig_app', idx_y = 1,
            #                                                                        record_on = 'update' ),
            # U_t_mac: macroscopic elastic energy:
            #                                       MATS2DMicroplaneDamageTraceUtmac(name = 'U_t_mac_22',
            #                                                                        var_x = 'eps_app', idx_x = 1,
            #                                                                        var_y = 'sig_app', idx_y = 1,
            #                                                                        record_on = 'update' ),
            #
            # direction 12:
            # G_f_mac: macroscopic fracture energy:
            #                                       MATS2DMicroplaneDamageTraceGfmac(name = 'G_f_mac_12',
            #                                                                        var_x = 'eps_app', idx_x = 2,
            #                                                                        var_y = 'sig_app', idx_y = 2,
            #                                                                        record_on = 'update' ),
            # E_t_mac: macroscopic total energy:
            #                                       MATS2DMicroplaneDamageTraceEtmac(name = 'E_t_mac_12',
            #                                                                        var_x = 'eps_app', idx_x = 2,
            #                                                                        var_y = 'sig_app', idx_y = 2,
            #                                                                        record_on = 'update' ),
            # U_t_mac: macroscopic elastic energy:
            #                                       MATS2DMicroplaneDamageTraceUtmac(name = 'U_t_mac_12',
            #                                                                        var_x = 'eps_app', idx_x = 2,
            #                                                                        var_y = 'sig_app', idx_y = 2,
            # record_on = 'update' ),

            #                                       RTraceArraySnapshot(name = 'fracture energy contributions',
            #                                                           var = 'fracture_energy_arr',

            # record_on = 'update' ),

            # decoupled energy contributions for G_f
            #             RTDofGraph(name = 'time - G_f',
            #                          var_x = 'time', idx_x = 0,
            #                          var_y = 'fracture_energy', idx_y = 0,
            #                          record_on = 'update' ),
            #                                     ###
            RTDofGraph(name='time - sig_norm',
                        var_x='time', idx_x=0,
                        var_y='sig_norm', idx_y=0,
                        record_on='update'),
            #                                       RTDofGraph(name = 'time - phi_pdc',
            #                                                   var_x = 'time', idx_x = 0,
            #                                                   var_y = 'phi_pdc', idx_y = 0,
            #                                                   record_on = 'update' ),
            # e_equiv_projection:
            #                                       RTDofGraph(name = 'e_equiv_projection - s_equiv',
            #                                                   var_x = 'equiv_projection', idx_x = 0,
            #                                                   var_y = 's_equiv', idx_y = 0,
            # record_on = 'update' ),

        ]
        return c

# @todo: - temporary alias rename the class and test it all

MATS2DMicroplaneDamage = MATSXDMicroplaneDamage
MA2DCompositeMicroplaneDamage = MATS2DMicroplaneDamage

MATS2DMicroplaneDamage = MATS2DMicroplaneDamage
MATS2DCompositeMicroplaneDamage = MA2DCompositeMicroplaneDamage

#-------------------------------
# @todo's:
#-------------------------------
#
# 1)Renaming:
#    Check if further renaming would be helpful (propositions):
#  - all vectorized versions with suffix '_vectorized'
#
# 2) Check if caching of n_mp is not working properly. It seams to work now.
#    What was the problem?
#
# 4) Note: parameters specified above correspond to the parameters for paper CST2008.
#    There is a deviation in the value for the stresses under tension stiffening for eps = 2E-3
#    it's about 7 MPa instead of 12 as is the case in the paper. Choices for 'stress_state'
#    and 'symmetrization' are not relevant in that context. What is the reason?

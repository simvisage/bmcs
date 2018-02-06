'''
Created on 29.03.2017

@author: abaktheer

Microplane damage model 2D - Jirasek [1999]
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, zeros,\
    einsum, zeros_like,\
    identity, linspace, hstack, \
    sqrt

from traits.api import \
    Constant, implements,\
    Float, HasTraits, \
    Property, cached_property
import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr


class MATSEvalMicroplaneFatigue(HasTraits):

    E = Float(34000,
              label="E",
              desc="Elastic modulus",
              enter_set=True,
              auto_set=False)

    nu = Float(0.2,
               label="G",
               desc="poisson's ratio",
               enter_set=True,
               auto_set=False)

    ep = Float(59e-6,
               label="a",
               desc="Lateral pressure coefficient",
               enter_set=True,
               auto_set=False)

    ef = Float(250e-6,
               label="a",
               desc="Lateral pressure coefficient",
               enter_set=True,
               auto_set=False)

    c_T = Float(0.00,
                label="a",
                desc="Lateral pressure coefficient",
                enter_set=True,
                auto_set=False)

    def _get_phi(self, e, sctx):

        phi = zeros(len(e))

        for i in range(0, len(e)):

            if e[i] >= self.ep:
                phi[i] = sqrt(
                    (self.ep / e[i]) * np.exp(- (e[i] - self.ep) / (self.ef - self.ep)))
            else:
                phi[i] = 1.0

        return phi


class MATSXDMicroplaneDamageFatigueJir(MATSEvalMicroplaneFatigue):

    '''
    Microplane Damage Fatigue Model.
    '''
    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------
    D4_e = Property

    def _get_D4_e(self):
        # Return the elasticity tensor
        return self._get_D_abef()

    #-------------------------------------------------------------------------
    # MICROPLANE-Kinematic constraints
    #-------------------------------------------------------------------------

    # get the dyadic product of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals

        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')

    @cached_property
    def _get__MPTT(self):
        # Third order tangential tensor for each microplane
        delta = identity(2)
        MPTT_nijr = 0.5 * (einsum('ni,jr -> nijr', self._MPN, delta) +
                           einsum('nj,ir -> njir', self._MPN, delta) - 2 *
                           einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN))
        return MPTT_nijr

    def _get_e_vct_arr(self, eps_ab):
        # Projection of apparent strain onto the individual microplanes
        e_ni = einsum('nj,ji->ni', self._MPN, eps_ab)
        return e_ni

    def _get_e_N_arr(self, e_vct_arr):
        # get the normal strain array for each microplane
        eN_n = einsum('ni,ni->n', e_vct_arr, self._MPN)
        return eN_n

    def _get_e_T_vct_arr(self, e_vct_arr):
        # get the tangential strain vector array for each microplane
        eN_n = self._get_e_N_arr_2(e_vct_arr)
        eN_vct_ni = einsum('n,ni->ni', eN_n, self._MPN)
        print 'eN_n', eN_n.shape
        print 'eN_vct_ni', eN_vct_ni.shape
        print 'e_vct_arr', e_vct_arr.shape

        return e_vct_arr - eN_vct_ni

    def _get_e_equiv_arr(self, e_vct_arr):
        '''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_arr = self._get_e_N_arr(e_vct_arr)
        # print 'e_N_arr', e_N_arr
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_arr = (np.abs(e_N_arr) + e_N_arr) / 2
        # normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_vct_arr = einsum('n,ni -> ni', e_N_arr, self._MPN)
        # print 'e_N_vct_arr', e_N_vct_arr
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_vct_arr = e_vct_arr - e_N_vct_arr
        # squared tangential strain vector for each microplane
        e_TT_arr = einsum('ni,ni -> n', e_T_vct_arr, e_T_vct_arr)
        # equivalent strain for each microplane
        e_equiv_arr = sqrt(e_N_pos_arr * e_N_pos_arr + c_T * e_TT_arr)
        # print 'e_equiv_arr', e_equiv_arr
        return e_equiv_arr

    def _get_e_max(self, e_equiv_arr, e_max_arr):
        '''
        Compares the equivalent microplane strain of a single microplane with the
        maximum strain reached in the loading history for the entire array
        '''
        bool_e_max = e_equiv_arr >= e_max_arr

        # The new value must be created, otherwise side-effect could occur
        # by writing into a state array.
        #
        new_e_max_arr = np.copy(e_max_arr)
        new_e_max_arr[bool_e_max] = e_equiv_arr[bool_e_max]
        return new_e_max_arr

    def _get_state_variables(self, sctx, eps_ab):
        '''
        Compares the list of current equivalent microplane strains with
        the values in the state array and returns the higher values
        '''
        e_vct_arr = self._get_e_vct_arr(eps_ab)
        e_equiv_arr = self._get_e_equiv_arr(e_vct_arr)
        print
        #e_max_arr_old = sctx.mats_state_array
        #e_max_arr_new = self._get_e_max(e_equiv_arr, e_max_arr_old)
        return e_equiv_arr

    def _get_phi_arr(self, sctx, eps_ab):
        # Returns a list of the integrity factors for all microplanes.
        e_max_arr = self._get_state_variables(sctx, eps_ab)

        phi_arr = self._get_phi(e_max_arr, sctx)[:]

        return phi_arr

    def _get_phi_mtx(self, sctx, eps_ab):
        # Returns the 2nd order damage tensor 'phi_mtx'

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_ab)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        return phi_ij

    def _get_beta_tns(self, phi_mtx):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        delta = identity(2)

        beta_ijkl = 0.25 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                            einsum('il,jk->ijkl', phi_mtx, delta) +
                            einsum('jk,il->ijkl', phi_mtx, delta) +
                            einsum('jl,ik->ijkl', phi_mtx, delta))

        return beta_ijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_ab):

        # -----------------------------------------------------------------------------------------------
        # update state variables
        # -----------------------------------------------------------------------------------------------
        # if sctx.update_state_on:
        #    #eps_n = eps_avg - d_eps
        #   e_max = self._get_state_variables(sctx, eps_app_eng)
        #    sctx.mats_state_array[:] = e_max

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        # if self.regularization:
        #    h = self.get_regularizing_length(sctx, eps_app_eng)
        #    self.phi_fn.h = h

        #------------------------------------------------------------------
        # Damage tensor (2th order):
        #------------------------------------------------------------------
        phi_ij = self._get_phi_mtx(sctx, eps_ab)

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_ijkl = self._get_beta_tns(phi_ij)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D4_mdm_ijab = einsum(
            'ijkl,klsr,absr->ijab', beta_ijkl, self.D4_e, beta_ijkl)

        sig_ij = einsum('ijab,ab -> ij', D4_mdm_ijab, eps_ab)

        return sig_ij, D4_mdm_ijab


class MATS2DMicroplaneDamageJir(MATSXDMicroplaneDamageFatigueJir, MATS3DEval):

    implements(IMATSEval)

    #-----------------------------------------------
    # number of microplanes - currently fixed for 3D
    #-----------------------------------------------
    n_mp = Constant(28)

    _alpha_list = Property(depends_on='n_mp')

    @cached_property
    def _get__alpha_list(self):
        return array([np.pi / self.n_mp * (i - 0.5)
                      for i in range(1, self.n_mp + 1)])

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        return array([[np.cos(alpha), np.sin(alpha)] for alpha in self._alpha_list])

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        return np.ones(self.n_mp) / self.n_mp * 2.0

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------
    stress_state = tr.Enum("plane_stress", "plane_strain", input=True)

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------
    E = tr.Float(34e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    def _get_lame_params(self):
        la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2 + 2 * self.nu)
        return la, mu

    D_ab = tr.Property(tr.Array, depends_on='+input')
    '''Elasticity matrix (shape: (3,3))
    '''
    @tr.cached_property
    def _get_D_ab(self):
        if self.stress_state == 'plane_stress':
            return self._get_D_ab_plane_stress()
        elif self.stress_state == 'plane_strain':
            return self._get_D_ab_plane_strain()

    def _get_D_ab_plane_stress(self):
        '''
        Elastic Matrix - Plane Stress
        '''
        E = self.E
        nu = self.nu
        D_stress = np.zeros([3, 3])
        D_stress[0, 0] = E / (1.0 - nu * nu)
        D_stress[0, 1] = E / (1.0 - nu * nu) * nu
        D_stress[1, 0] = E / (1.0 - nu * nu) * nu
        D_stress[1, 1] = E / (1.0 - nu * nu)
        D_stress[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_ab_plane_strain(self):
        '''
        Elastic Matrix - Plane Strain
        '''
        E = self.E
        nu = self.nu
        D_strain = np.zeros([3, 3])
        D_strain[0, 0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0, 1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2, 2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    map2d_ijkl2a = tr.Array(np.int_, value=[[[[0, 0],
                                              [0, 0]],
                                             [[2, 2],
                                              [2, 2]]],
                                            [[[2, 2],
                                              [2, 2]],
                                             [[1, 1],
                                                [1, 1]]]])
    map2d_ijkl2b = tr.Array(np.int_, value=[[[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                              [2, 1]]],
                                            [[[0, 2],
                                              [2, 1]],
                                             [[0, 2],
                                                [2, 1]]]])

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        return self.D_ab[self.map2d_ijkl2a, self.map2d_ijkl2b]


if __name__ == '__main__':

    #=========================
    # model behavior
    #=========================
    n = 200
    s_levels = linspace(0, 0.002, 10)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= 0
    s_history = s_levels.flatten()
    s_arr = hstack([linspace(s_history[i], s_history[i + 1], n)
                    for i in range(len(s_levels) - 1)])

    eps = array([array([[s_arr[i], 0],
                        [0, 0]]) for i in range(0, len(s_arr))])

    m2 = MATS2DMicroplaneDamageJir()
    sigma = zeros_like(eps)
    #sigma_kk = zeros(len(s_arr) + 1)
    w = zeros((len(eps[:, 0, 0]), 28))
    e_pi = zeros((len(eps[:, 0, 0]), 28))
    e = zeros((len(eps[:, 0, 0]), 28, 2))
    e_T = zeros((len(eps[:, 0, 0]), 28, 2))
    e_N = zeros((len(eps[:, 0, 0]), 28))
    sctx = zeros((len(eps[:, 0, 0]) + 1, 28))

    for i in range(0, len(eps[:, 0, 0])):

        sigma[i, :] = m2.get_corr_pred(sctx[i, :], eps[i, :])[0]
        sctx[i + 1] = m2._get_state_variables(sctx[i, :], eps[i, :])
        w[i, :] = 1 - m2._get_phi_arr(sctx[i], eps[i, :])
        e[i, :] = m2._get_e_vct_arr(eps[i, :])

    plt.subplot(221)
    plt.plot(eps[:, 0, 0], sigma[:, 0, 0], linewidth=1, label='sigma_11')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    plt.legend()

    plt.subplot(222)
    plt.plot(eps[:, 0, 0], w[:], linewidth=1, label='sigma_11')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    # plt.legend()

#     plt.subplot(222)
#     for i in range(0, 28):
#         plt.plot(
#             eps[:, 0, 0], w[:, i], linewidth=1, label='Damage of the microplanes', alpha=1)
#
#         plt.xlabel('Strain')
#         plt.ylabel('Damage of the microplanes')
#         # plt.legend()
#
#     plt.subplot(223)
#     for i in range(0, 28):
#         plt.plot(
#             eps[:, 0, 0], sctx[0:n, i], linewidth=1, label='Damage of the microplanes', alpha=1)
#
#         plt.xlabel('Strain')
#         plt.ylabel('Damage of the microplanes')
    # plt.legend()

#     plt.subplot(223)
#     for i in range(0, 28):
#         plt.plot(
#             eps[:, 0, 0], e_T[:, i, 0], linewidth=1, label='Tangential_strain')
#
#         plt.xlabel('Strain')
#         plt.ylabel('Tangential_strain')
#         # plt.legend()

    plt.show()

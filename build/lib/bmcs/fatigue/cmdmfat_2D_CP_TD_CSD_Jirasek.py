'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, zeros, trace, \
    einsum, zeros_like,\
    identity, sign, linspace, hstack, maximum,\
    sqrt
from scipy.linalg import \
    eigh
from traits.api import \
    Constant, provides,\
    Float, HasTraits, \
    Property, cached_property
from traitsui.api import \
    View,  Include
import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr

@provides(IMATSEval)
class MATSEvalMicroplaneFatigue(HasTraits):
    #--------------------------
    # material model parameters
    #--------------------------

    E = Float(34000.,
              label="E",
              desc="Young modulus",
              enter_set=True,
              auto_set=False)

    nu = Float(0.2,
               label="nu",
               desc="poission ratio",
               enter_set=True,
               auto_set=False)

    #---------------------------------------
    # Tangential constitutive law parameters
    #---------------------------------------
    gamma_T = Float(5000.,
                    label="Gamma",
                    desc=" Tangential Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)

    K_T = Float(10.0,
                label="K",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S = Float(0.00001,
              label="S",
              desc="Damage strength",
              enter_set=True,
              auto_set=False)

    r = Float(1.2,
              label="r",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1.0,
              label="c",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(2.0,
                       label="Tau_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(0.0,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    #-------------------------------------------
    # Normal_Tension constitutive law parameters
    #-------------------------------------------
    Ad = Float(10000.0,
               label="a",
               desc="brittleness coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = Float(0.5e-4,
                  label="a",
                  desc="threshold strain",
                  enter_set=True,
                  auto_set=False)

    #-----------------------------------------------
    # Normal_Compression constitutive law parameters
    #-----------------------------------------------
    K_N = Float(10000.,
                label="K_N",
                desc=" Normal isotropic harening",
                enter_set=True,
                auto_set=False)

    gamma_N = Float(15000.,
                    label="gamma_N",
                    desc="Normal kinematic hardening",
                    enter_set=True,
                    auto_set=False)

    sigma_0 = Float(20.,
                    label="sigma_0",
                    desc="Yielding stress",
                    enter_set=True,
                    auto_set=False)

    #--------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    #--------------------------------------------------------------
    def get_normal_law(self, eps, sctx):

        E_N = self.E / (1.0 - 2.0 * self.nu)

        w_N = sctx[:, 0]
        z_N = sctx[:, 1]
        alpha_N = sctx[:, 2]
        r_N = sctx[:, 3]
        eps_N_p = sctx[:, 4]

        pos = eps > 1e-6
        H = 1.0 * pos

        sigma_n_trial = (1 - H * w_N) * E_N * (eps - eps_N_p)
        Z = self.K_N * r_N
        X = self.gamma_N * alpha_N

        h = self.sigma_0 + Z
        pos_iso = h > 1e-6
        f_trial = abs(sigma_n_trial - X) - h * pos_iso

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p = eps_N_p + delta_lamda * sign(sigma_n_trial - X)
        r_N = r_N + delta_lamda
        alpha_N = alpha_N + delta_lamda * sign(sigma_n_trial - X)

        def Z_N(z_N): return 1. / self.Ad * (-z_N) / (1 + z_N)
        Y_N = 0.5 * H * E_N * eps ** 2
        Y_0 = 0.5 * E_N * self.eps_0 ** 2
        f = Y_N - (Y_0 + Z_N(z_N))

        thres_2 = f > 1e-6

        def f_w(Y): return 1 - 1. / (1 + self.Ad * (Y - Y_0))
        w_N = f_w(Y_N) * thres_2
        z_N = - w_N * thres_2

        n_mp = self.n_mp
        new_sctx = zeros((n_mp, 5))

        new_sctx[:, 0] = w_N
        new_sctx[:, 1] = z_N
        new_sctx[:, 2] = alpha_N
        new_sctx[:, 3] = r_N
        new_sctx[:, 4] = eps_N_p
        return new_sctx

    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, e_T, sctx, sigma_kk):

        E_T = self.E / (1. + self.nu)

        w_T = sctx[:, 5]
        z_T = sctx[:, 6]
        alpha_T = sctx[:, 7:9]
        eps_T_pi = sctx[:, 9:11]

        sig_pi_trial = E_T * (e_T - eps_T_pi)
        Z = self.K_T * z_T
        X = self.gamma_T * alpha_T
        norm_1 = sqrt(
            einsum('nj,nj -> n', (sig_pi_trial - X), (sig_pi_trial - X)))

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_kk / 3.0

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - w_T) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + sqrt(
            einsum('nj,nj -> n', (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi[:, 0] = eps_T_pi[:, 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[:, 0] - X[:, 0]) / (1.0 - w_T)) / norm_2
        eps_T_pi[:, 1] = eps_T_pi[:, 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[:, 1] - X[:, 1]) / (1.0 - w_T)) / norm_2
#         eps_T_pi[:, 2] = eps_T_pi[:, 2] +  plas_1 * delta_lamda * \
#             ((sig_pi_trial[:, 2] - X[:, 2]) / (1.0 - w_T)) / norm_2

        Y = 0.5 * E_T * \
            einsum('nj,nj -> n', (e_T - eps_T_pi), (e_T - eps_T_pi))

        w_T += ((1 - w_T) ** self.c) * \
            (delta_lamda * (Y / self.S) ** self.r) * \
            (self.tau_pi_bar / (self.tau_pi_bar - self.a * sigma_kk / 3.0))

        alpha_T[:, 0] = alpha_T[:, 0] + plas_1 * delta_lamda *\
            (sig_pi_trial[:, 0] - X[:, 0]) / norm_2
        alpha_T[:, 1] = alpha_T[:, 1] + plas_1 * delta_lamda *\
            (sig_pi_trial[:, 1] - X[:, 1]) / norm_2
#         alpha_T[:, 2] = alpha_T[:, 2]   + plas_1 * delta_lamda *\
#             (sig_pi_trial[:, 2] - X[:, 2]) / norm_2

        z_T = z_T + delta_lamda
        n_mp = self.n_mp
        new_sctx = zeros((n_mp, 6))
        new_sctx[:, 0] = w_T
        new_sctx[:, 1] = z_T
        new_sctx[:, 2:4] = alpha_T
        new_sctx[:, 4:6] = eps_T_pi
        return new_sctx


@provides(IMATSEval)
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

    def _get_e_vct_arr(self, eps_eng):
        # Projection of apparent strain onto the individual microplanes
        e_ni = einsum('nj,ji->ni', self._MPN, eps_eng)
        return e_ni

    def _get_e_N_arr(self, e_vct_arr):
        # get the normal strain array for each microplane
        eN_n = einsum('ni,ni->n', e_vct_arr, self._MPN)
        return eN_n

    def _get_e_T_vct_arr(self, e_vct_arr):
        # get the tangential strain vector array for each microplane
        eN_n = self._get_e_N_arr(e_vct_arr)
        eN_vct_ni = einsum('n,ni->ni', eN_n, self._MPN)
        return e_vct_arr - eN_vct_ni

    #-------------------------------------------------
    # Alternative methods for the kinematic constraint
    #-------------------------------------------------
    def _get_e_N_arr_2(self, eps_eng):
        return einsum('nij,ij->n', self._MPNN, eps_eng)

    def _get_e_T_vct_arr_2(self, eps_eng):
        MPTT_ijr = self._get__MPTT()
        return einsum('nijr,ij->nr', MPTT_ijr, eps_eng)

    def _get_e_vct_arr_2(self, eps_eng):
        return self._e_N_arr_2 * self._MPN + self._e_t_vct_arr_2

    #--------------------------------------------------------
    # return the state variables (Damage , inelastic strains)
    #--------------------------------------------------------
    def _get_state_variables(self, sctx, eps_app_eng, sigma_kk):

        e_N_arr = self._get_e_N_arr_2(eps_app_eng)
        e_T_vct_arr = self._get_e_T_vct_arr_2(eps_app_eng)

        sctx_arr = zeros_like(sctx)

        sctx_N = self.get_normal_law(e_N_arr, sctx)
        sctx_arr[:, 0:5] = sctx_N

        sctx_tangential = self.get_tangential_law(e_T_vct_arr, sctx, sigma_kk)
        sctx_arr[:, 5:11] = sctx_tangential

        return sctx_arr

    #-----------------------------------------------------------------
    # Returns a list of the plastic normal strain  for all microplanes.
    #-----------------------------------------------------------------
    def _get_eps_N_p_arr(self, sctx, eps_app_eng, sigma_kk):

        eps_N_p = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 4]
        return eps_N_p

    #----------------------------------------------------------------
    # Returns a list of the sliding strain vector for all microplanes.
    #----------------------------------------------------------------
    def _get_eps_T_pi_arr(self, sctx, eps_app_eng, sigma_kk):

        eps_T_pi_vct_arr = self._get_state_variables(
            sctx, eps_app_eng, sigma_kk)[:, 9:11]

        return eps_T_pi_vct_arr

    #-------------------------------------------------------------
    # Returns a list of the integrity factors for all microplanes.
    #-------------------------------------------------------------

    def _get_phi_arr(self, sctx, eps_app_eng, sigma_kk):

        w_n = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 0]
        w_T = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]

        w = zeros(self.n_mp)

        for i in range(0, self.n_mp):
            w[i] = maximum(w_n[i], w_T[i])

        phi_arr = sqrt(1. - w)

        return phi_arr

    #----------------------------------------------
    # Returns the 2nd order damage tensor 'phi_mtx'
    #----------------------------------------------
    def _get_phi_mtx(self, sctx, eps_app_eng, sigma_kk):

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng, sigma_kk)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        return phi_ij

    #----------------------------------------------------------------------
    # Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
    #(cf. [Jir99], Eq.(21))
    #----------------------------------------------------------------------
    def _get_beta_tns_sum_type(self, sctx, eps_app_eng, sigma_kk):

        delta = identity(2)

        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        beta_ijkl = 0.25 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                            einsum('il,jk->ijkl', phi_mtx, delta) +
                            einsum('jk,il->ijkl', phi_mtx, delta) +
                            einsum('jl,ik->ijkl', phi_mtx, delta))

        return beta_ijkl

    #----------------------------------------------------------------------
    # Returns the 4th order damage tensor 'beta4' using product-type symmetrization
    #(cf. [Baz97], Eq.(87))
    #----------------------------------------------------------------------
    def _get_beta_tns_product_type(self, sctx, eps_app_eng, sigma_kk):

        delta = identity(2)

        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)

        n_dim = 2
        phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
        phi_eig_value_real = array([pe.real for pe in phi_eig_value])
        phi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
        for i in range(n_dim):
            phi_pdc_mtx[i, i] = phi_eig_value_real[i]
        # w_mtx = tensorial square root of the second order damage tensor:
        w_pdc_mtx = sqrt(phi_pdc_mtx)

        # transform the matrix w back to x-y-coordinates:
        w_mtx = einsum('ik,kl,lj -> ij', phi_eig_mtx, w_pdc_mtx, phi_eig_mtx)
        #w_mtx = dot(dot(phi_eig_mtx, w_pdc_mtx), transpose(phi_eig_mtx))

        beta_ijkl = 0.5 * \
            (einsum('ik,jl -> ijkl', w_mtx, w_mtx) +
             einsum('il,jk -> ijkl', w_mtx, w_mtx))

        return beta_ijkl

    #-----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    #-----------------------------------------------------------
    def _get_eps_p_mtx(self, sctx, eps_app_eng, sigma_kk):

        # plastic normal strains
        eps_N_P_n = self._get_eps_N_p_arr(sctx, eps_app_eng, sigma_kk)

        # sliding tangential strains
        eps_T_pi_ni = self._get_eps_T_pi_arr(sctx, eps_app_eng, sigma_kk)
        delta = identity(2)

        # 2-nd order plastic (inelastic) tensor
        eps_p_ij = einsum('n,n,ni,nj -> ij', self._MPW, eps_N_P_n, self._MPN, self._MPN) + \
            0.5 * (einsum('n,nr,ni,rj->ij', self._MPW, eps_T_pi_ni, self._MPN, delta) +
                   einsum('n,nr,nj,ri->ij', self._MPW, eps_T_pi_ni, self._MPN, delta))

        return eps_p_ij

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, sigma_kk):

        # Corrector predictor computation.

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_ijkl = self._get_beta_tns_sum_type(
            sctx, eps_app_eng, sigma_kk)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D4_mdm_ijmn = einsum(
            'ijkl,klsr,mnsr->ijmn', beta_ijkl, self.D4_e, beta_ijkl)

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        # plastic strain tensor
        eps_p_ij = self._get_eps_p_mtx(sctx, eps_app_eng, sigma_kk)

        # elastic strain tensor
        eps_e_mtx = eps_app_eng - eps_p_ij

        # calculation of the stress tensor
        sig_eng = einsum('ijmn,mn -> ij', D4_mdm_ijmn, eps_e_mtx)

        return sig_eng, D4_mdm_ijmn


@provides(IMATSEval)
class MATS2DMicroplaneDamageJir(MATSXDMicroplaneDamageFatigueJir, MATS3DEval):

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
    n = 100
    s_levels = linspace(0, -0.02, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    s_history_1 = s_levels.flatten()

    # cyclic loading

    s_history_2 = [-0, -0.001, -0.00032, -
                   0.0015, -0.00065, -0.0020, -0.00095,
                   -0.0027, -0.0014, -0.004, -0.0022, -0.0055,
                   -0.0031, -0.007, -0.004, -0.008, -0.0045, -.009,
                   -0.0052, -0.01, -0.0058, -0.012, -0.0068, -0.015]

    s_history_2 = [0, 0.01]

    s_arr_1 = hstack([linspace(s_history_1[i], s_history_1[i + 1], n)
                      for i in range(len(s_levels) - 1)])

    s_arr_2 = hstack([linspace(s_history_2[i], s_history_2[i + 1], 200)
                      for i in range(len(s_history_2) - 1)])

    p = 1.0  # ratio of strain eps_11 (for bi-axial loading)
    m = 0.0  # ratio of strain eps_22 (for bi-axial loading)

    eps_1 = array([array([[p * s_arr_1[i], 0],
                          [0, m * s_arr_1[i]]]) for i in range(0, len(s_arr_1))])

    eps_2 = array([array([[p * s_arr_2[i], 0],
                          [0, m * s_arr_2[i]]]) for i in range(0, len(s_arr_2))])

    m2 = MATS2DMicroplaneDamageJir()
    n_mp = m2.n_mp
    sigma_1 = zeros_like(eps_1)
    sigma_kk_1 = zeros(len(s_arr_1) + 1)
    w_1_N = zeros((len(eps_1[:, 0, 0]), n_mp))
    w_1_T = zeros((len(eps_1[:, 0, 0]), n_mp))
    eps_P_N_1 = zeros((len(eps_1[:, 0, 0]), n_mp))
    eps_Pi_T_1 = zeros((len(eps_1[:, 0, 0]), n_mp, 2))
    e_1 = zeros((len(eps_1[:, 0, 0]), n_mp, 2))
    e_T_1 = zeros((len(eps_1[:, 0, 0]), n_mp, 2))
    e_N_1 = zeros((len(eps_1[:, 0, 0]), n_mp))
    sctx_1 = zeros((len(eps_1[:, 0, 0]) + 1, n_mp, 11))

    sigma_2 = zeros_like(eps_2)
    sigma_kk_2 = zeros(len(s_arr_2) + 1)
    w_2_N = zeros((len(eps_2[:, 0, 0]), n_mp))
    w_2_T = zeros((len(eps_2[:, 0, 0]), n_mp))
    eps_P_N_2 = zeros((len(eps_2[:, 0, 0]), n_mp))
    eps_Pi_T_2 = zeros((len(eps_2[:, 0, 0]), n_mp, 2))
    e_2 = zeros((len(eps_2[:, 0, 0]), n_mp, 2))
    e_T_2 = zeros((len(eps_2[:, 0, 0]), n_mp, 2))
    e_N_2 = zeros((len(eps_2[:, 0, 0]), n_mp))
    sctx_2 = zeros((len(eps_2[:, 0, 0]) + 1, n_mp, 11))

    for i in range(0, len(eps_1[:, 0, 0])):

        sigma_1[i, :] = m2.get_corr_pred(
            sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])[0]
        sigma_kk_1[i + 1] = trace(sigma_1[i, :])
        sctx_1[
            i + 1] = m2._get_state_variables(sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])
        w_1_N[i, :] = sctx_1[i, :, 0]
        w_1_T[i, :] = sctx_1[i, :, 5]
        eps_P_N_1[i, :] = sctx_1[i, :, 4]
        eps_Pi_T_1[i, :, :] = sctx_1[i, :, 9:11]
        #e_1[i, :] = m2._get_e_vct_arr(eps_1[i, :])
        #e_T_1[i, :] = m2._get_e_T_vct_arr_2(eps_1[i, :])
        #e_N_1[i, :] = m2._get_e_N_arr(e_1[i, :])

    for i in range(0, len(eps_2[:, 0, 0])):

        sigma_2[i, :] = m2.get_corr_pred(
            sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])[0]
        sigma_kk_2[i + 1] = trace(sigma_2[i, :])
        sctx_2[
            i + 1] = m2._get_state_variables(sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])
        w_2_N[i, :] = sctx_2[i, :, 0]
        w_2_T[i, :] = sctx_2[i, :, 5]
        eps_P_N_2[i, :] = sctx_2[i, :, 4]
        eps_Pi_T_2[i, :, :] = sctx_2[i, :, 9:11]
        #e_2[i, :] = m2._get_e_vct_arr(eps_2[i, :])
        #e_T_2[i, :] = m2._get_e_T_vct_arr_2(eps_2[i, :])
        #e_N_2[i, :] = m2._get_e_N_arr(e_2[i, :])

    # stress -strain
    plt.subplot(221)
    plt.plot(eps_1[:, 0, 0], sigma_1[:, 0, 0],
             linewidth=1, label='sigma_11_(monotonic)')
    plt.plot(eps_1[:, 0, 0], sigma_1[:, 1, 1], linewidth=1, label='sigma_22')
    #plt.plot(eps[:, 0, 0], sigma[:, 0, 1], linewidth=1, label='sigma_12')
    plt.plot(eps_2[:, 0, 0], sigma_2[:, 0, 0],
             linewidth=1, label='sigma_11_(cyclic)')
    plt.title('$\sigma - \epsilon$')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    plt.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='k', linewidth=1, alpha=0.5)
    plt.legend()

    # normal damage at the microplanes (TD)
    plt.subplot(222)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], w_1_N[:, i], linewidth=1.0, label='cyclic', alpha=1)
        plt.plot(
            eps_2[:, 0, 0], w_2_N[:, i], linewidth=1.0, label='monotonic', alpha=1)

        plt.xlabel('Strain')
        plt.ylabel('Damage')
        plt.title(' normal damage for all microplanes')

    # tangential damage at the microplanes (CSD)
    plt.subplot(223)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], w_1_T[:, i], linewidth=1.0, label='cyclic', alpha=1)
        plt.plot(
            eps_2[:, 0, 0], w_2_T[:, i], linewidth=1.0, label='monotonic', alpha=1)

        plt.xlabel('Strain')
        plt.ylabel('Damage')
        plt.title(' tangential damage for all microplanes')

    # tangential sliding strains at the microplanes (CSD)
    plt.subplot(224)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], eps_P_N_1[:, i], linewidth=1, label='plastic strain')

        plt.plot(
            eps_2[:, 0, 0], eps_P_N_2[:, i], linewidth=1, label='plastic strain')

        plt.xlabel('Strain')
        plt.ylabel('sliding_strain')

    plt.show()

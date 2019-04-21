'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from numpy import array,\
    einsum, zeros_like, identity, sign,\
    sqrt
from traits.api import Constant,\
    Float, Property, cached_property

import numpy as np
import traits.api as tr


class MATS3DMplCSDODF(MATS3DEval):
    '''Tangential constitutive law parameters
    '''
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

    eps_0 = Float(56e-6,
                  label="a",
                  desc="threshold strain",
                  enter_set=True,
                  auto_set=False)

    eps_f = Float(250e-6,
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

    zeta_G = Float(1.0,
                   label="zeta_G",
                   desc="anisotropy parameter",
                   enter_set=True,
                   auto_set=False)

    state_var_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''
    @cached_property
    def _get_state_shapes(self):
        return {'w_N_Emn': (self.n_mp,),
                'z_N_Emn': (self.n_mp,),
                'alpha_N_Emn': (self.n_mp,),
                'r_N_Emn': (self.n_mp,),
                'eps_N_p_Emn': (self.n_mp,),
                'sigma_N_Emn': (self.n_mp,),
                'w_T_Emn': (self.n_mp,),
                'z_T_Emn': (self.n_mp,),
                'alpha_T_Emna': (self.n_mp, 3),
                'eps_T_pi_Emna': (self.n_mp, 3), }

    #--------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    #--------------------------------------------------------------
    def get_normal_law(self, eps_N_Emn, w_N_Emn, z_N_Emn,
                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn):

        E_N = self.E / (1.0 - 2.0 * self.nu)

        pos = eps_N_Emn > 1e-6
        H = 1.0 * pos

        sigma_n_trial = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        Z = self.K_N * r_N_Emn
        X = self.gamma_N * alpha_N_Emn

        h = self.sigma_0 + Z
        pos_iso = h > 1e-6
        f_trial = abs(sigma_n_trial - X) - h * pos_iso

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn = eps_N_p_Emn + delta_lamda * sign(sigma_n_trial - X)
        r_N_Emn = r_N_Emn + delta_lamda
        alpha_N_Emn = alpha_N_Emn + delta_lamda * sign(sigma_n_trial - X)

#         def Z_N(z_N_Emn): return 1.0 / self.Ad * (-z_N_Emn) / (1.0 + z_N_Emn)
#         Y_N = 0.5 * H * E_N * eps_N_Emn ** 2.0
#         Y_0 = 0.5 * E_N * self.eps_0 ** 2.0
#         f = Y_N - (Y_0 + Z_N(z_N_Emn))
#
#         thres_2 = f > 1e-6
#
#         def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))
#         w_N_Emn = f_w(Y_N) * thres_2
#         z_N_Emn = - w_N_Emn * thres_2

        f_trial_Emn = eps_N_Emn - self.eps_0
        f_idx = np.where(f_trial_Emn > 0)
        z_N_Emn[f_idx] = eps_N_Emn[f_idx]
        w_N_Emn[f_idx] = (1. -
                          (self.eps_0 / z_N_Emn[f_idx]) * np.exp(- (z_N_Emn[f_idx] - self.eps_0) / (self.eps_f - self.eps_0)))

        sigma_N_Emn = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)

        return w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn

    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, w_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_kk):

        E_T = self.E / (1.0 + self.nu)

        sig_pi_trial = E_T * (eps_T_Emna - eps_T_pi_Emna)
        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        norm_1 = sqrt(
            einsum('Emna,Emna -> Emn', (sig_pi_trial - X), (sig_pi_trial - X)))

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_kk

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - w_T_Emn) + self.gamma_T + self.K_T) * plas_1

#         print 'f', f
#         print 'w_T_Emn', w_T_Emn
#         print 'delta_lamda', delta_lamda
#         print '*****'

        norm_2 = 1.0 * elas_1 + sqrt(
            einsum('Emna,Emna -> Emn', (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] = eps_T_pi_Emna[..., 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 0] - X[..., 0]) /
             (1.0 - w_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] = eps_T_pi_Emna[..., 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 1] - X[..., 1]) /
             (1.0 - w_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 2] = eps_T_pi_Emna[..., 2] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 2] - X[..., 2]) /
             (1.0 - w_T_Emn)) / norm_2

        Y = 0.5 * E_T * \
            einsum('Emna,Emna -> Emn', (eps_T_Emna - eps_T_pi_Emna),
                   (eps_T_Emna - eps_T_pi_Emna))

        w_T_Emn += ((1.0 - w_T_Emn) ** self.c) * \
            (delta_lamda * (Y / self.S) ** self.r)  # * \
        #(self.tau_pi_bar / (self.tau_pi_bar - self.a * sigma_kk / 3.0))

        alpha_T_Emna[..., 0] = alpha_T_Emna[..., 0] + plas_1 * delta_lamda *\
            (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] = alpha_T_Emna[..., 1] + plas_1 * delta_lamda *\
            (sig_pi_trial[..., 1] - X[..., 1]) / norm_2
        alpha_T_Emna[..., 2] = alpha_T_Emna[..., 2] + plas_1 * delta_lamda *\
            (sig_pi_trial[..., 2] - X[..., 2]) / norm_2
        z_T_Emn = z_T_Emn + delta_lamda

        return w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna

    #-------------------------------------------------------------------------
    # MICROPLANE-Kinematic constraints
    #-------------------------------------------------------------------------
    def _get_e_Emna(self, eps_Emab):
        # Projection of apparent strain onto the individual microplanes
        e_ni = einsum('nb,Emba->Emna', self._MPN, eps_Emab)
        return e_ni

    def _get_e_N_Emn(self, e_Emna):
        # get the normal strain array for each microplane
        e_N_Emn = einsum('Emna, na->Emn', e_Emna, self._MPN)
        return e_N_Emn

    def _get_e_T_Emna(self, e_Emna):
        # get the tangential strain vector array for each microplane
        e_N_Emn = self._get_e_N_Emn(e_Emna)
        e_N_Emna = einsum('Emn,na->Emna', e_N_Emn, self._MPN)
        return e_Emna - e_N_Emna

    #-------------------------------------------------
    # Alternative methods for the kinematic constraint
    #-------------------------------------------------
    # get the dyadic product of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')

    @cached_property
    def _get__MPTT(self):
        delta = identity(3)
        MPTT_nijr = 0.5 * (einsum('ni,jr -> nijr', self._MPN, delta) +
                           einsum('nj,ir -> njir', self._MPN, delta) - 2 *
                           einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN))
        return MPTT_nijr

    def _get_e_N_Emn_2(self, eps_Emab):
        # Projection of apparent strain onto the individual microplanes
        return einsum('nij,Emij->Emn', self._MPNN, eps_Emab)

    def _get_e_T_Emna_2(self, eps_Emab):
        # get the normal strain array for each microplane
        MPTT_ijr = self._get__MPTT()
        return einsum('nija,Emij->Emna', MPTT_ijr, eps_Emab)

    def _get_e_Emna_2(self, eps_Emab):
        # get the tangential strain vector array for each microplane
        return self._get_e_N_Emn_2(eps_Emab) * self._MPN +\
            self._get_e_T_Emna_2(eps_Emab)

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
        sctx_arr[:, 5:13] = sctx_tangential

        return sctx_arr

    #-----------------------------------------------------------------
    # Returns a list of the plastic normal strain  for all microplanes.
    #-----------------------------------------------------------------
    def _get_eps_N_p_Emn(self, eps_Emab, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn):
        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_N_p_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[4]
        return eps_N_p_Emn

    #----------------------------------------------------------------
    # Returns a list of the sliding strain vector for all microplanes.
    #----------------------------------------------------------------
    def _get_eps_T_pi_Emna(self, eps_Emab, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_kk_Em):

        eps_T_Emna = self._get_e_T_Emna_2(eps_Emab)
        eps_N_T_pi_Emna = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_kk_Em)[3]

        return eps_N_T_pi_Emna

    #-------------------------------------------------------------
    # Returns a list of the integrity factors for all microplanes.
    #-------------------------------------------------------------
    def _get_phi_Emn(self, eps_Emab, w_N_Emn, z_N_Emn,
                     alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                     w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna,  sigma_N_Emn):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emna_2(eps_Emab)

        w_N_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[0]
        sigma_N_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[5]
        w_T_Emn = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)[0]

        w_Emn = zeros_like(w_N_Emn)

#         w_Emn = np.zeros_like(w_N_Emn)
#         idx_1 = np.where(sigma_N_Emn >= 0)
#         w_Emn[idx_1] = w_N_Emn[idx_1]
#
#         idx_2 = np.where(sigma_N_Emn < 0)
#         w_Emn[idx_2] = w_T_Emn[idx_2]
#
        eig = np.linalg.eig(eps_Emab)[0]

        ter_1 = np.sum(eig)
        idx_1 = np.where(ter_1 > 0.0)
        idx_2 = np.where(ter_1 <= 0.0)

        w_Emn[idx_1] = w_N_Emn[idx_1]
        w_Emn[idx_2] = w_T_Emn[idx_2]

        phi_Emn = 1.0 - w_Emn

        return phi_Emn

    #----------------------------------------------
    # Returns the 2nd order damage tensor 'phi_mtx'
    #----------------------------------------------
    def _get_phi_Emab(self, phi_Emn):

        # integration terms for each microplanes
        phi_Emab = einsum('Emn,n,nab->Emab', phi_Emn, self._MPW, self._MPNN)

        return phi_Emab
#
#     #----------------------------------------------------------------------
#     # Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
#     #(cf. [Jir99], Eq.(21))
#     #----------------------------------------------------------------------
#     def _get_beta_Emabcd(self, phi_Emab):
#
#         delta = identity(3)
#
#         # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
#         beta_Emabcd = 0.25 * (einsum('Emik,jl->Emijkl', phi_Emab, delta) +
#                               einsum('Emil,jk->Emijkl', phi_Emab, delta) +
#                               einsum('Emjk,il->Emijkl', phi_Emab, delta) +
#                               einsum('Emjl,ik->Emijkl', phi_Emab, delta))
#
#         return beta_Emabcd
#
# #     #----------------------------------------------------------------------
# #     # Returns the 4th order damage tensor 'beta4' using product-type symmetrization
# #     #(cf. [Baz97], Eq.(87))
# #     #----------------------------------------------------------------------
# #     def _get_beta_tns_product_type(self, sctx, eps_app_eng, sigma_kk):
# #
# #         delta = identity(2)
# #
# #         phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
# #
# #         n_dim = 2
# #         phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
# #         phi_eig_value_real = array([pe.real for pe in phi_eig_value])
# #         phi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
# #         for i in range(n_dim):
# #             phi_pdc_mtx[i, i] = phi_eig_value_real[i]
# #         # w_mtx = tensorial square root of the second order damage tensor:
# #         w_pdc_mtx = sqrt(phi_pdc_mtx)
# #
# #         # transform the matrix w back to x-y-coordinates:
# #         w_mtx = einsum('ik,kl,lj -> ij', phi_eig_mtx, w_pdc_mtx, phi_eig_mtx)
# #         #w_mtx = dot(dot(phi_eig_mtx, w_pdc_mtx), transpose(phi_eig_mtx))
# #
# #         beta_ijkl = 0.5 * \
# #             (einsum('ik,jl -> ijkl', w_mtx, w_mtx) +
# #              einsum('il,jk -> ijkl', w_mtx, w_mtx))
# #
# #         return beta_ijkl
#
#     #-----------------------------------------------------------
#     # Integration of the (inelastic) strains for each microplane
#     #-----------------------------------------------------------
#     def _get_eps_p_Emab(self, eps_Emab, w_N_Emn, z_N_Emn,
#                         alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
#                         w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna,  sigma_N_Emn):
#
#         eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
#         eps_T_Emna = self._get_e_T_Emna_2(eps_Emab)
#
# #         # plastic normal strains
# #         eps_N_p_Emn = self.get_normal_law(
# # eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[5]
#
#         eps_N_p_Emn = self._get_eps_N_p_Emn(
#             eps_Emab, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)
#
# #         # sliding tangential strains
# #         eps_T_pi_Emna = self.get_tangential_law(
# # eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna,
# # sigma_N_Emn)[4]
#
#         eps_T_pi_Emna = self._get_eps_T_pi_Emna(
#             eps_Emab, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)
#
#         delta = identity(3)
#
#         # 2-nd order plastic (inelastic) tensor
#         eps_p_Emab = einsum('n,Emn,na,nb -> Emab', self._MPW, eps_N_p_Emn, self._MPN, self._MPN) + \
#             0.5 * (einsum('n,Emnf,na,fb-> Emab', self._MPW, eps_T_pi_Emna, self._MPN, delta) +
#                    einsum('n,Emnf,nb,fa-> Emab', self._MPW, eps_T_pi_Emna, self._MPN, delta))
#
#         return eps_p_Emab
#
#     #-------------------------------------------------------------------------
#     # Evaluation - get the corrector and predictor
#     #-------------------------------------------------------------------------
#
#     def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1, update_state, w_N_Emn, z_N_Emn,
#                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
#                       w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna):
#
#         # Corrector predictor computation.
#         if update_state:
#
#             eps_Emab_n = eps_Emab_n1 - deps_Emab
#
#             eps_N_Emn = self._get_e_N_Emn_2(eps_Emab_n)
#             eps_T_Emna = self._get_e_T_Emna_2(eps_Emab_n)
#
#             w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn = self.get_normal_law(
#                 eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)
#             w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna = self.get_tangential_law(
#                 eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)
#
#         #------------------------------------------------------------------
#         # Damage tensor (4th order) using product- or sum-type symmetrization:
#         #------------------------------------------------------------------
#         phi_Emn = self._get_phi_Emn(eps_Emab_n1, w_N_Emn, z_N_Emn,
#                                     alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
#                                     w_T_Emn, z_T_Emn, alpha_T_Emna,
#                                     eps_T_pi_Emna, sigma_N_Emn)
#
#         phi_Emab = self._get_phi_Emab(phi_Emn)
#
#         beta_Emabcd = self._get_beta_Emabcd(phi_Emab)
#
#         #------------------------------------------------------------------
#         # Damaged stiffness tensor calculated based on the damage tensor beta4:
#         #------------------------------------------------------------------
#         D_Emabcd = einsum(
#             'Emijab, abef, Emcdef -> Emijcd', beta_Emabcd, self.D_abef, beta_Emabcd)
#         #----------------------------------------------------------------------
#         # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
#         #----------------------------------------------------------------------
#         # plastic strain tensor
#         eps_p_Emab = self._get_eps_p_Emab(eps_Emab_n1, w_N_Emn, z_N_Emn,
#                                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
#                                           w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)
#
#         # elastic strain tensor
#         eps_e_Emab = eps_Emab_n1 - eps_p_Emab
#
#         # calculation of the stress tensor
#         sig_Emab = einsum('Emabcd,Emcd -> Emab', D_Emabcd, eps_e_Emab)
#
#         return D_Emabcd, sig_Emab

    #----------------------------------------------------------------
    #  the fourth order volumetric-identity tensor
    #----------------------------------------------------------------
    def _get_I_vol_abcd(self):

        delta = identity(3)
        I_vol_abcd = (1.0 / 3.0) * einsum('ab,cd -> abcd', delta, delta)
        return I_vol_abcd

    #----------------------------------------------------------------
    # Returns the fourth order deviatoric-identity tensor
    #----------------------------------------------------------------
    def _get_I_dev_abcd(self):

        delta = identity(3)
        I_dev_abcd = 0.5 * (einsum('ac,bd -> abcd', delta, delta) +
                            einsum('ad,bc -> abcd', delta, delta)) \
            - (1.0 / 3.0) * einsum('ab,cd -> abcd', delta, delta)

        return I_dev_abcd

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_vol_ab(self):

        delta = identity(3)
        P_vol_ab = (1.0 / 3.0) * delta
        return P_vol_ab

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_dev [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_dev_nabc(self):

        delta = identity(3)
        P_dev_nabc = 0.5 * einsum('nd,da,bc -> nabc', self._MPN, delta, delta)
        return P_dev_nabc

    #----------------------------------------------------------------
    # Returns the outer product of P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_PP_vol_abcd(self):

        delta = identity(3)
        PP_vol_abcd = (1.0 / 9.0) * einsum('ab,cd -> abcd', delta, delta)
        return PP_vol_abcd

    #----------------------------------------------------------------
    # Returns the inner product of P_dev
    #----------------------------------------------------------------
    def _get_PP_dev_nabcd(self):

        delta = identity(3)
        PP_dev_nabcd = 0.5 * (0.5 * (einsum('na,nc,bd -> nabcd', self._MPN, self._MPN, delta) +
                                     einsum('na,nd,bc -> nabcd', self._MPN, self._MPN, delta)) +
                              0.5 * (einsum('ac,nb,nd -> nabcd',  delta, self._MPN, self._MPN) +
                                     einsum('ad,nb,nc -> nabcd',  delta, self._MPN, self._MPN))) -\
            (1.0 / 3.0) * (einsum('na,nb,cd -> nabcd', self._MPN, self._MPN, delta) +
                           einsum('ab,nc,nd -> nabcd', delta, self._MPN, self._MPN)) +\
            (1.0 / 9.0) * einsum('ab,cd -> abcd', delta, delta)

        return PP_dev_nabcd

    def _get_I_vol_4(self):
        # The fourth order volumetric-identity tensor
        delta = identity(3)
        I_vol_ijkl = (1.0 / 3.0) * einsum('ij,kl -> ijkl', delta, delta)
        return I_vol_ijkl

    def _get_I_dev_4(self):
        # The fourth order deviatoric-identity tensor
        delta = identity(3)
        I_dev_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
                            einsum('il,jk -> ijkl', delta, delta)) \
            - (1 / 3.0) * einsum('ij,kl -> ijkl', delta, delta)

        return I_dev_ijkl

    def _get_P_vol(self):
        delta = identity(3)
        P_vol_ij = (1 / 3.0) * delta
        return P_vol_ij

    def _get_P_dev(self):
        delta = identity(3)
        P_dev_njkl = 0.5 * einsum('ni,ij,kl -> njkl', self._MPN, delta, delta)
        return P_dev_njkl

    def _get_PP_vol_4(self):
        # outer product of P_vol
        delta = identity(3)
        PP_vol_ijkl = (1 / 9.) * einsum('ij,kl -> ijkl', delta, delta)
        return PP_vol_ijkl

    def _get_PP_dev_4(self):
        # inner product of P_dev
        delta = identity(3)
        PP_dev_nijkl = 0.5 * (0.5 * (einsum('ni,nk,jl -> nijkl', self._MPN, self._MPN, delta) +
                                     einsum('ni,nl,jk -> nijkl', self._MPN, self._MPN, delta)) +
                              0.5 * (einsum('ik,nj,nl -> nijkl',  delta, self._MPN, self._MPN) +
                                     einsum('il,nj,nk -> nijkl',  delta, self._MPN, self._MPN))) -\
            (1 / 3.) * (einsum('ni,nj,kl -> nijkl', self._MPN, self._MPN, delta) +
                        einsum('ij,nk,nl -> nijkl', delta, self._MPN, self._MPN)) +\
            (1 / 9.) * einsum('ij,kl -> ijkl', delta, delta)
        return PP_dev_nijkl

    #--------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(29))
    #--------------------------------------------------------------------------
    def _get_S_1_Emabcd(self, eps_Emab, phi_Emn):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        #phi_Emn = 1.0 - self._get_omega(kappa)
        # print 'phi_Emn', phi_Emn

        PP_vol_abcd = self._get_PP_vol_abcd()
        PP_dev_nabcd = self._get_PP_dev_nabcd()
        I_dev_abcd = self._get_I_dev_abcd()

#         PP_vol_abcd = self._get_PP_vol_4()
#         PP_dev_nabcd = self._get_PP_dev_4()
#         I_dev_abcd = self._get_I_dev_4()

        S_1_Emabcd = K0 * einsum('Emn, n, abcd-> Emabcd', phi_Emn, self._MPW, PP_vol_abcd) + \
            G0 * 2.0 * self.zeta_G * einsum('Emn, n, nabcd-> Emabcd',
                                            phi_Emn, self._MPW, PP_dev_nabcd) - (1.0 / 3.0) * (
                2.0 * self.zeta_G - 1.0) * G0 * einsum('Emn, n, abcd-> Emabcd',
                                                       phi_Emn, self._MPW, I_dev_abcd)

        return S_1_Emabcd

#     #------------------------------------------
#     # scalar damage factor for each microplane
#     #------------------------------------------
#     def _get_d_Em(self, s_Emng, eps_Emab):
#
#         d_Emn = 1.0 - self.get_state_variables(s_Emng, eps_Emab)[0]
#
#         d_Em = (1.0 / 3.0) * einsum('Emn,n-> Em',  d_Emn, self._MPW)
#
#         return d_Em
#
#     #------------------------------------------
#     # The 4th order volumetric damage tensor
#     #------------------------------------------
#     def _get_M_vol_abcd(self, sctx, eps_app_eng, sigma_kk):
#
#         d = self._get_Em( s_Emng, eps_Emab)
#         delta = identity(2)
#
#         I_4th_abcd = 0.5 * (einsum('ac,bd -> ijkl', delta, delta) +
#                             einsum('il,jk -> ijkl', delta, delta))
#
#         # print 'M_vol', (1 - d) * I_4th_ijkl
#
#         return (1 - d) * I_4th_ijkl
#
#     #------------------------------------------
#     # The 4th order deviatoric damage tensor
#     #------------------------------------------
#     def _get_M_dev_tns(self, phi_mtx):
#
#         delta = identity(3)
#         I_4th_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
#                             einsum('il,jk -> ijkl', delta, delta))
#         tr_phi_mtx = np.trace(phi_mtx)
#
#         M_dev_ijkl = self.zeta_G * (0.5 * (einsum('ik,jl->ijkl', delta, phi_mtx) +
#                                            einsum('il,jk->ijkl', delta, phi_mtx)) +
#                                     0.5 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
#                                            einsum('il,jk->ijkl', phi_mtx, delta))) \
#             - (2. * self.zeta_G - 1.) * (tr_phi_mtx / 3.) * I_4th_ijkl
#
#         return M_dev_ijkl
#
#     #--------------------------------------------------------------------------
#     # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(31))
#     #--------------------------------------------------------------------------
#     def _get_S_2_Emabcd(self, sctx, eps_app_eng, sigma_kk):
#
#         K0 = self.E / (1. - 2. * self.nu)
#         G0 = self.E / (1. + self.nu)
#
#         I_vol_ijkl = self._get_I_vol_4()
#         I_dev_ijkl = self._get_I_dev_4()
#         phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
#         M_vol_ijkl = self._get_M_vol_tns(sctx, eps_app_eng, sigma_kk)
#         M_dev_ijkl = self._get_M_dev_tns(phi_mtx)
#
#         S_2_ijkl = K0 * einsum('ijmn,mnrs,rskl -> ijkl', I_vol_ijkl, M_vol_ijkl, I_vol_ijkl ) \
#             + G0 * einsum('ijmn,mnrs,rskl -> ijkl', I_dev_ijkl, M_dev_ijkl, I_dev_ijkl)\
#
#         return S_2_ijkl
#
#     #--------------------------------------------------------------------------
#     # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(34))
#     #--------------------------------------------------------------------------
#     def _get_S_3_Emabcd(self, sctx, eps_app_eng, sigma_kk):
#
#         K0 = self.E / (1. - 2. * self.nu)
#         G0 = self.E / (1. + self.nu)
#
#         I_vol_ijkl = self._get_I_vol_4()
#         I_dev_ijkl = self._get_I_dev_4()
#
#         # The fourth order elastic stiffness tensor
#         S_0_ijkl = K0 * I_vol_ijkl + G0 * I_dev_ijkl
#
#         d_n = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]
#
#         PP_vol_4 = self._get_PP_vol_4()
#         PP_dev_4 = self._get_PP_dev_4()
#
#         delta = identity(3)
#         I_4th_ijkl = einsum('ik,jl -> ijkl', delta, delta)
#
#         D_ijkl = einsum('n,n,ijkl->ijkl', d_n, self._MPW, PP_vol_4) + \
#             2 * self.zeta_G * einsum('n,n,nijkl->ijkl', d_n, self._MPW, PP_dev_4) - (
#                 1 / 3.) * (2 * self.zeta_G - 1) * einsum('n,n,ijkl->ijkl', d_n, self._MPW, I_dev_ijkl)
#
#         phi_ijkl = (I_4th_ijkl - D_ijkl)
#
#         S_ijkl = einsum('ijmn,mnkl', phi_ijkl, S_0_ijkl)
#
#         return S_ijkl
#
    #-------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor with the (double orthotropic) assumption
    #-------------------------------------------------------------------------
    def _get_S_4_Emabcd(self, eps_Emab, phi_Emab):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)
        I_vol_abcd = self._get_I_vol_abcd()
        I_dev_abcd = self._get_I_dev_abcd()

        delta = identity(3)
        #phi_Emab = self._get_phi_Emab(kappa)

        D_Emab = delta - phi_Emab

        d_Em = (1.0 / 3.0) * np.einsum('Emaa -> Em', D_Emab)

        D_bar_Emab = self.zeta_G * \
            (D_Emab - np.einsum('Em, ab -> Emab', d_Em, delta))

        S_4_Emabcd = K0 * I_vol_abcd - K0 * einsum('Em,abcd -> Emabcd', d_Em, I_vol_abcd) +\
            G0 * I_dev_abcd - G0 * einsum('Em,abcd -> Emabcd', d_Em, I_dev_abcd) +\
            (2.0 / 3.0) * (G0 - K0) * (einsum('ij,Emkl -> Emijkl', delta, D_bar_Emab) +
                                       einsum('Emij,kl -> Emijkl', D_bar_Emab, delta)) + 0.5 * (- K0 + 2.0 * G0) *\
            (0.5 * (einsum('ik,Emjl -> Emijkl', delta, D_bar_Emab) + einsum('Emil,jk -> Emijkl', D_bar_Emab, delta)) +
             0.5 * (einsum('Emil,jk -> Emijkl', D_bar_Emab, delta) + einsum('ik,Emjl -> Emijkl', delta, D_bar_Emab)))

        return S_4_Emabcd

    #----------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (restrctive orthotropic)
    #----------------------------------------------------------------------
    def _get_S_5_Emabcd(self, eps_Emab, phi_Emab):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        delta = identity(3)
        # commented out - @rch, @abdul please check
        # phi_Emab = self._get_phi_Emab(kappa)

        # damaged stiffness without simplification
        S_5_Emabcd = (1.0 / 3.0) * (K0 - G0) * 0.5 * ((einsum('ij,Emkl -> Emijkl', delta, phi_Emab) +
                                                       einsum('Emij,kl -> Emijkl', phi_Emab, delta))) + \
            G0 * 0.5 * ((0.5 * (einsum('ik,Emjl -> Emijkl', delta, phi_Emab) + einsum('Emil,jk -> Emijkl', phi_Emab, delta)) +
                         0.5 * (einsum('Emik,jl -> ijkl', phi_Emab, delta) + einsum('il,Emjk  -> Emijkl', delta, phi_Emab))))

        return S_5_Emabcd

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1, update_state, w_N_Emn, z_N_Emn,
                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna):

        if update_state:

            eps_Emab_n = eps_Emab_n1 - deps_Emab

            eps_N_Emn = self._get_e_N_Emn_2(eps_Emab_n)
            eps_T_Emna = self._get_e_T_Emna_2(eps_Emab_n)

            w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn = self.get_normal_law(
                eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna = self.get_tangential_law(
                eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        phi_Emn = self._get_phi_Emn(eps_Emab_n1, w_N_Emn, z_N_Emn,
                                    alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                                    w_T_Emn, z_T_Emn, alpha_T_Emna,
                                    eps_T_pi_Emna, sigma_N_Emn)

        phi_Emab = self._get_phi_Emab(phi_Emn)

        D_Emabcd = self._get_S_4_Emabcd(eps_Emab_n1, phi_Emab)

        sig_Emab = einsum('Emabcd,Emcd -> Emab', D_Emabcd, eps_Emab_n1)

        return D_Emabcd, sig_Emab

    #-----------------------------------------------
    # number of microplanes - currently fixed for 3D
    #-----------------------------------------------
    n_mp = Constant(28)

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        return array([[.577350259, .577350259, .577350259],
                      [.577350259, .577350259, -.577350259],
                      [.577350259, -.577350259, .577350259],
                      [.577350259, -.577350259, -.577350259],
                      [.935113132, .250562787, .250562787],
                      [.935113132, .250562787, -.250562787],
                      [.935113132, -.250562787, .250562787],
                      [.935113132, -.250562787, -.250562787],
                      [.250562787, .935113132, .250562787],
                      [.250562787, .935113132, -.250562787],
                      [.250562787, -.935113132, .250562787],
                      [.250562787, -.935113132, -.250562787],
                      [.250562787, .250562787, .935113132],
                      [.250562787, .250562787, -.935113132],
                      [.250562787, -.250562787, .935113132],
                      [.250562787, -.250562787, -.935113132],
                      [.186156720, .694746614, .694746614],
                      [.186156720, .694746614, -.694746614],
                      [.186156720, -.694746614, .694746614],
                      [.186156720, -.694746614, -.694746614],
                      [.694746614, .186156720, .694746614],
                      [.694746614, .186156720, -.694746614],
                      [.694746614, -.186156720, .694746614],
                      [.694746614, -.186156720, -.694746614],
                      [.694746614, .694746614, .186156720],
                      [.694746614, .694746614, -.186156720],
                      [.694746614, -.694746614, .186156720],
                      [.694746614, -.694746614, -.186156720]])

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        return array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505]) * 6.0

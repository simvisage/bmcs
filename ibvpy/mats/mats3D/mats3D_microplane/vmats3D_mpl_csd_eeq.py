'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 3D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from ibvpy.mats.mats_eval import IMATSEval
from numpy import array,\
    einsum, zeros_like, identity, sign,\
    sqrt
from traits.api import Constant, implements,\
    Float, Property, cached_property

import numpy as np
import traits.api as tr


class MATS3DMplCSDEEQ(MATS3DEval, MATS3D):

    implements(IMATSEval)

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

    state_array_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''
    @cached_property
    def _get_state_array_shapes(self):
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
    def get_normal_law(self, eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn):

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
    def get_tangential_law(self, eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_kk):

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

        eps_T_pi_Emna = np.zeros_like(eps_T_pi_Emna)

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

        #w_Emn = np.maximum(w_N_Emn, w_T_Emn)
        w_Emn[idx_1] = w_N_Emn[idx_1]
        w_Emn[idx_2] = w_T_Emn[idx_2]

        phi_Emn = sqrt(1.0 - w_Emn)

        return phi_Emn

    #----------------------------------------------
    # Returns the 2nd order damage tensor 'phi_mtx'
    #----------------------------------------------
    def _get_phi_Emab(self, phi_Emn):

        # integration terms for each microplanes
        phi_Emab = einsum('Emn,n,nab->Emab', phi_Emn, self._MPW, self._MPNN)

        return phi_Emab

    #----------------------------------------------------------------------
    # Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
    #(cf. [Jir99], Eq.(21))
    #----------------------------------------------------------------------
    def _get_beta_Emabcd(self, phi_Emab):

        delta = identity(3)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        beta_Emabcd = 0.25 * (einsum('Emik,jl->Emijkl', phi_Emab, delta) +
                              einsum('Emil,jk->Emijkl', phi_Emab, delta) +
                              einsum('Emjk,il->Emijkl', phi_Emab, delta) +
                              einsum('Emjl,ik->Emijkl', phi_Emab, delta))

        return beta_Emabcd

#     #----------------------------------------------------------------------
#     # Returns the 4th order damage tensor 'beta4' using product-type symmetrization
#     #(cf. [Baz97], Eq.(87))
#     #----------------------------------------------------------------------
#     def _get_beta_tns_product_type(self, sctx, eps_app_eng, sigma_kk):
#
#         delta = identity(2)
#
#         phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
#
#         n_dim = 2
#         phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
#         phi_eig_value_real = array([pe.real for pe in phi_eig_value])
#         phi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
#         for i in range(n_dim):
#             phi_pdc_mtx[i, i] = phi_eig_value_real[i]
#         # w_mtx = tensorial square root of the second order damage tensor:
#         w_pdc_mtx = sqrt(phi_pdc_mtx)
#
#         # transform the matrix w back to x-y-coordinates:
#         w_mtx = einsum('ik,kl,lj -> ij', phi_eig_mtx, w_pdc_mtx, phi_eig_mtx)
#         #w_mtx = dot(dot(phi_eig_mtx, w_pdc_mtx), transpose(phi_eig_mtx))
#
#         beta_ijkl = 0.5 * \
#             (einsum('ik,jl -> ijkl', w_mtx, w_mtx) +
#              einsum('il,jk -> ijkl', w_mtx, w_mtx))
#
#         return beta_ijkl

    #-----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    #-----------------------------------------------------------
    def _get_eps_p_Emab(self, eps_Emab, w_N_Emn, z_N_Emn,
                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna,  sigma_N_Emn):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emna_2(eps_Emab)

#         # plastic normal strains
#         eps_N_p_Emn = self.get_normal_law(
# eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[5]

        eps_N_p_Emn = self._get_eps_N_p_Emn(
            eps_Emab, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)

#         # sliding tangential strains
#         eps_T_pi_Emna = self.get_tangential_law(
# eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna,
# sigma_N_Emn)[4]

        eps_T_pi_Emna = self._get_eps_T_pi_Emna(
            eps_Emab, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        delta = identity(3)

        # 2-nd order plastic (inelastic) tensor
        eps_p_Emab = einsum('n,Emn,na,nb -> Emab', self._MPW, eps_N_p_Emn, self._MPN, self._MPN) + \
            0.5 * (einsum('n,Emnf,na,fb-> Emab', self._MPW, eps_T_pi_Emna, self._MPN, delta) +
                   einsum('n,Emnf,nb,fa-> Emab', self._MPW, eps_T_pi_Emna, self._MPN, delta))

        return eps_p_Emab

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1, update_state, w_N_Emn, z_N_Emn,
                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna):

        # Corrector predictor computation.
        if update_state:

            eps_Emab_n = eps_Emab_n1 - deps_Emab

            eps_N_Emn = self._get_e_N_Emn_2(eps_Emab_n)
            eps_T_Emna = self._get_e_T_Emna_2(eps_Emab_n)

            w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn = self.get_normal_law(
                eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna = self.get_tangential_law(
                eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        phi_Emn = self._get_phi_Emn(eps_Emab_n1, w_N_Emn, z_N_Emn,
                                    alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                                    w_T_Emn, z_T_Emn, alpha_T_Emna,
                                    eps_T_pi_Emna, sigma_N_Emn)

        phi_Emab = self._get_phi_Emab(phi_Emn)

        beta_Emabcd = self._get_beta_Emabcd(phi_Emab)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D_Emabcd = einsum(
            'Emijab, abef, Emcdef -> Emijcd', beta_Emabcd, self.D_abef, beta_Emabcd)
        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        # plastic strain tensor
        eps_p_Emab = self._get_eps_p_Emab(eps_Emab_n1, w_N_Emn, z_N_Emn,
                                          alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                                          w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # elastic strain tensor
        eps_e_Emab = eps_Emab_n1 - eps_p_Emab

        # calculation of the stress tensor
        sig_Emab = einsum('Emabcd,Emcd -> Emab', D_Emabcd, eps_e_Emab)

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

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
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
        la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2. + 2. * self.nu)
        return la, mu

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        la = self._get_lame_params()[0]
        mu = self._get_lame_params()[1]
        delta = identity(3)
        D_abef = (einsum(',ij,kl->ijkl', la, delta, delta) +
                  einsum(',ik,jl->ijkl', mu, delta, delta) +
                  einsum(',il,jk->ijkl', mu, delta, delta))

        return D_abef

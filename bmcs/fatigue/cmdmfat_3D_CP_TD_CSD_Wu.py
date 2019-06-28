'''
Created on 20.11.2017

@author: abaktheer

Microplane Fatigue model 

(compression plasticiy (CP) + Tensile Damage (TD) + Cumulative Damage Sliding (CSD))
 
Using the ODFs homogenization approach  [Wu, 2009]
'''

from numpy import \
    array, zeros, trace, einsum, zeros_like,\
    identity, sign, linspace, hstack
from traits.api import Constant, provides,\
    Float, HasTraits, Property, cached_property
from traitsui.api import View, Include

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
import matplotlib.pyplot as plt
import numpy as np


@provides(IMATSEval)
class MATSEvalMicroplaneFatigue(HasTraits):
    #--------------------------------
    # Elasticity material parameters
    #--------------------------------
    E = Float(34000.0,
              label="E",
              desc="Young modulus",
              enter_set=True,
              auto_set=False)

    nu = Float(0.2,
               label="nu",
               desc="poission ratio",
               enter_set=True,
               auto_set=False)

    #----------------------------------------
    # Tangential constitutive law parameters
    #----------------------------------------
    gamma_T = Float(5000.0,
                    label="Gamma_T",
                    desc="Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)

    K_T = Float(10.0,
                label="K_T",
                desc="Isotropic harening",
                enter_set=True,
                auto_set=False)

    S = Float(0.00001,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1.20,
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
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(0.0,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    #--------------------------------------------
    # Normal_Tension constitutive law parameters
    #--------------------------------------------
    Ad = Float(10000.0,
               label="a",
               desc="Brittlness parameter",
               enter_set=True,
               auto_set=False)

    eps_0 = Float(0.5e-4,
                  label="a",
                  desc="threshold strain",
                  enter_set=True,
                  auto_set=False)

    #------------------------------------------------
    # Normal_Compression constitutive law parameters
    #------------------------------------------------
    K_N = Float(10000.,
                label="K_N",
                desc="Normal isotropic harening",
                enter_set=True,
                auto_set=False)

    gamma_N = Float(15000.,
                    label="gamma_N",
                    desc="Normal kinematic hardening",
                    enter_set=True,
                    auto_set=False)

    sigma_0 = Float(20.0,
                    label="sigma_0",
                    desc="Yielding stress",
                    enter_set=True,
                    auto_set=False)

    #------------------------------------
    # anisotropy control parameter - [Wu]
    #------------------------------------

    zeta_G = Float(1.0,
                   label="zeta_G",
                   desc="anisotropy parameter",
                   enter_set=True,
                   auto_set=False)

    #------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    #------------------------------------------------------
    def get_normal_law(self, eps, sctx):
        # normal stiffness
        E_N = self.E / (1.0 - 2.0 * self.nu)
        # state variables
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
        # plasticity yield function
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
        # damage threshold function

        def f_w(Y): return 1 - 1. / (1 + self.Ad * (Y - Y_0))
        w_N = f_w(Y_N) * thres_2
        z_N = - w_N * thres_2

        new_sctx = zeros((28, 5))

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
        alpha_T = sctx[:, 7:10]
        eps_T_pi = sctx[:, 10:13]

        sig_pi_trial = E_T * (e_T - eps_T_pi)
        Z = self.K_T * z_T
        X = self.gamma_T * alpha_T
        norm_1 = np.sqrt(
            einsum('nj,nj -> n', (sig_pi_trial - X), (sig_pi_trial - X)))

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_kk / 3.0

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - w_T) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            einsum('nj,nj -> n', (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi[:, 0] = eps_T_pi[:, 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[:, 0] - X[:, 0]) / (1.0 - w_T)) / norm_2
        eps_T_pi[:, 1] = eps_T_pi[:, 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[:, 1] - X[:, 1]) / (1.0 - w_T)) / norm_2
        eps_T_pi[:, 2] = eps_T_pi[:, 2] + plas_1 * delta_lamda * \
            ((sig_pi_trial[:, 2] - X[:, 2]) / (1.0 - w_T)) / norm_2

        Y = 0.5 * E_T * \
            einsum('nj,nj -> n', (e_T - eps_T_pi), (e_T - eps_T_pi))

        w_T += ((1 - w_T) ** self.c) * \
            (delta_lamda * (Y / self.S) ** self.r) * \
            (self.tau_pi_bar / (self.tau_pi_bar - self.a * sigma_kk / 3.0))

        alpha_T[:, 0] = alpha_T[:, 0] + plas_1 * delta_lamda *\
            (sig_pi_trial[:, 0] - X[:, 0]) / norm_2
        alpha_T[:, 1] = alpha_T[:, 1] + plas_1 * delta_lamda *\
            (sig_pi_trial[:, 1] - X[:, 1]) / norm_2
        alpha_T[:, 2] = alpha_T[:, 2] + plas_1 * delta_lamda *\
            (sig_pi_trial[:, 2] - X[:, 2]) / norm_2

        z_T = z_T + delta_lamda

        new_sctx = zeros((28, 8))
        new_sctx[:, 0] = w_T
        new_sctx[:, 1] = z_T
        new_sctx[:, 2:5] = alpha_T
        new_sctx[:, 5:8] = eps_T_pi
        return new_sctx


@provides(IMATSEval)
class MATSXDMicroplaneDamageFatigueWu(MATSEvalMicroplaneFatigue):
    '''
    Microplane Damage Fatigue Model.
    '''

    #-------------------------------------------------------------------------
    #  MICROPLANE-Kinematic constraints
    #-------------------------------------------------------------------------

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

    # Projection of apparent strain onto the individual microplanes
    def _get_e_vct_arr(self, eps_eng):
        e_ni = einsum('nj,ji->ni', self._MPN, eps_eng)
        return e_ni

    # get the normal strain array for each microplane
    def _get_e_N_arr(self, e_vct_arr):
        eN_n = einsum('ni,ni->n', e_vct_arr, self._MPN)
        return eN_n

    # get the tangential strain vector array for each microplane
    def _get_e_T_vct_arr(self, e_vct_arr):
        eN_n = self._get_e_N_arr(e_vct_arr)
        eN_vct_ni = einsum('n,ni->ni', eN_n, self._MPN)
        return e_vct_arr - eN_vct_ni

    #---------------------------------------------------
    # Alternative methods for the kinematic constraints
    #---------------------------------------------------
    def _get_e_N_arr_2(self, eps_eng):
        return einsum('nij,ij->n', self._MPNN, eps_eng)

    def _get_e_T_vct_arr_2(self, eps_eng):
        MPTT_ijr = self._get__MPTT()
        return einsum('nijr,ij->nr', MPTT_ijr, eps_eng)

    def _get_e_vct_arr_2(self, eps_eng):
        return self._e_N_arr_2 * self._MPN + self._e_t_vct_arr_2

    #--------------------------------------------------------
    # return the state variables (damage , inelastic strains)
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

    #-------------------------------------------------------------
    # Returns a list of the integrity factors for all microplanes.
    #-------------------------------------------------------------
    def _get_phi_arr(self, sctx, eps_app_eng, sigma_kk):

        w_n = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 0]
        w_T = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]

        w = np.zeros(self.n_mp)

        for i in range(0, self.n_mp):
            w[i] = np.maximum(w_n[i], w_T[i])

        phi_arr = 1.0 - w

        return phi_arr

    #-------------------------------------------------------------
    # Returns the 2nd order integrity tensor 'phi_mtx'
    #-------------------------------------------------------------
    def _get_phi_mtx(self, sctx, eps_app_eng, sigma_kk):

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng, sigma_kk)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        return phi_ij

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
            sctx, eps_app_eng, sigma_kk)[:, 10:13]

        return eps_T_pi_vct_arr

    #-------------------------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane and return the plastic strain tensor
    #-------------------------------------------------------------------------
    def _get_eps_p_mtx(self, sctx, eps_app_eng, sigma_kk):

        # plastic normal strains
        eps_N_P_n = self._get_eps_N_p_arr(sctx, eps_app_eng, sigma_kk)

        # sliding tangential strains
        eps_T_pi_ni = self._get_eps_T_pi_arr(sctx, eps_app_eng, sigma_kk)
        delta = identity(3)

        # 2-nd order plastic (inelastic) tensor
        eps_p_ij = einsum('n,n,ni,nj -> ij', self._MPW, eps_N_P_n, self._MPN, self._MPN) + \
            0.5 * (einsum('n,nr,ni,rj->ij', self._MPW, eps_T_pi_ni, self._MPN, delta) +
                   einsum('n,nr,nj,ri->ij', self._MPW, eps_T_pi_ni, self._MPN, delta))

        return eps_p_ij

    '''----------------------------------------------------------------------------------------
    Construct the irreducible secant stiffness tensor (cf. [Wu.2009]) with different equations
    ----------------------------------------------------------------------------------------'''

    #----------------------------------------------------------------
    #  the fourth order volumetric-identity tensor
    #----------------------------------------------------------------
    def _get_I_vol_4(self):

        delta = identity(3)
        I_vol_ijkl = (1.0 / 3.0) * einsum('ij,kl -> ijkl', delta, delta)
        return I_vol_ijkl

    #----------------------------------------------------------------
    # Returns the fourth order deviatoric-identity tensor
    #----------------------------------------------------------------
    def _get_I_dev_4(self):

        delta = identity(3)
        I_dev_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
                            einsum('il,jk -> ijkl', delta, delta)) \
            - (1. / 3.0) * einsum('ij,kl -> ijkl', delta, delta)

        return I_dev_ijkl

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_vol(self):

        delta = identity(3)
        P_vol_ij = (1. / 3.0) * delta
        return P_vol_ij

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_dev [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_dev(self):

        delta = identity(3)
        P_dev_njkl = 0.5 * einsum('ni,ij,kl -> njkl', self._MPN, delta, delta)
        return P_dev_njkl

    #----------------------------------------------------------------
    # Returns the outer product of P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_PP_vol_4(self):

        delta = identity(3)
        PP_vol_ijkl = (1. / 9.) * einsum('ij,kl -> ijkl', delta, delta)
        return PP_vol_ijkl

    #----------------------------------------------------------------
    # Returns the inner product of P_dev
    #----------------------------------------------------------------
    def _get_PP_dev_4(self):

        delta = identity(3)
        PP_dev_nijkl = 0.5 * (0.5 * (einsum('ni,nk,jl -> nijkl', self._MPN, self._MPN, delta) +
                                     einsum('ni,nl,jk -> nijkl', self._MPN, self._MPN, delta)) +
                              0.5 * (einsum('ik,nj,nl -> nijkl',  delta, self._MPN, self._MPN) +
                                     einsum('il,nj,nk -> nijkl',  delta, self._MPN, self._MPN))) -\
            (1. / 3.) * (einsum('ni,nj,kl -> nijkl', self._MPN, self._MPN, delta) +
                         einsum('ij,nk,nl -> nijkl', delta, self._MPN, self._MPN)) +\
            (1. / 9.) * einsum('ij,kl -> ijkl', delta, delta)

        return PP_dev_nijkl

    #--------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(29))
    #--------------------------------------------------------------------------
    def _get_S_1_tns(self, sctx, eps_app_eng, sigma_kk):

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        phi_n = self._get_phi_arr(sctx, eps_app_eng, sigma_kk)

        PP_vol_4 = self._get_PP_vol_4()
        PP_dev_4 = self._get_PP_dev_4()
        I_dev_4 = self._get_I_dev_4()

        S_1_ijkl = K0 * einsum('n,n,ijkl->ijkl', phi_n, self._MPW, PP_vol_4) + \
            G0 * 2 * self.zeta_G * einsum('n,n,nijkl->ijkl', phi_n, self._MPW, PP_dev_4) - (1. / 3.) * (
                2 * self.zeta_G - 1) * G0 * einsum('n,n,ijkl->ijkl', phi_n, self._MPW, I_dev_4)

        return S_1_ijkl

    #------------------------------------------
    # scalar damage factor for each microplane
    #------------------------------------------
    def _get_d_scalar(self, sctx, eps_app_eng, sigma_kk):

        d_n = 1.0 - self._get_phi_arr(sctx, eps_app_eng, sigma_kk)

        d = (1.0 / 3.0) * einsum('n,n->',  d_n, self._MPW)

        return d

    #------------------------------------------
    # The 4th order volumetric damage tensor
    #------------------------------------------
    def _get_M_vol_tns(self, sctx, eps_app_eng, sigma_kk):

        d = self._get_d_scalar(sctx, eps_app_eng, sigma_kk)
        delta = identity(3)

        I_4th_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
                            einsum('il,jk -> ijkl', delta, delta))

        # print 'M_vol', (1 - d) * I_4th_ijkl

        return (1 - d) * I_4th_ijkl

    #------------------------------------------
    # The 4th order deviatoric damage tensor
    #------------------------------------------
    def _get_M_dev_tns(self, phi_mtx):

        delta = identity(3)
        I_4th_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
                            einsum('il,jk -> ijkl', delta, delta))
        tr_phi_mtx = trace(phi_mtx)

        M_dev_ijkl = self.zeta_G * (0.5 * (einsum('ik,jl->ijkl', delta, phi_mtx) +
                                           einsum('il,jk->ijkl', delta, phi_mtx)) +
                                    0.5 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                                           einsum('il,jk->ijkl', phi_mtx, delta))) \
            - (2. * self.zeta_G - 1.) * (tr_phi_mtx / 3.) * I_4th_ijkl

        return M_dev_ijkl

    #--------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(31))
    #--------------------------------------------------------------------------
    def _get_S_2_tns(self, sctx, eps_app_eng, sigma_kk):

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
        M_vol_ijkl = self._get_M_vol_tns(sctx, eps_app_eng, sigma_kk)
        M_dev_ijkl = self._get_M_dev_tns(phi_mtx)

        S_2_ijkl = K0 * einsum('ijmn,mnrs,rskl -> ijkl', I_vol_ijkl, M_vol_ijkl, I_vol_ijkl) \
            + G0 * einsum('ijmn,mnrs,rskl -> ijkl', I_dev_ijkl, M_dev_ijkl, I_dev_ijkl)\

        return S_2_ijkl

    #--------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(34))
    #--------------------------------------------------------------------------
    def _get_S_3_tns(self, sctx, eps_app_eng, sigma_kk):

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()

        # The fourth order elastic stiffness tensor
        S_0_ijkl = K0 * I_vol_ijkl + G0 * I_dev_ijkl

        d_n = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]

        PP_vol_4 = self._get_PP_vol_4()
        PP_dev_4 = self._get_PP_dev_4()

        delta = identity(3)
        I_4th_ijkl = einsum('ik,jl -> ijkl', delta, delta)

        D_ijkl = einsum('n,n,ijkl->ijkl', d_n, self._MPW, PP_vol_4) + \
            2 * self.zeta_G * einsum('n,n,nijkl->ijkl', d_n, self._MPW, PP_dev_4) - (
                1 / 3.) * (2 * self.zeta_G - 1) * einsum('n,n,ijkl->ijkl', d_n, self._MPW, I_dev_ijkl)

        phi_ijkl = (I_4th_ijkl - D_ijkl)

        S_ijkl = einsum('ijmn,mnkl', phi_ijkl, S_0_ijkl)

        return S_ijkl

    #-------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor using (double orthotropic) assumption
    #-------------------------------------------------------------------------
    def _get_S_4_tns(self, sctx, eps_app_eng, sigma_kk):

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()
        delta = identity(3)
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
        D_ij = delta - phi_mtx
        d = (1. / 3.) * trace(D_ij)
        D_bar_ij = self.zeta_G * (D_ij - d * delta)

        S_4_ijkl = (1 - d) * K0 * I_vol_ijkl + (1 - d) * G0 * I_dev_ijkl + (2 / 3.) * (G0 - K0) * \
            (einsum('ij,kl -> ijkl', delta, D_bar_ij) +
             einsum('ij,kl -> ijkl', D_bar_ij, delta)) + 0.5 * (- K0 + 2 * G0) *\
            (0.5 * (einsum('ik,jl -> ijkl', delta, D_bar_ij) + einsum('il,jk -> ijkl', D_bar_ij, delta)) +
             0.5 * (einsum('il,jk -> ijkl', D_bar_ij, delta) + einsum('ik,jl -> ijkl', delta, D_bar_ij)))

        return S_4_ijkl

    #-------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (double orthotropic N-T split)
    #-------------------------------------------------------------------------
    def _get_S_5_tns(self, sctx, eps_app_eng, sigma_kk):

        E_N = self.E / (3.0 - 2.0 * (1.0 + self.nu))
        E_T = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()
        delta = identity(3)
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
        D_ij = delta - phi_mtx
        d = (1. / 3.) * trace(D_ij)
        D_bar_ij = self.zeta_G * (D_ij - d * delta)

        S_5_ijkl = (1 - d) * E_N * I_vol_ijkl + (1 - d) * E_T * I_dev_ijkl + (2 / 3.) * (E_T - E_N) * \
            (einsum('ij,kl -> ijkl', delta, D_bar_ij) +
             einsum('ij,kl -> ijkl', D_bar_ij, delta)) + 0.5 * (2 * E_T - E_N) *\
            (0.5 * (einsum('ik,jl -> ijkl', delta, D_bar_ij) + einsum('il,jk -> ijkl', D_bar_ij, delta)) +
             0.5 * (einsum('il,jk -> ijkl', D_bar_ij, delta) + einsum('ik,jl -> ijkl', delta, D_bar_ij)))

        return S_5_ijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------
    def get_corr_pred(self, sctx, eps_app_eng, sigma_kk):

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------

        # secant stiffness tensor
        S_ijkl = self._get_S_4_tns(sctx, eps_app_eng, sigma_kk)

        # plastic strain tensor
        eps_p_ij = self._get_eps_p_mtx(sctx, eps_app_eng, sigma_kk)

        # elastic strain tensor
        eps_e_mtx = eps_app_eng - eps_p_ij

        # calculation of the stress tensor
        sig_eng = einsum('ijmn,mn -> ij', S_ijkl, eps_e_mtx)

        return sig_eng, S_ijkl


@provides(IMATSEval)
class MATS3DMicroplaneDamageWu(MATSXDMicroplaneDamageFatigueWu, MATS3DEval):

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
        # microplane normals:
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
        # Note that the values in the array must be multiplied by 6 (cf. [Baz05])!
        # The sum of of the array equals 0.5. (cf. [BazLuz04]))
        # The values are given for an Gaussian integration over the unit
        # hemisphere.
        return array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505]) * 6.0

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

    @cached_property
    def _get_elasticity_tensors(self):
        '''
        Intialize the fourth order elasticity tensor for 3D or 2D plane strain or 2D plane stress
        '''
        # ----------------------------------------------------------------------------
        # Lame constants calculated from E and nu
        # ----------------------------------------------------------------------------

        # first Lame paramter
        la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2 + 2 * self.nu)

        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------

        # construct the elasticity tensor (using Numpy - einsum function)
        delta = identity(3)
        D_ijkl = (einsum(',ij,kl->ijkl', la, delta, delta) +
                  einsum(',ik,jl->ijkl', mu, delta, delta) +
                  einsum(',il,jk->ijkl', mu, delta, delta))

        return D_ijkl


if __name__ == '__main__':
    #==========================================================================
    # Check the model behavior at the single material point
    #==========================================================================

    model = MATS3DMicroplaneDamageWu()

    p = 1.0  # ratio of strain eps_11 (for bi-axial loading)
    m = 0.0  # ratio of strain eps_22 (for bi-axial loading)

    #------------------------------------
    # monotonic loading
    #------------------------------------
    n = 100  # number of increments
    s_levels = linspace(0, -0.02, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    s_history_1 = s_levels.flatten()
    s_arr_1 = hstack([linspace(s_history_1[i], s_history_1[i + 1], n)
                      for i in range(len(s_levels) - 1)])

    eps_1 = array([array([[p * s_arr_1[i], 0, 0],
                          [0, m * s_arr_1[i], 0],
                          [0, 0, 0]]) for i in range(0, len(s_arr_1))])

    #--------------------------------------
    # construct the arrays
    #--------------------------------------
    sigma_1 = zeros_like(eps_1)
    sigma_kk_1 = zeros(len(s_arr_1) + 1)
    w_1_N = zeros((len(eps_1[:, 0, 0]), 28))
    w_1_T = zeros((len(eps_1[:, 0, 0]), 28))
    eps_P_N_1 = zeros((len(eps_1[:, 0, 0]), 28))
    eps_Pi_T_1 = zeros((len(eps_1[:, 0, 0]), 28, 3))
    sctx_1 = zeros((len(eps_1[:, 0, 0]) + 1, 28, 13))

    for i in range(0, len(eps_1[:, 0, 0])):
        sigma_1[i, :] = model.get_corr_pred(
            sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])[0]
        sigma_kk_1[i + 1] = trace(sigma_1[i, :])
        sctx_1[
            i + 1] = model._get_state_variables(sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])

        w_1_N[i, :] = sctx_1[i, :, 0]
        w_1_T[i, :] = sctx_1[i, :, 5]
        eps_P_N_1[i, :] = sctx_1[i, :, 4]
        eps_Pi_T_1[i, :, :] = sctx_1[i, :, 10:13]

    #-------------------------------------
    # cyclic loading
    #-------------------------------------
    s_history_2 = [-0, -0.001, -0.00034, -
                   0.0015, -0.00065, -0.0020, -0.00097,
                   -0.0027, -0.00145, -0.004, -0.0022, -0.0055,
                   -0.0031, -0.007, -0.004, -0.008, -0.0046, -.009,
                   -0.0052, -0.01, -0.0058, -0.012, -0.0068, -0.015]

    #s_history_2 = [0, 0.02]

    s_arr_2 = hstack([linspace(s_history_2[i], s_history_2[i + 1], 100)
                      for i in range(len(s_history_2) - 1)])

    eps_2 = array([array([[p * s_arr_2[i], 0, 0],
                          [0,  m * s_arr_2[i], 0],
                          [0, 0, 0]]) for i in range(0, len(s_arr_2))])

    #--------------------------------------
    # construct the arrays
    #--------------------------------------
    sigma_2 = zeros_like(eps_2)
    sigma_kk_2 = zeros(len(s_arr_2) + 1)
    w_2_N = zeros((len(eps_2[:, 0, 0]), 28))
    w_2_T = zeros((len(eps_2[:, 0, 0]), 28))
    eps_P_N_2 = zeros((len(eps_2[:, 0, 0]), 28))
    eps_Pi_T_2 = zeros((len(eps_2[:, 0, 0]), 28, 3))
    sctx_2 = zeros((len(eps_2[:, 0, 0]) + 1, 28, 13))

    for i in range(0, len(eps_2[:, 0, 0])):

        sigma_2[i, :] = model.get_corr_pred(
            sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])[0]
        sigma_kk_2[i + 1] = trace(sigma_2[i, :])
        sctx_2[
            i + 1] = model._get_state_variables(sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])

        w_2_N[i, :] = sctx_2[i, :, 0]
        w_2_T[i, :] = sctx_2[i, :, 5]
        eps_P_N_2[i, :] = sctx_2[i, :, 4]
        eps_Pi_T_2[i, :, :] = sctx_2[i, :, 10:13]

    '''====================================================
    plotting
    ===================================================='''

    #------------------------------------------------------
    # stress -strain
    #------------------------------------------------------
    plt.subplot(221)
    plt.plot(eps_1[:, 0, 0], sigma_1[:, 0, 0], color='k',
             linewidth=1, label='sigma_11_(monotonic)')
    plt.plot(eps_1[:, 0, 0], sigma_1[:, 1, 1],
             '--k', linewidth=1, label='sigma_22')
    #plt.plot(eps_1[:, 0, 0], sigma_1[:, 0, 1], linewidth=1, label='sigma_12')
    plt.plot(eps_2[:, 0, 0], sigma_2[:, 0, 0], color='g',
             linewidth=1, label='sigma_11_(cyclic)')

    plt.title('$\sigma - \epsilon$')
    plt.xlabel('strain')
    plt.ylabel('stress(MPa)')
    plt.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='k', linewidth=1, alpha=0.5)
    plt.legend()

    #------------------------------------------------------
    # normal damage at the microplanes (TD)
    #------------------------------------------------------
    plt.subplot(222)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], w_1_N[:, i], linewidth=1.0, label='cyclic', alpha=1)
        plt.plot(
            eps_2[:, 0, 0], w_2_N[:, i], linewidth=1.0, label='monotonic', alpha=1)

        plt.xlabel('strain')
        plt.ylabel('damage')
        plt.title(' normal damage for all microplanes')

    #---------------------------------------------------------
    # tangential damage at the microplanes (CSD)
    #---------------------------------------------------------
    plt.subplot(223)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], w_1_T[:, i], linewidth=1.0, label='cyclic', alpha=1)
        plt.plot(
            eps_2[:, 0, 0], w_2_T[:, i], linewidth=1.0, label='monotonic', alpha=1)

        plt.xlabel('strain')
        plt.ylabel('damage')
        plt.title(' tangential damage for all microplanes')

    #-----------------------------------------------------------
    # damage with sliding strains at the microplanes (CSD)
    #-----------------------------------------------------------
    plt.subplot(224)
    for i in range(0, 28):

        plt.plot(eps_Pi_T_1[:, i, 1], w_1_T[:, i])
        plt.plot(eps_Pi_T_2[:, i, 1], w_2_T[:, i])

        plt.xlabel('sliding strain')
        plt.ylabel('damage')

    plt.show()

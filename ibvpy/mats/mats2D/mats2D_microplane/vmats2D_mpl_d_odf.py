'''
Created on 08.02.2018

@author: abaktheer

Microplane damage model 2D - ODFS (Wu [2009])
'''

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from numpy import \
    array, einsum, identity, sqrt
from traits.api import \
    Constant, \
    Float, Property, cached_property

from ibvpy.mats.matsXD.vmatsXD_eval import MATSXDEval
import numpy as np
import traits.api as tr


class MATSXDMicroplaneDamageODF(MATSXDEval):

    epsilon_0 = Float(59e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    epsilon_f = Float(250e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    c_T = Float(0.00,
                label="a",
                desc="Lateral pressure coefficient",
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
    def _get_state_var_shapes(self):
        return {'kappa': (self.n_mp,),
                'omega': (self.n_mp,)}

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
                           einsum('nj,ir -> njir', self._MPN, delta) - 2.0 *
                           einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN))
        return MPTT_nijr

    def _get_e_Emna(self, eps_Emab):
        # Projection of apparent strain onto the individual microplanes
        e_ni = einsum('nb,Emba->Emna', self._MPN, eps_Emab)
        return e_ni

    def _get_e_N_Emn(self, e_Emna):
        # get the normal strain array for each microplane
        e_N_Emn = einsum('Emna, na->Emn', e_Emna, self._MPN)
        return e_N_Emn

    def _get_e_equiv_Emn(self, e_Emna):
        '''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        e_N_Emn = self._get_e_N_Emn(e_Emna)
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_Emn = (np.abs(e_N_Emn) + e_N_Emn) / 2
        # normal strain vector for each microplane
        e_N_Emna = einsum('Emn,ni -> Emni', e_N_Emn, self._MPN)
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_Emna = e_Emna - e_N_Emna
        # squared tangential strain vector for each microplane
        e_TT_Emn = einsum('Emni,Emni -> Emn', e_T_Emna, e_T_Emna)
        # equivalent strain for each microplane
        e_equiv_Emn = sqrt(e_N_pos_Emn * e_N_pos_Emn + c_T * e_TT_Emn)
        return e_equiv_Emn

    def update_state_variables(self, eps_Emab, kappa, omega):

        e_Emna = self._get_e_Emna(eps_Emab)
        eps_eq_Emn = self._get_e_equiv_Emn(e_Emna)
        f_trial_Emn = eps_eq_Emn - self.epsilon_0
        I = np.where(f_trial_Emn > 0)
        kappa[I] = eps_eq_Emn[I]
        omega[I] = self._get_omega(eps_eq_Emn[I])
        return I

    def _get_omega(self, kappa_Emn):
        '''
        Return new value of damage parameter
        @param kappa:
        '''
        omega_Emn = np.zeros_like(kappa_Emn)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        kappa_idx = np.where(kappa_Emn >= epsilon_0)
        omega_Emn[kappa_idx] = (
            1.0 - (epsilon_0 / kappa_Emn[kappa_idx] *
                   np.exp(-1.0 * (kappa_Emn[kappa_idx] - epsilon_0) /
                          (epsilon_f - epsilon_0))
                   ))
        return omega_Emn

    def _get_phi_Emab(self, kappa_Emn):
        # Returns the 2nd order damage tensor 'phi_mtx'
        # scalar integrity factor for each microplane
        phi_Emn = 1.0 - self._get_omega(kappa_Emn)
        # integration terms for each microplanes
        phi_Emab = einsum('Emn,n,nab->Emab', phi_Emn, self._MPW, self._MPNN)
        return phi_Emab

#     def _get_beta_Emabcd(self, phi_Emab):
#         '''
#         Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
#         (cf. [Jir99], Eq.(21))
#         '''
#         delta = identity(2)
#         beta_Emijkl = 0.25 * (einsum('Emik,jl->Emijkl', phi_Emab, delta) +
#                               einsum('Emil,jk->Emijkl', phi_Emab, delta) +
#                               einsum('Emjk,il->Emijkl', phi_Emab, delta) +
#                               einsum('Emjl,ik->Emijkl', phi_Emab, delta))
#
#         return beta_Emijkl

    #----------------------------------------------------------------
    #  the fourth order volumetric-identity tensor
    #----------------------------------------------------------------
    def _get_I_vol_abcd(self):

        delta = identity(2)
        I_vol_abcd = (1.0 / 3.0) * einsum('ab,cd -> abcd', delta, delta)
        return I_vol_abcd

    #----------------------------------------------------------------
    # Returns the fourth order deviatoric-identity tensor
    #----------------------------------------------------------------
    def _get_I_dev_abcd(self):

        delta = identity(2)
        I_dev_abcd = 0.5 * (einsum('ac,bd -> abcd', delta, delta) +
                            einsum('ad,bc -> abcd', delta, delta)) \
            - (1. / 3.0) * einsum('ab,cd -> abcd', delta, delta)

        return I_dev_abcd

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_vol_ab(self):

        delta = identity(2)
        P_vol_ab = (1.0 / 3.0) * delta
        return P_vol_ab

    #----------------------------------------------------------------
    # Returns the fourth order tensor P_dev [Wu.2009]
    #----------------------------------------------------------------
    def _get_P_dev_nabc(self):

        delta = identity(2)
        P_dev_nabc = 0.5 * einsum('nd,da,bc -> nabc', self._MPN, delta, delta)
        return P_dev_nabc

    #----------------------------------------------------------------
    # Returns the outer product of P_vol [Wu.2009]
    #----------------------------------------------------------------
    def _get_PP_vol_abcd(self):

        delta = identity(2)
        PP_vol_abcd = (1.0 / 9.0) * einsum('ab,cd -> abcd', delta, delta)
        return PP_vol_abcd

    #----------------------------------------------------------------
    # Returns the inner product of P_dev
    #----------------------------------------------------------------
    def _get_PP_dev_nabcd(self):

        delta = identity(2)
        PP_dev_nabcd = 0.5 * (0.5 * (einsum('na,nc,bd -> nabcd', self._MPN, self._MPN, delta) +
                                     einsum('na,nd,bc -> nabcd', self._MPN, self._MPN, delta)) +
                              0.5 * (einsum('ac,nb,nd -> nabcd', delta, self._MPN, self._MPN) +
                                     einsum('ad,nb,nc -> nabcd', delta, self._MPN, self._MPN))) - \
            (1.0 / 3.0) * (einsum('na,nb,cd -> nabcd', self._MPN, self._MPN, delta) +
                           einsum('ab,nc,nd -> nabcd', delta, self._MPN, self._MPN)) + \
            (1.0 / 9.0) * einsum('ab,cd -> abcd', delta, delta)

        return PP_dev_nabcd

    #--------------------------------------------------------------------------
    # Returns the fourth order secant stiffness tensor (cf. [Wu.2009], Eq.(29))
    #--------------------------------------------------------------------------
    def _get_S_1_Emabcd(self, eps_Emab, kappa, omega):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        phi_Emn = 1.0 - self._get_omega(kappa)

        PP_vol_abcd = self._get_PP_vol_abcd()
        PP_dev_nabcd = self._get_PP_dev_nabcd()
        I_dev_abcd = self._get_I_dev_abcd()

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
    # Returns the fourth order secant stiffness tensor using (double orthotropic) assumption
    #-------------------------------------------------------------------------
    def _get_S_4_Emabcd(self, eps_Emab, kappa, omega):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        I_vol_abcd = self._get_I_vol_abcd()
        I_dev_abcd = self._get_I_dev_abcd()
        delta = identity(2)

        phi_Emab = self._get_phi_Emab(kappa)
        D_Emab = delta - phi_Emab
        d_Em = (1.0 / 3.0) * np.trace(D_Emab)
        D_bar_Emab = self.zeta_G * (D_Emab - d_Em * delta)

        S_4_Emabcd = (1.0 - d_Em) * K0 * I_vol_abcd + (1.0 - d_Em) * G0 * I_dev_abcd + (2.0 / 3.0) * (G0 - K0) * \
            (einsum('ij,Emkl -> Emijkl', delta, D_bar_Emab) +
             einsum('Emij,kl -> Emijkl', D_bar_Emab, delta)) + 0.5 * (-K0 + 2.0 * G0) * \
            (0.5 * (einsum('ik,Emjl -> Emijkl', delta, D_bar_Emab) + einsum('Emil,jk -> Emijkl', D_bar_Emab, delta)) +
             0.5 * (einsum('Emil,jk -> Emijkl', D_bar_Emab, delta) + einsum('ik,Emjl -> Emijkl', delta, D_bar_Emab)))

        return S_4_Emabcd

    def _get_S_5_Emabcd(self, eps_Emab, kappa, omega):
        #----------------------------------------------------------------------
        # Returns the fourth order secant stiffness tensor (restrctive orthotropic)
        #----------------------------------------------------------------------

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        delta = identity(2)
        phi_Emab = self._get_phi_Emab(kappa)
        # damaged stiffness without simplification
        S_5_Emabcd = (1.0 / 3.0) * (K0 - G0) * 0.5 * ((einsum('ij,Emkl -> Emijkl', delta, phi_Emab) +
                                                       einsum('Emij,kl -> Emijkl', phi_Emab, delta))) + \
            G0 * 0.5 * ((0.5 * (einsum('ik,Emjl -> Emijkl', delta, phi_Emab) + einsum('Emil,jk -> Emijkl', phi_Emab, delta)) +
                         0.5 * (einsum('Emik,jl -> ijkl', phi_Emab, delta) + einsum('il,Emjk  -> Emijkl', delta, phi_Emab))))

        # print 'S_5_Emabcd', S_5_Emabcd
        return S_5_Emabcd
#
#     #-------------------------------------------------------------------------
#     # Returns the fourth order secant stiffness tensor (double orthotropic N-T split)
#     #-------------------------------------------------------------------------
#     def _get_S_5_Emabcd(self, sctx, eps_app_eng, sigma_kk):
#
#         E_N = self.E / (3.0 - 2.0 * (1.0 + self.nu))
#         E_T = self.E / (1. + self.nu)
#
#         I_vol_ijkl = self._get_I_vol_4()
#         I_dev_ijkl = self._get_I_dev_4()
#         delta = identity(3)
#         phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)
#         D_ij = delta - phi_mtx
#         d = (1. / 3.) * np.trace(D_ij)
#         D_bar_ij = self.zeta_G * (D_ij - d * delta)
#
#         S_5_ijkl = (1 - d) * E_N * I_vol_ijkl + (1 - d) * E_T * I_dev_ijkl + (2 / 3.) * (E_T - E_N) * \
#             (einsum('ij,kl -> ijkl', delta, D_bar_ij) +
#              einsum('ij,kl -> ijkl', D_bar_ij, delta)) + 0.5 * (2 * E_T - E_N) *\
#             (0.5 * (einsum('ik,jl -> ijkl', delta, D_bar_ij) + einsum('il,jk -> ijkl', D_bar_ij, delta)) +
#              0.5 * (einsum('il,jk -> ijkl', D_bar_ij, delta) + einsum('ik,jl -> ijkl', delta, D_bar_ij)))
#
#         return S_5_ijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab_n1, tn1, kappa, omega):

        I = self.update_variables(eps_Emab_n1, kappa, omega)

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        # if self.regularization:
        #    h = self.get_regularizing_length(sctx, eps_app_eng)
        #    self.phi_fn.h = h

#         #------------------------------------------------------------------
#         # Damage tensor (2th order):
#         #------------------------------------------------------------------
#         phi_Emab = self._get_phi_Emab(kappa_Emn)
#
#         #------------------------------------------------------------------
#         # Damage tensor (4th order) using product- or sum-type symmetrization:
#         #------------------------------------------------------------------
#         beta_Emabcd = self._get_beta_Emabcd(phi_Emab)
#
#         #------------------------------------------------------------------
#         # Damaged stiffness tensor calculated based on the damage tensor beta4:
#         #------------------------------------------------------------------
#         D_Emabcd, = einsum(
#             'Emijab, abef, Emabef -> Emijab', beta_Emabcd, self.D_abef, beta_Emabcd)

        D_Emabcd = self._get_S_1_Emabcd(eps_Emab_n1, kappa, omega)

        sig_Emab = einsum('Emabcd,Emcd -> Emab', D_Emabcd, eps_Emab_n1)

        return D_Emabcd, sig_Emab


class MATS2DMplDamageODF(MATSXDMicroplaneDamageODF, MATS2DEval):

    # implements(IMATSEval)

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

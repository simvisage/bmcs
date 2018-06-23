'''
Created on 15.02.2018

@author: abaktheer

Microplane damage model 2D - ODFS (Wu [2009])
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, einsum, identity, sqrt
from traits.api import \
    Constant, implements,\
    Float, Property, cached_property

import numpy as np
import traits.api as tr


class MATS3DMplDamageODF(MATS3DEval, MATS3D):

    implements(IMATSEval)

    epsilon_0 = Float(59.0e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    epsilon_f = Float(250.0e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    c_T = Float(1.0,
                label="a",
                desc="Lateral pressure coefficient",
                enter_set=True,
                auto_set=False)

    zeta_G = Float(1.0,
                   label="zeta_G",
                   desc="anisotropy parameter",
                   enter_set=True,
                   auto_set=False)

    E = tr.Float(34000.0,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    state_array_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''
    @cached_property
    def _get_state_array_shapes(self):
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
        delta = identity(3)
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

    def _get_e_N_arr_2(self, eps_Emab):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        return einsum('nij,Emij->Emn', self._MPNN, eps_Emab)

    def _get_e_t_vct_arr_2(self, eps_Emab):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        MPTT_ijr = self._get__MPTT()
        return einsum('nijr,Emij->Emnr', MPTT_ijr, eps_Emab)

    def _get_e_equiv_Emn(self, e_Emna):
        '''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        e_N_Emn = self._get_e_N_Emn(e_Emna)
        # print e_N_Emn[0, -1, :]
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_Emn = (np.abs(e_N_Emn) + e_N_Emn) / 2.0
        # normal strain vector for each microplane
        e_N_Emna = einsum('Emn,ni -> Emni', e_N_Emn, self._MPN)
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_Emna = e_Emna - e_N_Emna
        # squared tangential strain vector for each microplane
        e_TT_Emn = einsum('Emni,Emni -> Emn', e_T_Emna, e_T_Emna)
        # print e_TT_Emn[0, -1, :]
        # equivalent strain for each microplane
        e_equiv_Emn = sqrt(e_N_pos_Emn * e_N_pos_Emn + c_T * e_TT_Emn)
        return e_equiv_Emn

    def _get_state_variables(self, eps_Emab, kappa, omega):

        kappa_Emn = np.copy(kappa)
        omega_Emn = np.copy(omega)
        e_Emna = self._get_e_Emna(eps_Emab)
        eps_eq_Emn = self._get_e_equiv_Emn(e_Emna)
        f_trial_Emn = eps_eq_Emn - self.epsilon_0
        f_idx = np.where(f_trial_Emn > 0)
        kappa_Emn[f_idx] = eps_eq_Emn[f_idx]
        omega_Emn[f_idx] = self._get_omega(eps_eq_Emn[f_idx])

        return kappa_Emn, omega_Emn, f_idx

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
    def _get_S_1_Emabcd(self, eps_Emab, kappa, omega):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        phi_Emn = 1.0 - self._get_omega(kappa)
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
    def _get_S_4_Emabcd(self, eps_Emab, kappa, omega):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)
        I_vol_abcd = self._get_I_vol_abcd()
        I_dev_abcd = self._get_I_dev_abcd()

        delta = identity(3)
        phi_Emab = self._get_phi_Emab(kappa)

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
    def _get_S_5_Emabcd(self, eps_Emab, kappa, omega):

        K0 = self.E / (1.0 - 2.0 * self.nu)
        G0 = self.E / (1.0 + self.nu)

        delta = identity(3)
        phi_Emab = self._get_phi_Emab(kappa)

        # damaged stiffness without simplification
        S_5_Emabcd = (1.0 / 3.0) * (K0 - G0) * 0.5 * ((einsum('ij,Emkl -> Emijkl', delta, phi_Emab) +
                                                       einsum('Emij,kl -> Emijkl', phi_Emab, delta))) + \
            G0 * 0.5 * ((0.5 * (einsum('ik,Emjl -> Emijkl', delta, phi_Emab) + einsum('Emil,jk -> Emijkl', phi_Emab, delta)) +
                         0.5 * (einsum('Emik,jl -> ijkl', phi_Emab, delta) + einsum('il,Emjk  -> Emijkl', delta, phi_Emab))))

        return S_5_Emabcd

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, algorithmic, kappa, omega):

        if update_state:

            eps_Emab_n = eps_Emab_n1 - deps_Emab

            kappa_Emn, omega_Emn, f_idx = self._get_state_variables(
                eps_Emab_n, kappa, omega)

            kappa[...] = kappa_Emn[...]
            omega[...] = omega_Emn[...]

        kappa_Emn, omega_Emn, f_idx = self._get_state_variables(
            eps_Emab_n1, kappa, omega)

        D_Emabcd = self._get_S_4_Emabcd(eps_Emab_n1, kappa, omega)

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

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

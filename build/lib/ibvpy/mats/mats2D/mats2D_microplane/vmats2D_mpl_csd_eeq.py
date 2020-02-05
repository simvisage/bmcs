'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from traits.api import Constant, \
    Float, Dict, Property, cached_property

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
import numpy as np
import traits.api as tr


class MATS2DMplCSDEEQ(MATS2DEval):

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

    S_T = Float(0.00001,
                label="S",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = Float(1.2,
                label="r",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = Float(1.2,
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
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #-------------------------------------------
    Ad = Float(10000.0,
               label="a",
               desc="brittleness coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = Float(0.0002,
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

    state_var_shapes = Property(Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''
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
        delta = np.identity(2)
        D_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                  np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                  np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_abef

    @cached_property
    def _get_state_var_shapes(self):
        return {'w_N_Emn': (self.n_mp,),
                'z_N_Emn': (self.n_mp,),
                'alpha_N_Emn': (self.n_mp,),
                'r_N_Emn': (self.n_mp,),
                'eps_N_p_Emn': (self.n_mp,),
                'sigma_N_Emn': (self.n_mp,),
                'w_T_Emn': (self.n_mp,),
                'z_T_Emn': (self.n_mp,),
                'alpha_T_Emna': (self.n_mp, 2),
                'eps_T_pi_Emna': (self.n_mp, 2),
                }

    #--------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # (without cumulative normal strain for fatigue under tension)
    #--------------------------------------------------------------
    def get_normal_law(self, eps_N_Emn, w_N_Emn, z_N_Emn,
                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn):

        E_N = self.E / (1.0 - 2.0 * self.nu)

        pos = eps_N_Emn > 1e-6
        H = 1.0 * pos

        sigma_n_trial = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        Z = self.K_N * r_N_Emn
        X = self.gamma_N* alpha_N_Emn
        h = self.sigma_0+ Z
        pos_iso = h > 1e-6
        f_trial = abs(sigma_n_trial - X) - h * pos_iso

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn = eps_N_p_Emn + delta_lamda * np.sign(sigma_n_trial - X)
        r_N_Emn = r_N_Emn + delta_lamda
        alpha_N_Emn = alpha_N_Emn + delta_lamda * np.sign(sigma_n_trial - X)

        def Z_N(z_N_Emn): return 1.0 / self.Ad * (-z_N_Emn) / (1.0 + z_N_Emn)

        Y_N = 0.5 * H * E_N * eps_N_Emn ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0
        f = Y_N - (Y_0 + Z_N(z_N_Emn))

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        w_N_Emn = f_w(Y_N) * thres_2
        z_N_Emn = -w_N_Emn * thres_2


        sigma_N_Emn = (1.0 - H * w_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)

        return w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn


    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, w_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        E_T = self.E / (1.0 + self.nu)

        sig_pi_trial = E_T * (eps_T_Emna - eps_T_pi_Emna)
        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        norm_1 = np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))
        )

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_N_Emn / 3.0

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - w_T_Emn) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] = eps_T_pi_Emna[..., 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 0] - X[..., 0]) /
             (1.0 - w_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] = eps_T_pi_Emna[..., 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 1] - X[..., 1]) /
             (1.0 - w_T_Emn)) / norm_2

        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna)
            )

        w_T_Emn += ((1 - w_T_Emn) ** self.c_T) * \
            (delta_lamda * (Y / self.S_T) ** self.r_T) * \
            (self.tau_pi_bar / (self.tau_pi_bar - self.a * sigma_N_Emn / 3.0))

        alpha_T_Emna[..., 0] = alpha_T_Emna[..., 0] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] = alpha_T_Emna[..., 1] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        z_T_Emn = z_T_Emn + delta_lamda

        return w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna

#     #-------------------------------------------------------------------------
#     # MICROPLANE-Kinematic constraints
#     #-------------------------------------------------------------------------
#     def _get_e_Emna(self, eps_Emab):
#         # Projection of apparent strain onto the individual microplanes
#         e_ni = np.einsum('nb,Emba->Emna', self._MPN, eps_Emab)
#         return e_ni
# 
#     def _get_e_N_Emn(self, e_Emna):
#         # get the normal strain array for each microplane
# 
#         e_N_Emn = np.einsum('nij,...ij->...n', e_Emna, self._MPN)
#         return e_N_Emn
# 
#     def _get_e_T_Emna(self, e_Emna):
#         # get the tangential strain vector array for each microplane
#         MPTT_ijr = self._get__MPTT()
#         return np.einsum('nija,...ij->...na', MPTT_ijr, e_Emna)

    #-------------------------------------------------
    # Alternative methods for the kinematic constraint
    #-------------------------------------------------
    
    # get the operator of the microplane normals
    _MPNN = Property(depends_on='n_mp')
    
    @cached_property
    def _get__MPNN(self):
        MPNN_nij = np.einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')
    
    @cached_property
    def _get__MPTT(self):
        delta = np.identity(2)
        MPTT_nijr = 0.5 * (
            np.einsum('ni,jr -> nijr', self._MPN, delta) +
            np.einsum('nj,ir -> njir', self._MPN, delta) - 2 *
            np.einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr

    def _get_e_N_Emn_2(self, eps_Emab):
        # get the normal strain array for each microplane
        return np.einsum('nij,...ij->...n', self._MPNN, eps_Emab)

    def _get_e_T_Emnar_2(self, eps_Emab):
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self._get__MPTT()
        return np.einsum('nija,...ij->...na', MPTT_ijr, eps_Emab)

    #--------------------------------------------------------
    # return the state variables (Damage , inelastic strains)
    #--------------------------------------------------------
    def _get_state_variables(self, eps_Emab, tn1,
                             omegaN, z_N_Emn,
                             alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                             w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna):
 
        e_N_arr = self._get_e_N_Emn_2(eps_Emab)
        e_T_vct_arr = self._get_e_T_Emnar_2(eps_Emab)
 
        omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn = self.get_normal_law(e_N_arr,  omegaN, z_N_Emn,
                                                                                              alpha_N_Emn, r_N_Emn, eps_N_p_Emn)
 
        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna = self.get_tangential_law(e_T_vct_arr, w_T_Emn, z_T_Emn,
                                                                                alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)
 
        return omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna

    #-----------------------------------------------------------------
    # Returns a list of the plastic normal strain  for all microplanes.
    #-----------------------------------------------------------------
    def _get_eps_N_p_Emn(self, eps_Emab, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn):
        
        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        
        eps_N_p_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn)[4]
            
        return eps_N_p_Emn

    #----------------------------------------------------------------
    # Returns a list of the sliding strain vector for all microplanes.
    #----------------------------------------------------------------
    def _get_eps_T_pi_arr(self, eps_Emab, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)
        
        eps_N_T_pi_Emna = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)[3]
           
        return eps_N_T_pi_Emna

    #-------------------------------------------------------------
    # Returns a list of the integrity factors for all microplanes.
    #-------------------------------------------------------------
    def _get_phi_Emn(self, eps_Emab, w_N_Emn, z_N_Emn,
                     alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                     w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        w_N_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[0]
            
        w_T_Emn = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)[0]

        w_Emn = np.zeros_like(w_N_Emn)

        #w_Emn = np.maximum(w_N_Emn, w_T_Emn)

        w_Emn = w_T_Emn
        #print('w_N_Emn', w_N_Emn)
        #print('w_T_Emn', w_T_Emn)
        #print('w_Emn', w_Emn)

        phi_Emn = np.sqrt(1.0 - w_Emn)
        #print('phi_Emn', phi_Emn)

        return phi_Emn

    #----------------------------------------------
    # Returns the 2nd order damage tensor 'phi_mtx'
    #----------------------------------------------
    def _get_phi_Emab(self, eps_Emab, w_N_Emn, z_N_Emn,
                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        # scalar integrity factor for each microplane
        phi_Emn = self._get_phi_Emn(eps_Emab, w_N_Emn, z_N_Emn,
                                    alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                                    w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # integration terms for each microplanes
        phi_Emab = np.einsum('...n,n,nab->...ab', phi_Emn,
                             self._MPW, self._MPNN)

        return phi_Emab

    #----------------------------------------------------------------------
    # Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
    # (cf. [Jir99], Eq.(21))
    #----------------------------------------------------------------------
    def _get_beta_Emabcd(self, eps_Emab, w_N_Emn, z_N_Emn,
                         alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                         w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        delta = np.identity(2)

        phi_Emab = self._get_phi_Emab(eps_Emab, w_N_Emn, z_N_Emn,
                                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        beta_Emabcd = 0.25 * (
            np.einsum('...ik,jl->...ijkl', phi_Emab, delta) +
            np.einsum('...il,jk->...ijkl', phi_Emab, delta) +
            np.einsum('...jk,il->...ijkl', phi_Emab, delta) +
            np.einsum('...jl,ik->...ijkl', phi_Emab, delta)
        )

        return beta_Emabcd

    #---------------------------------------------------------------------
    # Extra homogenization of damage tensor in case of two damage parameters
    # Returns the 4th order damage tensor 'beta4' using (ref. [Baz99], Eq.(63))
    #---------------------------------------------------------------------

    def _get_beta_Emabcd_2(self, eps_Emab, w_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn, w_T_Emn, z_T_Emn,
             alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        # Returns the 4th order damage tensor 'beta4' using
        #(cf. [Baz99], Eq.(63))
        
        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)
        
        w_N_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[0]
            
        w_T_Emn = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)[0]

        delta = np.identity(2)
        beta_N = np.sqrt(1. - w_N_Emn)
        beta_T = np.sqrt(1. - w_T_Emn)

        #beta_N = 1. - w_N_Emn
        #beta_T = 1. - w_T_Emn

        #print(' w_N_Emn ',  w_N_Emn)
        #print(' w_N_Emn ', w_T_Emn)
        #print('beta_N ', beta_N)
        #print('beta_T ', beta_T)

        beta_ijkl = np.einsum('n, ...n,ni, nj, nk, nl -> ...ijkl', self._MPW, beta_N, self._MPN, self._MPN, self._MPN, self._MPN) + \
            0.25 * (np.einsum('n, ...n,ni, nk, jl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,ni, nl, jk -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nk, il -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nl, ik -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) -
                    4.0 * np.einsum('n, ...n, ni, nj, nk, nl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, self._MPN, self._MPN))

        return beta_ijkl
    #-----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    #-----------------------------------------------------------

    def _get_eps_p_Emab(self, eps_Emab, w_N_Emn, z_N_Emn,
                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                        w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        # plastic normal strains
        eps_N_p_Emn = self.get_normal_law(
            eps_N_Emn, w_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn)[4]

        # sliding tangential strains
        eps_T_pi_Emna = self.get_tangential_law(
            eps_T_Emna, w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)[3]
            
        print('eps_N_T_pi_Emna ' ,eps_T_pi_Emna ) 
         
        delta = np.identity(2)

        # 2-nd order plastic (inelastic) tensor
        eps_p_Emab = (
            np.einsum('n,...n,na,nb->...ab',
                      self._MPW, eps_N_p_Emn, self._MPN, self._MPN) +
            0.5 * (
                np.einsum('n,...nf,na,fb->...ab',
                          self._MPW, eps_T_pi_Emna, self._MPN, delta) +
                np.einsum('n,...nf,nb,fa->...ab', self._MPW,
                          eps_T_pi_Emna, self._MPN, delta)
            )
        )

        return eps_p_Emab

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab, t_n1, w_N_Emn, z_N_Emn,
                      alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                      w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna):

        # Corrector predictor computation.

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_Emabcd = self._get_beta_Emabcd(
            eps_Emab, w_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn
        )

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------

        D_Emabcd = np.einsum(
            '...ijab, abef, ...cdef->...ijcd', beta_Emabcd, self.D_abef, beta_Emabcd)
        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        # plastic strain tensor
        eps_p_Emab = self._get_eps_p_Emab(
            eps_Emab, w_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # elastic strain tensor
        eps_e_Emab = eps_Emab - eps_p_Emab

        # calculation of the stress tensor
        sig_Emab = np.einsum('...abcd,...cd->...ab', D_Emabcd, eps_e_Emab)

        return D_Emabcd, sig_Emab


# class MATS2DMplCSDEEQ(MATSXDMplCDSEEQ, MATS2DEval):

    # implements(IMATSEval)

    #-----------------------------------------------
    # number of microplanes 
    #-----------------------------------------------
    n_mp = Constant(360)

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        # microplane normals:
        alpha_list = np.linspace(0, 2 * np.pi, self.n_mp)

        MPN = np.array([[np.cos(alpha), np.sin(alpha)]
                        for alpha in alpha_list])

        return MPN

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        MPW = np.ones(self.n_mp) / self.n_mp * 2

        return MPW

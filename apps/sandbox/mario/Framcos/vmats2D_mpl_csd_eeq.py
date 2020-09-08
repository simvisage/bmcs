'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from traits.api import Constant, \
    Dict, Property, cached_property

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
import numpy as np
import traits.api as tr


class MATS2DMplCSDEEQ(MATS2DEval):

    concrete_type = tr.Int


    gamma_T = tr.Float(100000.,
                       label="Gamma",
                        desc=" Tangential Kinematic hardening modulus",
                        enter_set=True,
                        auto_set=False)

    K_T = tr.Float(10000.,
                label="K",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S_T = tr.Float(0.005,
                label="S",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = tr.Float(9.,
                label="r",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)
    e_T = tr.Float(12.,
                label="e",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = tr.Float(4.6,
                label="c",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = tr.Float(1.7,
                       label="Tau_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = tr.Float(0.003,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    # -------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # -------------------------------------------
    Ad = tr.Float(100.0,
               label="a",
               desc="brittleness coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = tr.Float(0.00008,
                  label="a",
                  desc="threshold strain",
                  enter_set=True,
                  auto_set=False)

    # -----------------------------------------------
    # Normal_Compression constitutive law parameters
    # -----------------------------------------------
    K_N = tr.Float(10000.,
                label="K_N",
                desc=" Normal isotropic harening",
                enter_set=True,
                auto_set=False)

    gamma_N = tr.Float(5000.,
                    label="gamma_N",
                    desc="Normal kinematic hardening",
                    enter_set=True,
                    auto_set=False)

    sigma_0 = tr.Float(30.,
                    label="sigma_0",
                    desc="Yielding stress",
                    enter_set=True,
                    auto_set=False)

    # -------------------------------------------------------------------------
    # Cached elasticity tensors
    # -------------------------------------------------------------------------

    E = tr.Float(35e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    def __init__(self, concrete_type):

        self.concrete_type = concrete_type

        if concrete_type == 0:      #     # C40 MA

            self.gamma_T = 100000

            self.K_T = 10000.0

            self.S_T = 0.005

            self.r_T = 9.

            self.e_T = 12.

            self.c_T = 4.6

            self.tau_pi_bar = 1.7

            self.a = 0.003,

            self.Ad = 100.0

            self.eps_0 = 0.00008

            self.K_N = 10000.

            self.gamma_N = 5000.

            self.sigma_0 = 30.

            #-------------------------------------------------------------------------
            # Cached elasticity tensors
            #-------------------------------------------------------------------------

            self.E = 35e+3

            self.nu = 0.2

        if concrete_type == 1:      #     #   C80 MA

            self.gamma_T = 1000000.

            self.K_T = 30000.0

            self.S_T = 0.01

            self.r_T = 14.

            self.c_T = 6.

            self.e_T = 14.

            self.tau_pi_bar = 2.0

            self.a = 0.01

            self.Ad = 1000.0

            self.eps_0 = 0.0001

            self.K_N = 30000.

            self.gamma_N = 20000.

            self.sigma_0 = 60.

            self.E = 42e+3

            self.nu = 0.2

        if concrete_type == 2:      #    PARAMETERS FOR C120 MA

            self.gamma_T = 2000000.

            self.K_T = 2200.0

            self.S_T = 0.015

            self.r_T = 17.5

            self.c_T = 8.

            self.e_T = 10.

            self.tau_pi_bar = 1.8

            self.a = 0.008

            self.Ad = 1000.0

            self.eps_0 = 0.0001

            self.K_N = 35000.

            self.gamma_N = 25000.

            self.sigma_0 = 90.

            self.E = 44e+3

            self.nu = 0.2

        if concrete_type == 3:      # TENSILE PARAMETRIC STUDY

            self.gamma_T = 80000.

            self.K_T = 10000.0

            self.S_T = 0.000000001

            self.r_T = 1.21

            self.e_T = 1.

            self.c_T = 1.85

            self.tau_pi_bar = 0.1

            self.a = 0.001

            self.Ad = 1500.0

            self.eps_0 = 0.00008

            self.K_N = 4000.

            self.gamma_N = 20000.

            self.sigma_0 = 180.

            self.E = 35e+3

            self.nu = 0.2

        # COMPRESSION
        if concrete_type == 4:

            self.gamma_T = 10000.

            self.K_T = 10000.0

            self.S_T = 0.000007

            self.r_T = 1.2

            self.e_T = 1.

            self.c_T = 1.25

            self.tau_pi_bar = 5.

            self.a = 0.001

            self.Ad = 1500.0

            self.eps_0 = 0.00008

            self.K_N = 10000.

            self.gamma_N = 10000.

            self.sigma_0 = 30.

            self.E = 35e+3

            self.nu = 0.2

        # BI-AXIAL ENVELOPE
        if concrete_type == 5:

            self.gamma_T = 10000.

            self.K_T = 10000.0

            self.S_T = 0.000007

            self.r_T = 1.2

            self.e_T = 1.

            self.c_T = 1.8

            self.tau_pi_bar = 5.

            self.a = 0.01

            self.Ad = 50000.0

            self.eps_0 = 0.00008

            self.K_N = 15000.

            self.gamma_N = 20000.

            self.sigma_0 = 30.

            self.E = 35e+3

            self.nu = 0.2

        if concrete_type == 6:  # # Paper 2D resdistribution

            self.gamma_T = 800000.

            self.K_T = 50000.0

            self.S_T = 0.029

            self.r_T = 13.

            self.e_T = 11.

            self.c_T = 8

            self.tau_pi_bar = 2.

            self.a = 0.012

            self.Ad = 1000.0

            self.eps_0 = 0.0001

            self.K_N = 80000.

            self.gamma_N = 100000.

            self.sigma_0 = 80.

            # -------------------------------------------------------------------------
            # Cached elasticity tensors
            # -------------------------------------------------------------------------

            self.E = 42e+3

            self.nu = 0.2

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
    def get_normal_law(self, eps_N_Emn, omega_N_Emn, z_N_Emn,
                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux):

        eps_N_Aux = self._get_e_N_Emn_2(eps_aux)

        E_N = self.E / (1.0 - 2.0 * self.nu)

        # When deciding if a microplane is in tensile or compression, we define a strain boundary such that that
        # sigmaN <= 0 if eps_N < 0, avoiding entering in the quadrant of compressive strains and traction

        sigma_trial = E_N * (eps_N_Emn - eps_N_p_Emn)
        pos1 = [(eps_N_Emn < -1e-6) & (sigma_trial > 1e-6)] # looking for microplanes violating strain boundary
        sigma_trial [pos1[0]] = 0
        pos = eps_N_Emn > 1e-6                            # microplanes under traction
        pos2 = eps_N_Emn < -1e-6                          # microplanes under compression
        H = 1.0 * pos
        H2 = 1.0 * pos2

        # thermo forces
        sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)
        sigma_N_Emn_tilde[pos1[0]] = 0                      # imposing strain boundary

        Z = self.K_N * z_N_Emn
        X = self.gamma_N * alpha_N_Emn * H2
        h = (self.sigma_0 + Z) * H2

        f_trial = (abs(sigma_N_Emn_tilde - X) - h) * H2

        # threshold plasticity

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N / (1 - omega_N_Emn) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn = eps_N_p_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)
        z_N_Emn = z_N_Emn + delta_lamda
        alpha_N_Emn = alpha_N_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)

        def R_N(r_N_Emn): return (1.0 / self.Ad) * (-r_N_Emn / (1.0 + r_N_Emn))

        Y_N = 0.5 * H * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0

        f = (Y_N - (Y_0 + R_N(r_N_Emn)))

        # threshold damage

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        omega_N_Emn[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        r_N_Emn[f > 1e-6] = -omega_N_Emn[f > 1e-6]

        sigma_N_Emn = (1.0 - H * omega_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        pos1 = [(eps_N_Emn < -1e-6) & (sigma_trial > 1e-6)] # looking for microplanes violating strain boundary
        sigma_N_Emn[pos1[0]] = 0


        Z = self.K_N * z_N_Emn
        X = self.gamma_N * alpha_N_Emn * H2

        return omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z, X, Y_N



    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        E_T = self.E / (1.0 + self.nu)

        # E_T = self.E * (1.0 - 4 * self.nu) / \
        #     ((1.0 + self.nu) * (1.0 - 2 * self.nu))

        # thermo forces

        sig_pi_trial = E_T * (eps_T_Emna - eps_T_pi_Emna)

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        norm_1 = np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))
        )
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))

        # threshold

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_N_Emn

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - omega_T_Emn) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] = eps_T_pi_Emna[..., 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 0] - X[..., 0]) /
             (1.0 - omega_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] = eps_T_pi_Emna[..., 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 1] - X[..., 1]) /
             (1.0 - omega_T_Emn)) / norm_2

        omega_T_Emn += ((1 - omega_T_Emn) ** self.c_T) * \
            (delta_lamda * (Y / self.S_T) ** self.r_T) * \
            (self.tau_pi_bar / (self.tau_pi_bar + self.a * sigma_N_Emn)) ** self.e_T

        alpha_T_Emna[..., 0] = alpha_T_Emna[..., 0] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] = alpha_T_Emna[..., 1] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        z_T_Emn = z_T_Emn + delta_lamda

        sigma_T_Emna = np.einsum(
            '...n,...na->...na', (1 - omega_T_Emn), E_T * (eps_T_Emna - eps_T_pi_Emna))

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))

        return omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z, X, Y

#     #-------------------------------------------------------------------------
#     # MICROPLANE-Kinematic constraints
#     #-------------------------------------------------------------------------

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
    def _get_state_variables(self, eps_Emab,
                             int_var, eps_aux):

        e_N_arr = self._get_e_N_Emn_2(eps_Emab)
        e_T_vct_arr = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn = int_var[:, 0]
        z_N_Emn = int_var[:, 1]
        alpha_N_Emn = int_var[:, 2]
        r_N_Emn = int_var[:, 3]
        eps_N_p_Emn = int_var[:, 4]
        sigma_N_Emn = int_var[:, 5]

        omega_T_Emn = int_var[:, 9]
        z_T_Emn = int_var[:, 10]
        alpha_T_Emna = int_var[:, 11:13]
        eps_T_pi_Emna = int_var[:, 13:15]


        omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(e_N_arr,  omega_N_Emn, z_N_Emn,
                                                                                                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(e_T_vct_arr, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4)consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        # 16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T

        int_var[:, 0] = omega_N_Emn
        int_var[:, 1] = z_N_Emn
        int_var[:, 2] = alpha_N_Emn
        int_var[:, 3] = r_N_Emn
        int_var[:, 4] = eps_N_p_Emn
        int_var[:, 5] = sigma_N_Emn
        int_var[:, 6] = Z_n
        int_var[:, 7] = X_n
        int_var[:, 8] = Y_n

        int_var[:, 9] = omega_T_Emn
        int_var[:, 10] = z_T_Emn
        int_var[:, 11:13] = alpha_T_Emna
        int_var[:, 13:15] = eps_T_pi_Emna
        int_var[:, 15:17] = sigma_T_Emna
        int_var[:, 17] = Z_T
        int_var[:, 18:20] = X_T
        int_var[:, 20] = Y_T


        return int_var

    #---------------------------------------------------------------------
    # Extra homogenization of damage tensor in case of two damage parameters
    # Returns the 4th order damage tensor 'beta4' using (ref. [Baz99], Eq.(63))
    #---------------------------------------------------------------------

    def _get_beta_Emabcd_2(self, eps_Emab, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, eps_aux):

        # Returns the 4th order damage tensor 'beta4' using
        #(cf. [Baz99], Eq.(63))

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(
            eps_N_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(
            eps_T_Emna, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        delta = np.identity(2)
        beta_N = np.sqrt(1. - omega_N_Emn)
        beta_T = np.sqrt(1. - omega_T_Emn)

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

    def _get_eps_p_Emab(self, eps_Emab, omega_N_Emn, z_N_Emn,
                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        # plastic normal strains
        omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(
            eps_N_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        # sliding tangential strains
        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(
            eps_T_Emna, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

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

    def get_corr_pred(self, eps_Emab, t_n1, int_var, eps_aux, F):


        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4)consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        # 16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T


        # Corrector predictor computation.

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------


        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn = int_var[:, 0]
        z_N_Emn = int_var[:, 1]
        alpha_N_Emn = int_var[:, 2]
        r_N_Emn = int_var[:, 3]
        eps_N_p_Emn = int_var[:, 4]
        sigma_N_Emn = int_var[:, 5]

        omega_T_Emn = int_var[:, 9]
        z_T_Emn = int_var[:, 10]
        alpha_T_Emna = int_var[:, 11:13]
        eps_T_pi_Emna = int_var[:, 13:15]


        beta_Emabcd = self._get_beta_Emabcd_2(
            eps_Emab, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, eps_aux
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
            eps_Emab, omega_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux)

        # elastic strain tensor
        eps_e_Emab = eps_Emab - eps_p_Emab

        # calculation of the stress tensor
        sig_Emab = np.einsum('...abcd,...cd->...ab', D_Emabcd, eps_e_Emab)

        return D_Emabcd, sig_Emab, eps_p_Emab

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
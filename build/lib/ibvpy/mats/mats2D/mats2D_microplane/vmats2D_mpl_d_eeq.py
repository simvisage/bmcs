'''
Created on 29.03.2017

@author: abaktheer

Microplane damage model 2D - Jirasek [1999]
'''

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from numpy import \
    array, einsum, identity, sqrt
from simulator.i_model import IModel
from traits.api import \
    Constant, provides, \
    Float, Property, cached_property

from ibvpy.mats.matsXD.vmatsXD_eval import MATSXDEval
import numpy as np
import traits.api as tr


@provides(IModel)
class MATSXDMicroplaneDamageEEQ(MATSXDEval):

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

    state_var_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''

    @cached_property
    def _get_state_var_shapes(self):
        return dict(kappa=(self.n_mp,),
                    omega=(self.n_mp,))

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
        phi_Emn = np.sqrt(1.0 - self._get_omega(kappa_Emn))
        # integration terms for each microplanes
        phi_Emab = einsum('Emn,n,nab->Emab', phi_Emn, self._MPW, self._MPNN)
        return phi_Emab

    def _get_beta_Emabcd(self, phi_Emab):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        delta = identity(2)
        beta_Emijkl = 0.25 * (einsum('Emik,jl->Emijkl', phi_Emab, delta) +
                              einsum('Emil,jk->Emijkl', phi_Emab, delta) +
                              einsum('Emjk,il->Emijkl', phi_Emab, delta) +
                              einsum('Emjl,ik->Emijkl', phi_Emab, delta))

        return beta_Emijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_Emab, tn1, kappa, omega):

        self.update_state_variables(eps_Emab, kappa, omega)

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
        phi_Emab = self._get_phi_Emab(kappa)

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_Emabcd = self._get_beta_Emabcd(phi_Emab)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D_Emijab = einsum(
            'Emijab, abef, Emcdef -> Emijcd',
            beta_Emabcd, self.D_abef, beta_Emabcd
        )

        sig_Emab = einsum('Emabef,Emef -> Emab', D_Emijab, eps_Emab)

        return D_Emijab, sig_Emab


class MATS2DMplDamageEEQ(MATSXDMicroplaneDamageEEQ, MATS2DEval):
    '''Number of microplanes - currently fixed for 3D
    '''
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

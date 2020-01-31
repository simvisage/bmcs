'''
Created on 15.02.2018

@author: abaktheer

Microplane damage model 2D - Jirasek [1999]
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from numpy import \
    array, einsum, identity, sqrt
from simulator.api import IModel
from simulator.api import \
    TLoopImplicit, TStepBC
from traits.api import \
    Constant, provides, \
    Float, Property, cached_property

import numpy as np
import traits.api as tr


@provides(IModel)
class MATS3DMplDamageEEQ(MATS3DEval):
    # To use the model directly in the simulator specify the
    # time stepping classes
    tloop_type = TLoopImplicit
    tstep_type = TStepBC

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

    c_T = Float(0.01,
                label="a",
                desc="Lateral pressure coefficient",
                enter_set=True,
                auto_set=False)

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    state_var_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''
    @cached_property
    def _get_state_var_shapes(self):
        return {'kappa_n': (self.n_mp,),
                'omega_n': (self.n_mp,)}

    U_var_shape = (6,)
    '''Shape of the primary variable required by the TStepState.
    '''

    node_name = 'Desmorat model'

    tree_node_list = tr.List([])

    #=========================================================================
    # Evaluation - get the corrector and predictor
    #=========================================================================

    def get_corr_pred(self, eps_ab, tn1, kappa_n, omega_n):

        self._update_state_variables(eps_ab, kappa_n, omega_n)
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
        phi_ab = self._get_phi_ab(kappa_n)

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_abcd = self._get_beta_abcd(phi_ab)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D_ijab = einsum(
            '...ijab, abef, ...cdef -> ...ijcd',
            beta_abcd, self.D_abef, beta_abcd
        )

        sig_ab = einsum(
            '...abef,...ef -> ...ab',
            D_ijab, eps_ab
        )

        return sig_ab, D_ijab

    #=========================================================================
    # MICROPLANE-Kinematic constraints
    #=========================================================================

    _MPNN = Property(depends_on='n_mp')
    r'''Get the dyadic product of the microplane normals
    '''
    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals

        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    _MPTT = Property(depends_on='n_mp')
    r'''Get the third order tangential tensor (operator) for each microplane
    '''
    @cached_property
    def _get__MPTT(self):
        # Third order tangential tensor for each microplane
        delta = identity(3)
        MPTT_nijr = 0.5 * (
            einsum('ni,jr -> nijr', self._MPN, delta) +
            einsum('nj,ir -> njir', self._MPN, delta) - 2.0 *
            einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr

    def _get_e_na(self, eps_ab):
        r'''
        Projection of apparent strain onto the individual microplanes
        '''
        e_ni = einsum(
            'nb,...ba->...na',
            self._MPN, eps_ab
        )
        return e_ni

    def _get_e_N_n(self, e_na):
        r'''
        Get the normal strain array for each microplane
        '''
        e_N_n = einsum(
            '...na, na->...n',
            e_na, self._MPN
        )
        return e_N_n

    def _get_e_equiv_n(self, e_na):
        r'''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        e_N_n = self._get_e_N_n(e_na)
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_n = (np.abs(e_N_n) + e_N_n) / 2.0
        # normal strain vector for each microplane
        e_N_na = einsum('...n,ni -> ...ni', e_N_n, self._MPN)
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_na = e_na - e_N_na
        # squared tangential strain vector for each microplane
        e_TT_n = einsum('...ni,...ni -> ...n', e_T_na, e_T_na)
        # equivalent strain for each microplane
        e_equiv_n = sqrt(e_N_pos_n * e_N_pos_n + c_T * e_TT_n)
        return e_equiv_n

    def _update_state_variables(self, eps_ab, kappa_n, omega_n):
        e_na = self._get_e_na(eps_ab)
        eps_eq_n = self._get_e_equiv_n(e_na)
        f_trial_n = eps_eq_n - self.epsilon_0
        I = np.where(f_trial_n > 0)
        k_n = np.max(np.array([kappa_n[I], eps_eq_n[I]]), axis=0)
        kappa_n[I] = k_n
        omega_n[I] = self._get_omega(k_n)

    def _get_omega(self, kappa_n):
        '''
        Return new value of damage parameter
        @param kappa:
        '''
        omega_n = np.zeros_like(kappa_n)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        I = np.where(kappa_n >= epsilon_0)
        omega_n[I] = (
            1.0 - (epsilon_0 / kappa_n[I] *
                   np.exp(-1.0 * (kappa_n[I] - epsilon_0) /
                          (epsilon_f - epsilon_0))
                   ))
        return omega_n

    def _get_phi_ab(self, kappa_n):
        # Returns the 2nd order damage tensor 'phi_mtx'
        # scalar integrity factor for each microplane
        phi_n = np.sqrt(1.0 - self._get_omega(kappa_n))
        # print 'phi_Emn', phi_Emn[:, -1, :]
        # integration terms for each microplanes
        phi_ab = einsum('...n,n,nab->...ab', phi_n, self._MPW, self._MPNN)
        return phi_ab

    def _get_beta_abcd(self, phi_ab):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        delta = identity(3)
        beta_ijkl = 0.25 * (
            einsum('...ik,jl->...ijkl', phi_ab, delta) +
            einsum('...il,jk->...ijkl', phi_ab, delta) +
            einsum('...jk,il->...ijkl', phi_ab, delta) +
            einsum('...jl,ik->...ijkl', phi_ab, delta)
        )
        return beta_ijkl

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

    def _get_var_dict(self):
        var_dict = super(MATS3DMplDamageEEQ, self)._get_var_dict()
        var_dict.update(
            phi_ab=self.get_phi_ab
        )
        return var_dict

    def get_phi_ab(self, eps_ab, tn1, kappa_n, omega_n):
        return self._get_phi_ab(kappa_n)


if __name__ == '__main__':

    mm = MATS3DMplDamageEEQ()
    print(mm.var_dict)
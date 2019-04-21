'''
Created on 30.10.2018

@author: abaktheer
'''
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from simulator.api import \
    TLoopImplicit, TStepBC
from traits.api import \
    Float, List
from traitsui.api import View, VGroup, Item

import numpy as np
import traits.api as tr


class MATS3DDesmorat(MATS3DEval):
    '''Damage - plasticity model by Desmorat.
    '''
    # To use the model directly in the simulator specify the
    # time stepping classes
    tloop_type = TLoopImplicit
    tstep_type = TStepBC

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    E_1 = tr.Float(16.0e+3,
                   label="E_1",
                   desc="first Young's Modulus",
                   auto_set=False,
                   input=True)
    E_2 = tr.Float(19.0e+3,
                   label="E_2",
                   desc="second Young's Modulus",
                   auto_set=False,
                   input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poisson ratio",
                  auto_set=False,
                  input=True)

    gamma = Float(110.0,
                  label="Gamma",
                  desc="kinematic hardening modulus",
                  MAT=True,
                  symbol=r'\gamma',
                  unit='MPa/mm',
                  enter_set=True,
                  auto_set=False)

    K = Float(130.0,
              label="K",
              desc="isotropic hardening modulus",
              MAT=True,
              symbol='K',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    S = Float(476.0e-6,
              label="S",
              desc="damage strength",
              MAT=True,
              symbol='S',
              unit='MPa/mm',
              enter_set=True,
              auto_set=False)

    tau_bar = Float(6.0,
                    label="Tau_0 ",
                    desc="yield stress",
                    symbol=r'\bar{\tau}',
                    unit='MPa',
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    U_var_shape = (6,)
    '''Shape of the primary variable required by the TStepState.
    '''

    state_var_shapes = {'sigma_ab': (3, 3),
                        'sigma_pi_ab': (3, 3),
                        'eps_pi_ab': (3, 3),
                        'alpha_ab': (3, 3),
                        'z_a': (),
                        'omega_a': ()}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    node_name = 'Desmorat model'

    tree_node_list = List([])

    def _get_lame_1_params(self):
        la = self.E_1 * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_1 / (2. + 2. * self.nu)
        return la, mu

    D_1_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_1_abef(self):
        la, mu = self._get_lame_1_params()
        delta = np.identity(3)
        D_1_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                    np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                    np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_1_abef

    def _get_lame_2_params(self):
        la = self.E_2 * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_2 / (2. + 2. * self.nu)
        return la, mu

    D_2_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_2_abef(self):
        la = self._get_lame_2_params()[0]
        mu = self._get_lame_2_params()[1]
        delta = np.identity(3)
        D_2_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                    np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                    np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_2_abef

    def get_corr_pred(self, eps_ab, tn1,
                      sigma_ab, sigma_pi_ab, eps_pi_ab,
                      alpha_ab, z_a, omega_a):
        r'''
        Corrector predictor computation.
        '''
        D_1_abef = self.D_1_abef
        D_2_abef = self.D_2_abef
        sigma_pi_ab_trial = np.einsum(
            '...ijkl,...kl->...ij',
            D_2_abef, eps_ab - eps_pi_ab
        )
        a = sigma_pi_ab_trial - self.gamma * alpha_ab
        norm_a = np.sqrt(np.einsum(
            '...ij,...ij',
            a, a)
        )
        f = norm_a - self.tau_bar - self.K * z_a

        # identify the inelastic material points to perform return mapping
        I = np.where(f > 1e-6)
        delta_pi_I = (
            f[I] /
            (self.E_2 + (self.K + self.gamma) * (1. - omega_a[I]))
        )
        b_I = norm_a[I]
        return_ab_I = np.einsum(
            '...ij,...->...ij',
            a[I], delta_pi_I / b_I
        )
        eps_pi_ab[I] += return_ab_I
        eps_diff_ab_I = eps_ab[I] - eps_pi_ab[I]
        Y_a_I = 0.5 * (
            np.einsum(
                '...ij,...ijkl,...kl',
                eps_ab[I], D_1_abef, eps_ab[I]
            )
            +
            np.einsum(
                '...ij,...ijkl,...kl',
                eps_diff_ab_I, D_2_abef, eps_diff_ab_I
            )
        )
        omega_a[I] += (Y_a_I / self.S) * delta_pi_I
        omega_a[I][np.where(omega_a[I] >= 0.99)] = 0.99
        alpha_ab[I] += np.einsum(
            '...ij,...->...ij',
            return_ab_I, (1.0 - omega_a[I])
        )
        z_a[I] += delta_pi_I * (1.0 - omega_a[I])

        # evaluate the material stress and stiffness tensors
        phi_n = 1.0 - omega_a
        # this is a side effect - recording a returned value
        # simultaneously as a state variable - not ideal! dangerous.
        sigma_ab[...] = (
            np.einsum(
                '...,...ijkl,...kl->...ij',
                phi_n, D_1_abef, eps_ab
            ) +
            np.einsum(
                '...,...ijkl,...kl->...ij',
                phi_n, D_2_abef, eps_ab - eps_pi_ab
            )
        )
        # secant stiffness matrix
        D_abef = np.einsum(
            '...,...ijkl->...ijkl',
            phi_n, D_1_abef + D_2_abef
        )
        return sigma_ab, D_abef

    traits_view = View(
        VGroup(
            Item('E_1', full_size=True, resizable=True),
            Item('E_2'),
            Item('nu'),
            label='Elastic parameters'
        ),
        VGroup(
            Item('gamma', full_size=True, resizable=True),
            Item('K'),
            Item("S"),
            Item("tau_bar"),
            label='Inelastic parameters'
        )
    )
    tree_view = traits_view


'''
Created on 30.10.2018

@author: abaktheer
'''
from traits.api import \
    Float, List
from traitsui.api import View, VGroup, Item

from bmcs.simulator import \
    Model, TLoopImplicit, TStepBC
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import \
    MATS3D
import numpy as np
import traits.api as tr


class MATS3DDesmorat(Model, MATS3DEval, MATS3D):
    '''Damage - plasticity model by Desmorat.
    '''
    # To use the model directly in the simulator specify the
    # time stepping classes
    tloop_type = TLoopImplicit
    tstep_type = TStepBC

    U_var_shape = (3, 3)
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

    def _get_lame_1_params(self):
        la = self.E_1 * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_1 / (2. + 2. * self.nu)
        return la, mu

    D_1_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_1_abef(self):
        la = self._get_lame_1_params()[0]
        mu = self._get_lame_1_params()[1]
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

    tree_node_list = List([])

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

    def _get_state_variables(self, eps_ab, eps_pi_ab, alpha_ab, z_a, omega_a):

        D_1_abef = self.D_1_abef
        D_2_abef = self.D_2_abef

        sigma_pi_ab_trial = (
            np.einsum('...ijkl,...kl->...ij', D_2_abef, eps_ab - eps_pi_ab))

        #print('eps_ab', eps_ab)
        #print('D_2_abef', D_2_abef)
        #print('sigma_pi_ab_trial', sigma_pi_ab_trial)

        a = sigma_pi_ab_trial - self.gamma * alpha_ab
        #print('a', a)

        norm_a = np.sqrt(np.einsum('...ij,...ij', a, a))
        #print('norm_a', norm_a)

        n = a / norm_a

        f = np.sqrt(np.einsum('...ij,...ij', a, a)
                    ) - self.tau_bar - self.K * z_a

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_pi = f / \
            (self.E_2 + (self.K + self.gamma) * (1. - omega_a)) * plas_1

        b = 1.0 * elas_1 + norm_a * plas_1

#         delta_lamda = f / (np.einsum('...ij,...ijkl,...kl',
#                                      n, D_2_abef, n) + self.gamma * np.einsum('...ij,...ij', n, n) + self.K)
#
#         delta_pi = delta_lamda / (1.0 - omega_a)

        eps_pi_ab = eps_pi_ab + (a * delta_pi / b)
        print('eps_ab', eps_ab[..., 0, 0])
        print('eps_pi_ab', eps_pi_ab[..., 0, 0])

        eps_diff_ab = eps_ab - eps_pi_ab

        Y_a = 0.5 * (
            (
                np.einsum('...ij,...ijkl,...kl',
                          eps_ab, D_1_abef, eps_ab)
            ) +
            0.5 * (
                np.einsum('...ij,...ijkl,...kl',
                          eps_diff_ab, D_2_abef, eps_diff_ab)
            )
        )
        omega_a = omega_a + (Y_a / self.S) * delta_pi

        if omega_a >= 0.99:
            omega_a = 0.99

        alpha_ab = alpha_ab + plas_1 * \
            (a * delta_pi / b) * (1.0 - omega_a)

        z_a = z_a + delta_pi * (1.0 - omega_a)

        return eps_pi_ab, alpha_ab, z_a, omega_a

    def get_corr_pred(self, eps_ab_k, tn1,
                      sigma_ab_n, sigma_pi_ab_n, eps_pi_ab_n,
                      alpha_ab_n, z_a_n, omega_a_n):
        r'''
        Corrector predictor computation.
        '''
        Em_len = len(eps_ab_k.shape) - 2
        new_shape = tuple([1 for i in range(Em_len)]) + self.D_abef.shape
        D_1_abef = self.D_1_abef.reshape(*new_shape)
        D_2_abef = self.D_2_abef.reshape(*new_shape)

        eps_pi_ab_k, alpha_ab_k, z_a_k, omega_a_k = self._get_state_variables(
            eps_ab_k, eps_pi_ab_n, alpha_ab_n, z_a_n, omega_a_n
        )

        phi_n = 1.0 - omega_a_k

        sigma_ab = phi_n * (np.einsum('...ijkl,...kl->...ij',
                                      D_1_abef, eps_ab_k) +
                            np.einsum('...ijkl,...kl->...ij',
                                      D_2_abef, eps_ab_k - eps_pi_ab_k))

        D_abef = phi_n * np.einsum(' ...ijkl->...ijkl',
                                   D_1_abef + D_2_abef)

        return D_abef, sigma_ab

    def update_state(self, eps_ab_n1, tn1,
                     sigma_ab_n, sigma_pi_ab_n, eps_pi_ab_n,
                     alpha_ab_n, z_a_n, omega_a_n):
        eps_ab_n1[...], eps_pi_ab_n[...], alpha_ab_n[...], z_a_n[...], omega_a_n[...] = \
            self._get_state_variables(
            eps_ab_n1, eps_pi_ab_n, alpha_ab_n, z_a_n, omega_a_n
        )

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
            Item("Tau_0 "),
            label='Inelastic parameters'
        )
    )
    tree_view = traits_view

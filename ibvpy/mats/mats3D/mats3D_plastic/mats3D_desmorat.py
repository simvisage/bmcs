'''
Created on 30.10.2018

@author: aguilar
'''
from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
from ibvpy.mats.mats3D.mats3D_eval import \
    MATS3DEval
from ibvpy.mats.mats3D.vmats3D_eval import MATS3D
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements,  \
    Constant, Float, WeakRef, List, Str, Property, cached_property, \
    Trait, on_trait_change, Instance, Callable
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

import numpy as np
import traits.api as tr


class MATS3DDesmorat(MATS3DEval, MATS3D):

    node_name = 'bond model: damage-plasticity'

    '''Damage - plasticity model of bond.
    '''

    tr.implements(IMATSEval)

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
        la = self.E_2 * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E_2 / (2. + 2. * self.nu)
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

    state_array_shapes = {'sigma_Emab': (3, 3),
                          'sigma_pi_Emab': (3, 3),
                          'eps_pi_Emab': (3, 3),
                          'alpha_Emab': (3, 3),
                          'z_Ema': (),
                          'omega_Ema': ()}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    def _get_state_variables(self, eps_Emab, eps_pi_Emab, alpha_Emab, z_Ema, omega_Ema):

        D_1_abef = self.D_1_abef
        D_2_abef = self.D_2_abef

        sigma_pi_Emab_trial = (
            np.einsum('...ijkl,...kl->...ij', D_2_abef, eps_Emab))

        norm = sigma_pi_Emab_trial - self.gamma * alpha_Emab,
        f = np.sqrt(np.einsum('Emnj,Emnj', norm, norm)
                    ) - self.tau_bar - self.K * z_Ema

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

#         delta_pi = f / \
#             (self.E_2 + (self.K + self.gamma) * (1. - omega_Ema)) * plas_1
#
        norm3 = 1.0 * elas_1 + \
            np.sqrt(np.einsum('...ab,...ab', norm, norm)) * plas_1

        delta_lamda = f / (np.einsum('...ij,...ijkl,...kl',
                                     norm, D_1_abef, norm) + self.gamma * np.einsum('...ab,...ab', norm, norm) + self.K)

        delta_pi = delta_lamda / (1.0 - omega_Ema)

        eps_pi_Emab = eps_pi_Emab + plas_1 * (norm * delta_pi / norm3)

        eps_diff_Emab = eps_Emab - eps_pi_Emab

        Y_Ema = 0.5 * (np.einsum('...ij,...ijkl,...kl',
                                 eps_Emab, D_1_abef, eps_Emab)) + \
            0.5 * (np.einsum('...ij,...ijkl,...kl',
                             eps_diff_Emab, D_2_abef, eps_diff_Emab))

        omega_Ema = omega_Ema + (Y_Ema / self.S) * delta_pi

        if omega_Ema >= 0.99:
            omega_Ema = 0.99

        alpha_Emab = alpha_Emab + plas_1 * \
            (norm * delta_pi / norm3) * (1.0 - omega_Ema)

        z_Ema = z_Ema + delta_pi

        return eps_pi_Emab, alpha_Emab, z_Ema, omega_Ema

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, algorithmic,
                      sigma_Emab, sigma_pi_Emab, eps_pi_Emab,
                      alpha_Emab, z_Ema, omega_Ema):
        r'''
        Corrector predictor computation.
        '''
        if update_state:
            eps_Emab_n = eps_Emab_n1 - deps_Emab

            Em_len = len(eps_Emab_n1.shape) - 2
            new_shape = tuple([1 for i in range(Em_len)]) + self.D_abef.shape
            D_1_abef = self.D_1_abef.reshape(*new_shape)
            D_2_abef = self.D_2_abef.reshape(*new_shape)

            eps_pi_Emab, alpha_Emab, z_Ema, omega_Ema = self._get_state_variables(
                eps_Emab_n, eps_pi_Emab, alpha_Emab, z_Ema, omega_Ema)

            phi_Emn = 1.0 - omega_Ema

            sigma_Emab = phi_Emn * (np.einsum('...ijkl,...kl->...ij',
                                              D_1_abef, eps_Emab_n) +
                                    np.einsum('...ijkl,...kl->...ij',
                                              D_2_abef, eps_Emab_n - eps_pi_Emab))

            D_abef = phi_Emn * np.einsum(' ...ijkl->...ijkl',
                                         D_1_abef + D_2_abef)

        return D_abef, sigma_Emab

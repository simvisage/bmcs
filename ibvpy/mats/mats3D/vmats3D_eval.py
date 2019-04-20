'''
Created on Feb 8, 2018

@author: rch
'''

import copy

from view.ui import BMCSTreeNode

import numpy as np
import traits.api as tr


class MATS3D(BMCSTreeNode):

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
        delta = np.identity(3)
        D_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                  np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                  np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_abef

    state_var_shapes = tr.Property(tr.Dict())

    def _get_state_var_shapes(self):
        raise NotImplementedError

    var_dict = tr.Property()

    def _get_var_dict(self):
        return dict(eps_ab=self.get_eps_ab,
                    sig_ab=self.get_sig_ab)

    def get_eps_ab(self, eps_ab, tn1, **state):
        return eps_ab

    def get_sig_ab(self, eps_ab, tn1, **state):
        state_copy = copy.deepcopy(state)
        sig_ab, _ = self.get_corr_pred(
            eps_ab, tn1, **state_copy
        )
        return sig_ab

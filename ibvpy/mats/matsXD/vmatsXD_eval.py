
import copy

from ibvpy.mats.mats_eval import \
    MATSEval
from traits.api import \
    Constant, Float, Property, cached_property, Array,\
    Dict, Str, Callable

import numpy as np
import traitsui.api as ui


class MATSXDEval(MATSEval):
    '''Base class for elastic models.
    '''

    n_dims = Constant(Float)
    '''Number of spatial dimensions of an integration 
    cell for the material model
    '''

    E = Float(34e+3,
              label="E",
              desc="Young's Modulus",
              auto_set=False,
              input=True)

    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio",
               auto_set=False,
               input=True)

    def _get_lame_params(self):
        # First Lame parameter (bulk modulus)
        la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2. + 2. * self.nu)
        return la, mu

    D_abef = Property(Array, depends_on='+input')
    '''Material stiffness - rank 4 tensor
    '''
    @cached_property
    def _get_D_abef(self):
        la, mu = self._get_lame_params()
        delta = np.identity(self.n_dims)
        return (
            np.einsum(',ij,kl->ijkl', la, delta, delta) +
            np.einsum(',ik,jl->ijkl', mu, delta, delta) +
            np.einsum(',il,jk->ijkl', mu, delta, delta)
        )

    def get_corr_pred(self, eps_Emab, tn1):
        '''
        Corrector predictor computation.
        @param eps_Emab input variable - strain tensor
        '''
        sigma_Emab = np.einsum(
            'abcd,...cd->...ab', self.D_abef, eps_Emab
        )
        Em_len = len(eps_Emab.shape) - 2
        new_shape = tuple([1 for _ in range(Em_len)]) + self.D_abef.shape
        D_abef = self.D_abef.reshape(*new_shape)
        return sigma_Emab, D_abef

    #=========================================================================
    # Response variables
    #=========================================================================
    def get_eps_ab(self, eps_ab, tn1, **state):
        return eps_ab

    def get_sig_ab(self, eps_ab, tn1, **state):
        return self.get_sig(eps_ab, tn1, **state)

    var_dict = Property(Dict(Str, Callable))
    '''Dictionary of response variables
    '''
    @cached_property
    def _get_var_dict(self):
        return dict(eps_ab=self.get_eps_ab,
                    sig_ab=self.get_sig_ab)

    def x_var_dict_default(self):
        return {'sig_app': self.get_sig_ab,
                'eps_app': self.get_eps_ab,
                'max_principle_sig': self.get_max_principle_sig,
                'strain_energy': self.get_strain_energy}

    view_traits = ui.View(
        ui.VSplit(
            ui.Group(
                ui.Item('E'),
                ui.Item('nu'),),
        ),
        resizable=True
    )

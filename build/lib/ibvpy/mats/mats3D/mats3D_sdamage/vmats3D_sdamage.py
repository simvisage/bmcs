
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import \
    EitherType

import numpy as np
import traits.api as tr

from .vstrain_norm2d import Rankine


# from strain_norm3d import Energy, Euclidean, Mises, Rankine, Mazars, \
#     IStrainNorm3D
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS3DScalarDamage(MATS3DEval):
    r'''
    Isotropic damage model.
    '''
    node_name = 'Scalar damage'

    stiffness = tr.Enum("secant", "algorithmic",
                        input=True)
    r'''Selector of the stiffness calculation.
    '''
    strain_norm = EitherType(klasses=[Rankine,
                                      ], input=True)
    r'''Selector of the strain norm defining the load surface.
    '''

    epsilon_0 = tr.Float(5e-2,
                         label="eps_0",
                         desc="Strain at the onset of damage",
                         auto_set=False,
                         input=True)
    r'''Damage function parameter - slope of the damage function.
    '''

    epsilon_f = tr.Float(191e-1,
                         label="eps_f",
                         desc="Slope of the damage function",
                         auto_set=False,
                         input=True)
    r'''Damage function parameter - slope of the damage function.
    '''

    changed = tr.Event
    r'''This event can be used by the clients to trigger 
    an action upon the completed reconfiguration of 
    the material model
    '''

    state_var_shapes = {'kappa': (),
                        'omega': ()}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    def init(self, kappa, omega):
        r'''
        Initialize the state variables.
        '''
        kappa[...] = 0
        omega[...] = 0

    def get_corr_pred(self, eps_Emab_n1, tn1, kappa, omega):
        r'''
        Corrector predictor computation.
        '''
        I = self.update_state_variables(eps_Emab_n1, kappa, omega)
        phi_Em = (1.0 - omega)
        D_Emabcd = np.einsum(
            '...,abcd->...abcd',
            phi_Em, self.D_abef
        )
        sigma_Emab = np.einsum(
            '...abcd,...cd->...ab',
            D_Emabcd, eps_Emab_n1
        )

        # algorithmic switched off - because the derivative
        # of the strain norm is still not available
        if False:  # algorithmic:
            D_Emabcd_red_I = self._get_D_abcd_alg_reduction(
                kappa[I], eps_Emab_n1[I])
            D_Emabcd[I] -= D_Emabcd_red_I

        return sigma_Emab, D_Emabcd

    def update_state_variables(self, eps_Emab, kappa, omega):
        eps_eq_Em = self.strain_norm.get_eps_eq(eps_Emab, kappa)
        f_trial_Em = eps_eq_Em - self.epsilon_0
        I = np.where(f_trial_Em > 0)
        kappa[I] = eps_eq_Em[I]
        omega[I] = self._get_omega(eps_eq_Em[I])
        return I

    def _get_omega(self, kappa_Em):
        r'''
        Return new value of damage parameter
        @param kappa_Em: maximum strain norm achieved so far 
        '''
        omega_Em = np.zeros_like(kappa_Em)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        I = np.where(kappa_Em >= epsilon_0)
        omega_Em[I] = (
            1.0 - (epsilon_0 / kappa_Em[I] *
                   np.exp(-1.0 * (kappa_Em[I] - epsilon_0) /
                          (epsilon_f - epsilon_0))
                   ))
        return omega_Em

    def _get_domega(self, kappa_Em):
        '''
        Return new value of damage parameter derivative
        @param kappa_Em: maximum strain norm achieved so far
        '''
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        domega_Em = np.zeros_like(kappa_Em)
        I = np.where(kappa_Em >= epsilon_0)
        factor_1 = epsilon_0 / (kappa_Em[I] * kappa_Em[I])
        factor_2 = epsilon_0 / (kappa_Em[I] * (epsilon_f - epsilon_0))
        domega_Em[I] = (
            (factor_1 + factor_2) * np.exp(-(kappa_Em[I] - epsilon_0) /
                                           (epsilon_f - epsilon_0))
        )
        return domega_Em

    def _get_D_abcd_alg_reduction(self, kappa_Em, eps_Emab_n1):
        '''Calculate the stiffness term to be subtracted
        from the secant stiffness to get the algorithmic stiffness.
        '''
        domega_Em = self._get_domega(kappa_Em)
        deps_eq_Emcd = self.strain_norm.get_deps_eq(eps_Emab_n1)
        return np.einsum('...,...cd,abcd,...cd->...abcd',
                         domega_Em, deps_eq_Emcd, self.D_abef, eps_Emab_n1)

    traits_view = View(
        VSplit(
            Group(
                Item('E'),
                Item('nu'),
                Item('epsilon_0'),
                Item('epsilon_f'),
                Item('strain_norm')
            ),
            Group(
                Item('stress_state', style='custom'),
                Item('stiffness', style='custom'),
                Spring(resizable=True),
                label='Configuration parameters',
                show_border=True,
            ),
        ),
        resizable=True
    )

    tree_view = View(
        Group(
            Item('E', full_size=True, resizable=True),
            Item('nu'),
            Item('epsilon_0'),
            Item('epsilon_f'),
            Item('strain_norm')
        ),
    )

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = tr.Trait(tr.Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'omega': self.get_omega}

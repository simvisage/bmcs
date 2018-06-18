
from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn, DamageFn, GfDamageFn
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats2D.vmats2D_eval import MATS2D
from ibvpy.mats.mats_eval import IMATSEval
from traits.api import Float, Property, Array
from traitsui.api import VGroup, UItem, \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import EitherType

import numpy as np
import traits.api as tr
from vstrain_norm2d import Rankine


# from scipy.linalg import eig, inv
#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------


class MATS2DScalarDamage(MATS2DEval, MATS2D):
    r'''
    Isotropic damage model.
    '''

    tr.implements(IMATSEval)

    omega_fn = tr.Instance(GfDamageFn)

    def _omega_fn_default(self):
        return GfDamageFn()

    stiffness = tr.Enum("secant", "algorithmic",
                        input=True)
    r'''Selector of the stiffness calculation.
    '''
    strain_norm = EitherType(klasses=[Rankine,
                                      ], input=True)
    r'''Selector of the strain norm defining the load surface.
    '''

    changed = tr.Event
    r'''This event can be used by the clients to trigger 
    an action upon the completed reconfiguration of 
    the material model
    '''

    state_array_shapes = {'kappa': (),
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

    def get_corr_pred(self, eps_Emab_n1, deps_Emab, tn, tn1,
                      update_state, algorithmic, kappa, omega):
        r'''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        if update_state:
            eps_Emab_n = eps_Emab_n1 - deps_Emab
            kappa_Em, omega_Em, f_idx = self._get_state_variables(
                eps_Emab_n, kappa, omega)
            kappa[...] = kappa_Em
            omega[...] = omega_Em

        kappa_Em, omega_Em, f_idx = self._get_state_variables(
            eps_Emab_n1, kappa, omega
        )
        phi_Em = (1.0 - omega_Em)
        D_Emabcd = np.einsum(
            'Em,abcd->Emabcd', phi_Em, self.D_abcd
        )
        sigma_Emab = np.einsum(
            'Emabcd,Emcd->Emab', D_Emabcd, eps_Emab_n1
        )
        if algorithmic:
            D_Emabcd_red = self._get_D_abcd_alg_reduction(
                kappa_Em, eps_Emab_n1, f_idx)
            D_Emabcd[f_idx] -= D_Emabcd_red[f_idx]

        return D_Emabcd, sigma_Emab

    def _get_state_variables(self, eps_Emab, kappa, omega):
        kappa_Em = np.copy(kappa)
        eps_eq_Em = self.strain_norm.get_eps_eq(eps_Emab, kappa_Em)
        f_idx = self._get_f_trial(eps_eq_Em)
        kappa_Em[f_idx] = eps_eq_Em[f_idx]
        omega_Em = self._get_omega(eps_eq_Em, f_idx)
        return kappa_Em, omega_Em, f_idx

    def _get_f_trial(self, eps_eq_Em):
        return self.omega_fn.get_f_trial(eps_eq_Em)

    def _get_omega(self, kappa_Em, idx_map):
        r'''
        Return new value of damage parameter
        @param kappa_Em: maximum strain norm achieved so far 
        '''
        return self.omega_fn(kappa_Em, idx_map)

    def _get_domega(self, kappa_Em, idx_map):
        '''
        Return new value of damage parameter derivative
        @param kappa_Em: maximum strain norm achieved so far
        '''
        return self.omega_fn.diff(kappa_Em, idx_map)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        domega_Em = np.zeros_like(kappa_Em)
        kappa_idx = np.where(kappa_Em >= epsilon_0)
        factor_1 = epsilon_0 / (kappa_Em[kappa_idx] * kappa_Em[kappa_idx])
        factor_2 = epsilon_0 / (kappa_Em[kappa_idx] * (epsilon_f - epsilon_0))
        domega_Em[kappa_idx] = (
            (factor_1 + factor_2) * np.exp(-(kappa_Em[kappa_idx] - epsilon_0) /
                                           (epsilon_f - epsilon_0))
        )
        return domega_Em

    def _get_D_abcd_alg_reduction(self, kappa_Em, eps_Emab_n1, f_idx):
        '''Calculate the stiffness term to be subtracted
        from the secant stiffness to get the algorithmic stiffness.
        '''
        domega_Em = self._get_domega(kappa_Em, f_idx)
        deps_eq_Emcd = self.strain_norm.get_deps_eq(eps_Emab_n1)
        return np.einsum('...,...cd,abcd,...cd->...abcd',
                         domega_Em, deps_eq_Emcd, self.D_abcd, eps_Emab_n1)

    def get_G_f(self):
        eps_t = np.linspace(0, 01, 1000)
        omega_t = self._get_omega(eps_t)
        sig_t = (1 - omega_t) * self.E * eps_t
        return np.trapz(sig_t, eps_t)

    tree_view = View(
        Item('E')
    )

    trait_view = View(
        VSplit(
            Group(
                Item('E'),
                Item('nu'),
                Item('strain_norm@'),
                Item('omega_fn@')
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


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    f_t = 2.4
    G_f = 0.090
    E = 20000.0
    omega_fn_gf = GfDamageFn(G_f=G_f, f_t=f_t, E=E, L_s=22.5)
    print omega_fn_gf.eps_0
    print omega_fn_gf(0.000148148)

#
#     mats = MATS2DScalarDamage(E=30000,
#                               stiffness='algorithmic',
#                               nu=0.0,
#                               )
#
#     tv = mats.trait_view('trait_view')

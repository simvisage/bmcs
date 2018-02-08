
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats_eval import IMATSEval
from traitsui.api import \
    Item, View, VSplit, Group, Spring
from util.traits.either_type import EitherType

from ibvpy.mats.mats2D.vmats2D_eval import MATS2D
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

    state_array_shapes = tr.Dict({'kappa': (),
                                  'omega': ()})
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
                      update_state, kappa, omega):
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
            'Em,Emabcd,Emcd->Emab', phi_Em, D_Emabcd, eps_Emab_n1
        )

        if self.stiffness == "algorithmic":
            D_Emabcd_red = self._get_D_abcd_alg_reduction(
                kappa_Em[f_idx], eps_Emab_n1[f_idx])
            D_Emabcd[f_idx] -= D_Emabcd_red

        return D_Emabcd, sigma_Emab

    def _get_state_variables(self, eps_Emab, kappa, omega):
        kappa_Em = np.copy(kappa)
        omega_Em = np.copy(omega)
        eps_eq_Em = self.strain_norm.get_eps_eq(eps_Emab, kappa_Em)
        f_trial_Em = eps_eq_Em - self.epsilon_0
        f_idx = np.where(f_trial_Em > 0)
        kappa_Em[f_idx] = eps_eq_Em[f_idx]
        omega_Em[f_idx] = self._get_omega(eps_eq_Em[f_idx])
        return kappa_Em, omega_Em, f_idx

    def _get_omega(self, kappa_Em):
        r'''
        Return new value of damage parameter
        @param kappa_Em: maximum strain norm achieved so far 
        '''
        omega_Em = np.zeros_like(kappa_Em)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        kappa_idx = np.where(kappa_Em >= epsilon_0)
        omega_Em[kappa_idx] = (
            1.0 - (epsilon_0 / kappa_Em[kappa_idx] *
                   np.exp(-1.0 * (kappa_Em[kappa_idx] - epsilon_0) /
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
        kappa_idx = np.where(kappa_Em >= epsilon_0)
        factor_1 = epsilon_0 / (kappa_Em[kappa_idx] * kappa_Em[kappa_idx])
        factor_2 = epsilon_0 / (kappa_Em[kappa_idx] * (epsilon_f - epsilon_0))
        domega_Em[kappa_idx] = (
            (factor_1 + factor_2) * np.exp(-(kappa_Em[kappa_idx] - epsilon_0) /
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
                         domega_Em, deps_eq_Emcd, self.D_abcd, eps_Emab_n1)

    view_traits = View(VSplit(Group(Item('E'),
                                    Item('nu'),
                                    Item('epsilon_0'),
                                    Item('epsilon_f'),
                                    Item('strain_norm')),
                              Group(Item('stress_state', style='custom'),
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

    mats = MATS2DScalarDamage(E=30000,
                              stiffness='algorithmic',
                              nu=0.0,
                              epsilon_0=0.3,
                              epsilon_f=0.5)

    eps_Emab = np.array([[[[0.8, 0],
                           [0, 0.0]]
                          ]], dtype=np.float_)
    s_Emg = np.array([[[0, 0],
                       ]], dtype=np.float_)

    D_Emabcd, sig_Emab = mats.get_corr_pred(
        eps_Emab, eps_Emab, 0, 0, False, s_Emg)

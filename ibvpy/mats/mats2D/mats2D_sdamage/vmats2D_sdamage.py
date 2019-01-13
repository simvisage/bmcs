
from traits.api import on_trait_change, Bool
from traitsui.api import UItem, \
    Item, View, Group, Spring

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn, DamageFn, GfDamageFn
from bmcs.simulator import \
    Model, TLoopImplicit, TStepBC
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats2D.vmats2D_eval import MATS2D
import numpy as np
import traits.api as tr

from .vstrain_norm2d import StrainNorm2D, SN2DRankine


class MATS2DScalarDamage(Model, MATS2DEval, MATS2D):
    r'''
    Isotropic damage model.
    '''

    tloop_type = TLoopImplicit
    tstep_type = TStepBC

    node_name = 'isotropic damage model'

    tree_node_list = tr.List

    def _tree_node_list_default(self):
        return [self.strain_norm, self.omega_fn, ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.strain_norm,
            self.omega_fn,
        ]

    stiffness = tr.Enum("secant", "algorithmic",
                        input=True)
    r'''Selector of the stiffness calculation.
    '''
    #=========================================================================
    # Material model
    #=========================================================================
    strain_norm_type = tr.Trait('Rankine',
                                {'Rankine': SN2DRankine,
                                 },
                                MAT=True
                                )

    @on_trait_change('strain_norm_type')
    def _set_strain_norm(self):
        self.strain_norm = self.strain_norm_type_(mats=self)
        self._update_node_list()

    strain_norm = tr.Instance(StrainNorm2D,
                              report=True)
    '''Material model'''

    def _strain_norm_default(self):
        return self.strain_norm_type_(mats=self)

    omega_fn = tr.Instance(GfDamageFn)

    def _omega_fn_default(self):
        return GfDamageFn(mats=self)

    changed = tr.Event
    r'''This event can be used by the clients to trigger 
    an action upon the completed reconfiguration of 
    the material model
    '''

    state_var_shapes = {'kappa_n': (),
                        'omega_n': ()}
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

    algorithmic = Bool(True)

    def get_corr_pred(self, eps_ab_k, tn1, kappa_n, omega_n):
        r'''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        kappa_k, omega_k, f_idx = self._get_state_variables(
            eps_ab_k, kappa_n, omega_n
        )
        phi_k = (1.0 - omega_k)
        D_abcd_k = np.einsum(
            '...,abcd->...abcd',
            phi_k, self.D_abcd
        )
        sig_ab_k = np.einsum(
            '...abcd,...cd->...ab',
            D_abcd_k, eps_ab_k
        )
        if self.algorithmic:
            D_abcd_k_red = self._get_D_abcd_alg_reduction(
                kappa_k, eps_ab_k, f_idx)
            D_abcd_k[f_idx] -= D_abcd_k_red[f_idx]

        return sig_ab_k, D_abcd_k

    def update_state(self, eps_Emab_n1, t_n1, kappa_n, omega_n):
        eps_Emab_n = eps_Emab_n1
        kappa_n[...], omega_n[...], f_idx = self._get_state_variables(
            eps_Emab_n, kappa_n, omega_n
        )

    def _get_state_variables(self, eps_ab, kappa, omega):
        kappa_Em = np.copy(kappa)
        eps_eq_Em = self.strain_norm.get_eps_eq(eps_ab, kappa_Em)
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

    def _get_D_abcd_alg_reduction(self, kappa_Em, eps_Emab_n1, f_idx):
        '''Calculate the stiffness term to be subtracted
        from the secant stiffness to get the algorithmic stiffness.
        '''
        domega_Em = self._get_domega(kappa_Em, f_idx)
        deps_eq_Emcd = self.strain_norm.get_deps_eq(eps_Emab_n1)
        return np.einsum('...,...cd,abcd,...cd->...abcd',
                         domega_Em, deps_eq_Emcd, self.D_abcd, eps_Emab_n1)

    traits_view = View(
        Group(
            Group(
                Item('E', resizable=True),
                Item('nu', resizable=True),
                Item('strain_norm_type', resizable=True),
                Group(
                    UItem('omega_fn@', full_size=True),
                    label='Damage function'
                )
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

    tree_view = traits_view


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    f_t = 2.4
    G_f = 0.090
    E = 20000.0
    omega_fn_gf = GfDamageFn(G_f=G_f, f_t=f_t, L_s=22.5)
    print(omega_fn_gf.eps_0)
    print(omega_fn_gf(0.000148148))

    mats = MATS2DScalarDamage(E=30000,
                              stiffness='algorithmic',
                              nu=0.0,
                              )
    print(mats.strain_norm)
    mats.configure_traits()

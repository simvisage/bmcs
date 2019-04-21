
from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from simulator.api import \
    TLoopImplicit, TStepBC
from traits.api import on_trait_change, Bool
from traitsui.api import UItem, \
    Item, View, Group, Spring

from ibvpy.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn, DamageFn, GfDamageFn
import numpy as np
import traits.api as tr

from .vstrain_norm2d import StrainNorm2D, SN2DRankine


class MATS2DScalarDamage(MATS2DEval):
    r'''Isotropic damage model.
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

    algorithmic = Bool(True)

    def get_corr_pred(self, eps_ab, tn1, kappa, omega):
        r'''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        eps_eq = self.strain_norm.get_eps_eq(eps_ab, kappa)
        I = self.omega_fn.get_f_trial(eps_eq)
        eps_eq_I = eps_eq[I]
        kappa[I] = eps_eq_I
        omega[I] = self.omega_fn(eps_eq_I)
        phi = (1.0 - omega)
        D_abcd = np.einsum(
            '...,abcd->...abcd',
            phi, self.D_abcd
        )
        sig_ab = np.einsum(
            '...abcd,...cd->...ab',
            D_abcd, eps_ab
        )
        if self.algorithmic:
            domega_I = self.omega_fn.diff(eps_eq_I)
            domega_I[np.where(np.fabs(domega_I) < .001)] = .001
            deps_eq_cd_I = self.strain_norm.get_deps_eq(eps_ab[I])
            D_abcd[I] -= np.einsum(
                '...,...cd,abcd,...cd->...abcd',
                domega_I, deps_eq_cd_I, self.D_abcd, eps_ab[I]
            )
        return sig_ab, D_abcd

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

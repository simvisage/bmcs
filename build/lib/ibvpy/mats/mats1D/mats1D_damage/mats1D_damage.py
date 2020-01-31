
from math import exp, sin

from traits.api import \
    Enum, Float,  \
    Trait,  Event, \
    Dict
from traitsui.api import \
    Item, View, Group, Spring

from ibvpy.mats.mats1D.mats1D_eval import MATS1DEval
import numpy as np


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS1DDamage(MATS1DEval):
    '''
    Scalar Damage Model.
    '''

    E = Float(1.,  # 34e+3,
              modified=True,
              label="E",
              desc="Young's Modulus",
              enter_set=True,
              auto_set=False)

    epsilon_0 = Float(1.,  # 59e-6,
                      modified=True,
                      label="eps_0",
                      desc="Breaking Strain",
                      enter_set=True,
                      auto_set=False)

    epsilon_f = Float(1.,  # 191e-6,
                      modified=True,
                      label="eps_f",
                      desc="Shape Factor",
                      enter_set=True,
                      auto_set=False)

    stiffness = Enum("secant", "algorithmic",
                     modified=True)

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    traits_view = View(Group(Group(Item('E'),
                                   Item('epsilon_0'),
                                   Item('epsilon_f'),
                                   label='Material parameters',
                                   show_border=True),
                             Group(Item('stiffness', style='custom'),
                                   Spring(resizable=True),
                                   label='Configuration parameters',
                                   show_border=True,
                                   ),
                             layout='tabbed'
                             ),
                       resizable=True
                       )

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def get_state_array_size(self):
        '''
        Give back the nuber of floats to be saved
        @param sctx:spatial context
        '''
        return 2

    def new_cntl_var(self):
        return np.zeros(1, np.float_)

    def new_resp_var(self):
        return np.zeros(1, np.float_)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        if eps_avg == None:
            eps_avg = eps_app_eng

        E = self.E
        D_el = np.array([E])

        if sctx.update_state_on:

            kappa_n = sctx.mats_state_array[0]
            kappa_k = sctx.mats_state_array[1]
            sctx.mats_state_array[0] = kappa_k

        kappa_k, omega = self._get_state_variables(sctx, eps_avg)
        sctx.mats_state_array[1] = kappa_k

        if self.stiffness == "algorithmic":
            D_e_dam = np.array([self._get_alg_stiffness(sctx, eps_app_eng,
                                                        kappa_k,
                                                        omega)])
        else:
            D_e_dam = np.array([(1 - omega) * D_el])

        sigma = np.dot(np.array([(1 - omega) * D_el]), eps_app_eng)

        # print the stress you just computed and the value of the apparent E

        return sigma, D_e_dam

    #--------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #--------------------------------------------------------------------------

    def _get_state_variables(self, sctx, eps):

        kappa_n, kappa_k = sctx.mats_state_array

        kappa_k = max(abs(eps), kappa_n)

        omega = self._get_omega(sctx, kappa_k)

        return kappa_k, omega

    def _get_omega(self, sctx, kappa):
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        if kappa >= epsilon_0:
            return 1. - epsilon_0 / kappa * exp(-1 * (kappa - epsilon_0) / epsilon_f)
        else:
            return 0.

    def _get_alg_stiffness(self, sctx, eps_app_eng, e_max, omega):
        E = self.E
        D_el = np.array([E])
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        dodk = (epsilon_0 / (e_max * e_max) * exp(-(e_max - epsilon_0) / epsilon_f) +
                epsilon_0 / e_max / epsilon_f * exp(-(e_max - epsilon_0) / epsilon_f))
        D_alg = (1 - omega) * D_el - D_el * eps_app_eng * dodk
        return D_alg

    #--------------------------------------------------------------------------
    # Response trace evaluators
    #--------------------------------------------------------------------------

    def get_omega(self, sctx, eps_app_eng, eps_avg=None):
        if eps_avg == None:
            eps_avg = eps_app_eng
        return self._get_omega(sctx, eps_avg)

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'omega': self.get_omega}

    #-------------------------------------------------------------------------
    # List of response tracers to be constructed within the mats_explorer
    #-------------------------------------------------------------------------
    def _get_explorer_rtrace_list(self):
        '''Return the list of relevant tracers to be used in mats_explorer.
        '''
        return []

    def _get_explorer_config(self):
        from ibvpy.api import TLine, RTDofGraph, BCDof
        ec = super(MATS1DDamage, self)._get_explorer_config()
        ec['mats_eval'] = MATS1DDamage(E=1.0, epsilon_0=1.0, epsilon_f=5)
        ec['bcond_list'] = [BCDof(var='u',
                                  dof=0, value=1.7,
                                  time_function=lambda t: (1 + 0.1 * t) * sin(t))]
        ec['tline'] = TLine(step=0.1, max=10)
        ec['rtrace_list'] = [
            RTDofGraph(name='strain - stress',
                       var_x='eps_app', idx_x=0,
                       var_y='sig_app', idx_y=0,
                       record_on='update'),
            RTDofGraph(name='time - damage',
                       var_x='time', idx_x=0,
                       var_y='omega', idx_y=0,
                       record_on='update')
        ]
        return ec


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    mats_eval = MATS1DDamage()
    mats_eval.configure_traits()

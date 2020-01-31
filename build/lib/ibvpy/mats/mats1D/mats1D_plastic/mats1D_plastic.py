#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 4, 2009 by: rch

from math import copysign, sin

from traits.api import \
    Float,  \
    Trait,    \
    Dict
from traitsui.api import \
    Item, View, Group, Spring

from ibvpy.mats.mats1D.mats1D_eval import MATS1DEval
import numpy as np


# from dacwt import DAC
def sign(val):
    return copysign(1, val)


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------


class MATS1DPlastic(MATS1DEval):

    '''
    Scalar Damage Model.
    '''

    E = Float(1.,  # 34e+3,
              label="E",
              desc="Young's Modulus",
              enter_set=True,
              auto_set=False)

    sigma_y = Float(1.,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.1,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.1,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    traits_view = View(Group(Group(Item('E'),
                                   Item('sigma_y'),
                                   Item('K_bar'),
                                   Item('H_bar'),
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

        eps_p_n - platic strain 
        alpha_n - hardening
        q_n - back stress  

        '''
        return 3

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
        eps_n1 = float(eps_app_eng)
        E = self.E

        if eps_avg == None:
            eps_avg = eps_n1

        if sctx.update_state_on:
            eps_n = eps_avg - float(d_eps)
            sctx.mats_state_array[:] = self._get_state_variables(sctx, eps_n)

        eps_p_n, q_n, alpha_n = sctx.mats_state_array
        sigma_trial = self.E * (eps_n1 - eps_p_n)
        xi_trial = sigma_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        sig_n1 = np.zeros((1,), dtype='float_')
        D_n1 = np.zeros((1, 1), dtype='float_')
        if f_trial <= 1e-8:
            sig_n1[0] = sigma_trial
            D_n1[0, 0] = E
        else:
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            sig_n1[0] = sigma_trial - d_gamma * self.E * sign(xi_trial)
            D_n1[0, 0] = (self.E * (self.K_bar + self.H_bar)) / \
                (self.E + self.K_bar + self.H_bar)

        return sig_n1, D_n1

    #--------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #--------------------------------------------------------------------------

    def _get_state_variables(self, sctx, eps_n):

        eps_p_n, q_n, alpha_n = sctx.mats_state_array

        # Get the characteristics of the trial step
        #
        sig_trial = self.E * (eps_n - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        if f_trial > 1e-8:

            #
            # Tha last equilibrated step was inelastic. Here the
            # corresponding state variables must be calculated once
            # again. This might be expensive for 2D and 3D models. Then,
            # some kind of caching should be considered for the state
            # variables determined during iteration. In particular, the
            # computation of d_gamma should be outsourced into a separate
            # method that can in general perform an iterative computation.
            #
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            eps_p_n += d_gamma * sign(xi_trial)
            q_n += d_gamma * self.H_bar * sign(xi_trial)
            alpha_n += d_gamma

        newarr = np.array([eps_p_n, q_n, alpha_n], dtype='float_')

        return newarr

    #-----------------------------------------------------------
    # Response trace evaluators
    #--------------------------------------------------------------------------
    def get_eps_p(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[0]])

    def get_q(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[1]])

    def get_alpha(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[2]])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'eps_p': self.get_eps_p,
                'q': self.get_q,
                'alpha': self.get_alpha}

    def _get_explorer_config(self):
        from ibvpy.api import TLine, BCDof, RTDofGraph
        c = super(MATS1DPlastic, self)._get_explorer_config()
        # overload the default configuration
        c['bcond_list'] = [BCDof(var='u',
                                 dof=0, value=2.0,
                                 time_function=lambda t: sin(t))]
        c['rtrace_list'] = [
            RTDofGraph(name='strain - stress',
                       var_x='eps_app', idx_x=0,
                       var_y='sig_app', idx_y=0,
                       record_on='update'),
            RTDofGraph(name='time - plastic_strain',
                       var_x='time', idx_x=0,
                       var_y='eps_p', idx_y=0,
                       record_on='update'),
            RTDofGraph(name='time - back stress',
                       var_x='time', idx_x=0,
                       var_y='q', idx_y=0,
                       record_on='update'),
            RTDofGraph(name='time - hardening',
                       var_x='time', idx_x=0,
                       var_y='alpha', idx_y=0,
                       record_on='update')
        ]
        c['tline'] = TLine(step=0.3, max=10)
        return c


if __name__ == '__main__':

    #-------------------------------------------------------------------------
    # Example
    #-------------------------------------------------------------------------

    mats_eval = MATS1DPlastic()
    mats_eval.configure_traits()

#    ex = MATSExplore( dim = MATS1DExplore( mats_eval  = MATS1DDamage( ) ) )
#    app = IBVPyApp( tloop = ex )
#    app.main()

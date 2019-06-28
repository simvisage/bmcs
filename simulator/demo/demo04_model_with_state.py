
from traits.api import \
    Float
import traits.has_traits

import numpy as np
from simulator.api import \
    Model, Simulator, TLoopImplicit, TStepState

from .interaction_scripts import run_rerun_test
traits.has_traits.CHECK_INTERFACES = 2


class ModelWithState(Model):
    '''Model with a state management distinguishing .
    '''
    tloop_type = TLoopImplicit
    tstep_type = TStepState

    U_var_shape = (1,)
    '''Shape of the primary variable required by the TStepState.
    '''
    state_var_shapes = dict(
        eps_p_n=(1,),
        q_n=(1,),
        alpha_n=(1,)
    )
    '''Shape of the state variables.
    '''

    E = Float(1.,
              label="E",
              desc="Young's Modulus",
              enter_set=True,
              auto_set=False)

    sigma_y = Float(1.,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.1,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.1,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    def get_corr_pred(self, U_k, t_n1, eps_p_n, q_n, alpha_n):
        '''Return the value and the derivative of a function
        '''
        eps_n1 = U_k
        E = self.E

        sig_trial = self.E * (eps_n1 - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        D_shape = sig_trial.shape[:-2] + (1, 1)
        D_k = np.zeros(D_shape, dtype='float_')

        sig_k = sig_trial
        D_k[...] = E
        I = np.where(f_trial > 1e-8)
        d_gamma = f_trial[I] / (self.E + self.K_bar + self.H_bar)
        sig_k[I] -= self.E * d_gamma * np.sign(xi_trial[I])
        D_k[I] = (
            (self.E * (self.K_bar + self.H_bar)) /
            (self.E + self.K_bar + self.H_bar)
        )
        return sig_k, D_k

    def update_state(self, U_k, t_n1, eps_p_n, q_n, alpha_n):
        '''In-place update of state variables. 
        '''
        eps_n = U_k
        sig_trial = self.E * (eps_n - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)
        I = np.where(f_trial > 1e-8)
        d_gamma = f_trial[I] / (self.E + self.K_bar + self.H_bar)
        eps_p_n[I] += d_gamma * np.sign(xi_trial[I])
        q_n[I] += d_gamma * self.H_bar * np.sign(xi_trial[I])
        alpha_n[I] += d_gamma


# Construct a Simulator
m = ModelWithState(sigma_y=0.5)
s = Simulator(model=m)
run_rerun_test(s)

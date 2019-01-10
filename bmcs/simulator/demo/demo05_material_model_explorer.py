
from traits.api import \
    Float
import traits.has_traits
from bmcs.simulator import \
    Model, Simulator, TLoopImplicit, TStepState
import numpy as np
from .interaction_scripts import run_rerun_test
traits.has_traits.CHECK_INTERFACES = 2


class MaterialModel(Model):
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

    R_0 = Float(1.0, auto_set=False, enter_set=True)
    '''Target value of a function (load).
    '''

    def get_corr_pred(self, U_k, t_n, eps_p_n, q_n, alpha_n):
        '''Return the value and the derivative of a function
        '''
        sig_t = self.R_0 * t_n

        eps_n1 = U_k
        E = self.E

        sig_trial = self.E * (eps_n1 - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        sig_n1 = np.zeros_like(sig_trial)
        D_n1 = np.zeros((1, 1), dtype='float_')
        if f_trial <= 1e-8:
            sig_n1[:] = sig_trial
            D_n1[:, :] = E
        else:
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            sig_n1[:] = sig_trial - d_gamma * self.E * np.sign(xi_trial)
            D_n1[:, :] = (self.E * (self.K_bar + self.H_bar)) / \
                (self.E + self.K_bar + self.H_bar)
        return sig_t - sig_n1, D_n1

    def update_state(self, U_k, t_n, eps_p_n, q_n, alpha_n):
        '''In-place update os state variables. 
        '''
        eps_n = U_k
        sig_trial = self.E * (eps_n - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)
        if f_trial > 1e-8:
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            eps_p_n += d_gamma * np.sign(xi_trial)
            q_n += d_gamma * self.H_bar * np.sign(xi_trial)
            alpha_n += d_gamma


# Construct a Simulator
m = ModelWithState(R_0=2.0)
s = Simulator(model=m)
run_rerun_test(s)

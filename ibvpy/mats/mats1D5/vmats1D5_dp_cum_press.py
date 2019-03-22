'''
Created on Feb 14, 2019

@author: rch
'''
from ibvpy.api import MATSEval
import numpy as np
from simulator.i_model import IModel
import traits.api as tr


@tr.provides(IModel)
class MATS1D5DPCumPress(MATSEval):

    node_name = 'Pressure sensitive cumulative damage plasticity'

    E_N = tr.Float(100, label='E_N',
                   desc='Normal stiffness of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    E_T = tr.Float(100, label='E_T',
                   desc='Shear modulus of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    gamma = tr.Float(40.0, label='gamma',
                     desc='Kinematic Hardening Modulus',
                     MAT=True,
                     enter_set=True, auto_set=False)

    K = tr.Float(1, label='K',
                 desc='Isotropic hardening modulus',
                 MAT=True,
                 enter_set=True, auto_set=False)

    S = tr.Float(0.1, label='S',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    r = tr.Float(0.1, label='r',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    c = tr.Float(1, Label='c',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    tau_0 = tr.Float(1, label='tau_0',
                     desc='Reversibility limit',
                     MAT=True,
                     enter_set=True, auto_set=False)

    m = tr.Float(1, label='m',
                 desc='Lateral Pressure Coefficient',
                 MAT=True,
                 enter_set=True, auto_set=False)

    tau_bar = tr.Float(1.1)

    state_var_shapes = dict(s_pi=(),
                            alpha=(),
                            z=(),
                            omega=())

    D_rs = tr.Property(depends_on='E_N,E_T')

    @tr.cached_property
    def _get_D_rs(self):
        return np.array([[self.E_T, 0],
                         [0, self.E_N]], dtype=np.float_)

    def init(self, s_pi, alpha, z, omega):
        r'''
        Initialize the state variables.
        '''
        s_pi[...] = 0
        alpha[...] = 0
        z[...] = 0
        omega[...] = 0

    algorithmic = tr.Bool(True)

    def get_corr_pred(self, u_r, t_n, s_pi, alpha, z, omega):

        s = u_r[..., 0]
        w = u_r[..., 1]
        # For normal
        H_w_N = np.array(w <= 0.0, dtype=np.float_)
        E_alg_N = H_w_N * self.E_N
        sig_N = E_alg_N * w

        # For tangential
        #Y = 0.5 * self.E_T * (u_T - s_pi)**2
        tau_pi_trial = self.E_T * (s - s_pi)
        Z = self.K * z
        X = self.gamma * alpha

        f = np.fabs(tau_pi_trial - X) - Z - self.tau_0  # + self.m * sig_N
        I = f > 1e-6

        sig_T = self.E_T * s

        # Return mapping
        delta_lambda_I = f[I] / \
            (self.E_T / (1 - omega[I]) + self.gamma + self.K)

        # update all state variables

        s_pi[I] += (delta_lambda_I *
                    np.sign(tau_pi_trial[I] - X[I]) / (1 - omega[I]))

        Y = 0.5 * self.E_T * (s - s_pi)**2

        omega[I] += (delta_lambda_I * (1 - omega[I])
                     ** self.c * (Y[I] / self.S)**self.r)

        sig_T[I] = (1 - omega[I]) * self.E_T * (s[I] - s_pi[I])

        alpha[I] += delta_lambda_I * np.sign(tau_pi_trial[I] - X[I])

        z[I] += delta_lambda_I

        # Algorithmic Stiffness

        E_alg_T = (1 - omega) * self.E_T

        # Consistent tangent operator
        if False:
            E_alg_T = (
                (1 - omega) * self.E_T -
                (1 - omega) * self.E_T ** 2 /
                (self.E_T + (self.gamma + self.K) * (1 - omega)) -
                ((1 - omega) ** self.c * (self.E_T ** 2) * ((Y / self.S) ** self.r)
                 * np.sign(tau_pi_trial - X) * (s - s_pi)) /
                ((self.E_T / (1 - omega)) + self.gamma + self.K)
            )

        if False:
            E_alg_T = (
                (1 - omega) * self.E_T -
                ((self.E_T**2 * (1 - omega)) /
                 (self.E_T + (self.gamma + self.K) * (1 - omega)))
                -
                ((1 - omega)**self.c *
                 (Y_I / self.S)**self.r *
                 self.E_T**2 * (s - s_pi) * self.tau_bar /
                 (self.tau_bar - self.m * sig_N) * np.sign(tau_pi_trial - X)) /
                (self.E_T / (1 - omega) + self.gamma + self.K)
            )

        sig = np.zeros_like(u_r)
        sig[..., 0] = sig_T
        sig[..., 1] = sig_N
        E_TN = np.einsum('abEm->Emab',
                         np.array(
                             [
                                 [E_alg_T, np.zeros_like(E_alg_T)],
                                 [np.zeros_like(E_alg_N), E_alg_N]
                             ])
                         )
        return sig, E_TN
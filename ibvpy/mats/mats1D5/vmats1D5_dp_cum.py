'''
Created on Feb 14, 2019

@author: rch
'''

from sqlalchemy.orm import state
from traits.api import on_trait_change

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from ibvpy.api import MATSEval
import numpy as np
from simulator.i_model import IModel
import traits.api as tr


@tr.provides(IModel)
class MATS1D5DPCum(MATSEval):

    node_name = 'Cumulative damage plasticity model'

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

    c = tr.Float(1, Label='c',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    tau_bar = tr.Float(1, label='tau_bar',
                       desc='Reversibility limit',
                       MAT=True,
                       enter_set=True, auto_set=False)

    state_var_shapes = dict(s_p=(),
                            alpha=(),
                            z=(),
                            omega=(),
                            kappa=())

    uncoupled_dp = tr.Bool(False,
                           MAT=True,
                           label='Uncoupled d-p'
                           )

    s_0 = tr.Float(MAT=True,
                   desc='Elastic strain/displacement limit')

    def __init__(self, *args, **kw):
        super(MATS1D5DP, self).__init__(*args, **kw)
        self._update_s0()

    @on_trait_change('tau_bar,E_T')
    def _update_s0(self):
        if not self.uncoupled_dp:
            if self.E_T == 0:
                self.s_0 = 0
            else:
                self.s_0 = self.tau_bar / self.E_T
            self.omega_fn.s_0 = self.s_0

    omega_fn_type = tr.Trait('FRP',
                             dict(li=LiDamageFn,
                                  jirasek=JirasekDamageFn,
                                  abaqus=AbaqusDamageFn,
                                  FRP=FRPDamageFn,
                                  multilinear=MultilinearDamageFn
                                  ),
                             MAT=True,
                             )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        print('resetting')
        self.omega_fn = self.omega_fn_type_(s_0=self.s_0)

    omega_fn = tr.Instance(IDamageFn, report=True)

    def _omega_fn_default(self):
        return MultilinearDamageFn()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def init(self, s_pi, alpha, z, omega, kappa):
        r'''
        Initialize the state variables.
        '''
        s_pi[...] = 0
        alpha[...] = 0
        z[...] = 0
        omega[...] = 0
        kappa[...] = 0

    algorithmic = tr.Bool(True)

    def get_corr_pred(self, u_r, t_n, s_p, alpha, z, omega, kappa):

        s = u_r[..., 0]
        w = u_r[..., 1]
        # For normal
        H_w_N = np.array(w <= 0.0, dtype=np.float_)
        E_alg_N = H_w_N * self.E_N
        sig_N = E_alg_N * w
        sig_pi_trial = self.E_T * (s - s_p)
        Z = self.K * z
        # for handling the negative values of isotropic hardening
        h_1 = self.tau_bar + Z
        pos_iso = h_1 > 1e-6

        X = self.gamma * alpha

        # for handling the negative values of kinematic hardening (not yet)
        # h_2 = h * np.sign(sig_pi_trial - X) * \
        #    np.sign(sig_pi_trial) + X * np.sign(sig_pi_trial)
        #pos_kin = h_2 > 1e-6

        f = np.fabs(sig_pi_trial - X) - h_1 * pos_iso

        I = f > 1e-6
        # Return mapping
        delta_lamda_I = f[I] / (self.E_T + self.gamma + np.fabs(self.K))
        # update all the state variables
        s_p[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        z[I] += delta_lamda_I
        alpha[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        kappa[...] = np.max(np.array([kappa, np.fabs(s)]), axis=0)
        omega[...] = self.omega(kappa)
        tau = (1 - omega) * self.E_T * (s - s_p)

        E_alg_T = (1 - omega) * self.E_T

        domega_ds = self.omega_derivative(kappa)
        # Consistent tangent operator
        E_alg_T[I] = -self.E_T / (self.E_T + self.K + self.gamma) \
            * domega_ds[I] * self.E_T * (s[I] - s_p[I]) \
            + (1 - omega[I]) * self.E_T * (self.K + self.gamma) / \
            (self.E_T + self.K + self.gamma)

        sig = np.zeros_like(u_r)
        sig[..., 0] = tau
        sig[..., 1] = sig_N
        E_TN = np.einsum('abEm->Emab',
                         np.array(
                             [
                                 [E_alg_T, np.zeros_like(E_alg_T)],
                                 [np.zeros_like(E_alg_N), E_alg_N]
                             ])
                         )
        return sig, E_TN

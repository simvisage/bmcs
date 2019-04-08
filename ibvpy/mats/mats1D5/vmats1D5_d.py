'''
Created on Feb 14, 2019

@author: rch
'''

from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from ibvpy.api import MATSEval
from simulator.i_model import IModel
from traits.api import on_trait_change
import numpy as np
import traits.api as tr


@tr.provides(IModel)
class MATS1D5D(MATSEval):

    node_name = "damage bond model"

    E_T = tr.Float(100.0, tooltip='Shear stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{s}',
                   desc='Shear-modulus of the interface',
                   auto_set=True, enter_set=True)

    E_N = tr.Float(100.0, tooltip='Normal stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{n}',
                   desc='Normal stiffness of the interface',
                   auto_set=False, enter_set=True)

    state_var_shapes = dict(omega=(),
                            kappa=())

    omega_fn_type = tr.Trait('jirasek',
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
        print('RESETTING')
        self.omega_fn = self.omega_fn_type_()

    omega_fn = tr.Instance(IDamageFn, report=True)

    def _omega_fn_default(self):
        return self.omega_fn_type_()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def init(self, s_pi, alpha, z, omega, kappa):
        r'''
        Initialize the state variables.
        '''
        omega[...] = 0
        kappa[...] = 0

    algorithmic = tr.Bool(True)

    def get_corr_pred(self, u_r, t_n, omega, kappa):

        s = u_r[..., 0]
        w = u_r[..., 1]
        # For normal
        H_w_N = np.array(w <= 0.0, dtype=np.float_)
        E_alg_N = H_w_N * self.E_N
        sig_N = E_alg_N * w
        kappa[...] = np.max(np.array([kappa, np.fabs(s)]), axis=0)
        omega[...] = self.omega(kappa)
        tau = (1 - omega) * self.E_T * s
        domega_ds = self.omega_derivative(kappa)
        E_alg_T = ((1 - omega) - domega_ds * s) * self.E_T

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

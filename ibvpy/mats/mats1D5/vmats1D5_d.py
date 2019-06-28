'''
Created on Feb 14, 2019

@author: rch
'''

from ibvpy.api import MATSEval
from ibvpy.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from simulator.i_model import IModel
from traits.api import on_trait_change

import numpy as np
import traits.api as tr
import traitsui.api as ui


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
        self.omega_fn = self.omega_fn_type_()

    omega_fn = tr.Instance(IDamageFn, report=True)

    def _omega_fn_default(self):
        return self.omega_fn_type_()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def init(self, omega, kappa):
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

        if self.algorithmic:
            E_alg_T = ((1 - omega) - domega_ds * s) * self.E_T
        else:
            E_alg_T = (1 - omega) * self.E_T

        sig = np.zeros_like(u_r)
        sig[..., 0] = tau
        sig[..., 1] = sig_N
        E_TN = np.einsum('ab...->...ab',
                         np.array(
                             [
                                 [E_alg_T, np.zeros_like(E_alg_T)],
                                 [np.zeros_like(E_alg_N), E_alg_N]
                             ])
                         )
        return sig, E_TN

    traits_view = ui.View(
        ui.Item('E_T'),
        ui.Item('E_N')
    )


if __name__ == '__main__':
    tau_bar = 3.0
    s_max = 0.1
    E_T = 10000
    s_0 = tau_bar / E_T
    m = MATS1D5D(E_T=E_T, omega_fn_type='jirasek', algorithmic=True)
    m.omega_fn.trait_set(s_0=s_0, s_f=100 * s_0)
    s_r = np.linspace(0, s_max, 60)
    w_r = np.linspace(0.0, 0.0, 60)
    u_r = np.vstack([s_r, w_r]).T
    state_arr = {var: np.zeros(s_r.shape + var_shape, dtype=np.float_)
                 for var, var_shape in m.state_var_shapes.items()}
    m.init(**state_arr)
    sig, D = m.get_corr_pred(u_r, 0.0, **state_arr)
    D1 = D[..., 0, 0]

    delta_s = s_max / 100
    delta_sig = np.array([sig[..., 0], sig[..., 0] + D1 * delta_s])
    delta_u = np.array([s_r, s_r + delta_s])

    import pylab as p
    p.plot(s_r, sig)
    p.plot(delta_u, delta_sig, color='red')
    p.show()

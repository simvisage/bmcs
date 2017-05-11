'''
Created on 05.12.2016

@author: abaktheer
'''

from ibvpy.api import MATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import implements, Int, Array, \
    Constant, Float, Tuple, List, on_trait_change, \
    Instance, Trait, Bool
from traitsui.api import View, VGroup, Item, UItem
from view.ui import BMCSTreeNode

from mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn
import numpy as np


class MATSEvalFatigue(MATSEval):

    E_m = Float(30000, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(200000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(200,
                label="G",
                desc="Shear Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(1,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1,
              label="r",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1,
              label="c",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(5,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    pressure = Float(-5,
                     label="Pressure",
                     desc="Lateral pressure",
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    n_s = Constant(4)

    state_array_size = Int(4)

    state_arr_shape = Tuple((4,))

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, w):

        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        Y = 0.5 * self.E_b * (eps[:, :, 1] - xs_pi) ** 2
        sig_pi_trial = self.E_b * (eps[:, :, 1] - xs_pi)

        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(sig_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * self.pressure / 3

        elas = f <= 1e-6
        plas = f > 1e-6

        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        # Return mapping
        delta_lamda = f / (self.E_b / (1 - w) + self.gamma + self.K) * plas
        # update all the state variables

        xs_pi = xs_pi + delta_lamda * np.sign(sig_pi_trial - X) / (1 - w)
        Y = 0.5 * self.E_b * (eps[:, :, 1] - xs_pi) ** 2

        w = w + (1 - w) ** self.c * (delta_lamda * (Y / self.S) ** self.r)

        sig[:, :, 1] = (1 - w) * self.E_b * (eps[:, :, 1] - xs_pi)
        #X = X + self.gamma * delta_lamda * np.sign(sig_pi_trial - X)
        alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)
        z = z + delta_lamda

        # Consistent tangent operator
        D_ed = self.E_b * (1 - w) - ((1 - w) * self.E_b ** 2) / (self.E_b + (self.gamma + self.K) * (1 - w))\
            - ((1 - w) ** self.c * (self.E_b ** 2) * ((Y / self.S) ** self.r)
               * np.sign(sig_pi_trial - X) * (eps[:, :, 1] - xs_pi)) / ((self.E_b / (1 - w)) + self.gamma + self.K)

        D[:, :, 1, 1] = (1 - w) * self.E_b * elas + D_ed * plas

        return sig, D, xs_pi, alpha, z, w


class MATSBondSlipDP(MATSEval, BMCSTreeNode):

    node_name = 'bond model: damage-plasticity'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    E_m = Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(200000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = Float(12900.0,
                label="E_b",
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = Float(100.0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = Float(1000.0,
              label="K",
              desc="Isotropic harening",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5.0,
                    label="Tau_pi_bar",
                    desc="Reversibility limit",
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    uncoupled_dp = Bool(False,
                        MAT=True,
                        label='Uncoupled d-p'
                        )
    s_0 = Float

    def __init__(self, *args, **kw):
        super(MATSBondSlipDP, self).__init__(*args, **kw)
        self._update_s0()

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        if not self.uncoupled_dp:
            self.s_0 = self.tau_bar / self.E_b
            self.omega_fn.s_0 = self.s_0

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn
                               ),
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_(s_0=self.s_0)

    omega_fn = Instance(IDamageFn,
                        MAT=True)

    def _omega_fn_default(self):
        # return JirasekDamageFn()
        return LiDamageFn(alpha_1=1.,
                          alpha_2=100.
                          )

    state_array_size = Int(5)

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def get_corr_pred(self, s, d_s, tau, t_n, t_n1,
                      s_p, alpha, z, kappa, omega):

        n_e, n_ip, n_s = s.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        sig_pi_trial = self.E_b * (s[:, :, 1] - s_p)

        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(sig_pi_trial - X) - self.tau_bar - Z

        elas = f <= 1e-6
        plas = f > 1e-6

        d_tau = np.einsum('...st,...t->...s', D, d_s)
        tau += d_tau

        # Return mapping
        delta_lamda = f / (self.E_b + self.gamma + self.K) * plas
        # update all the state variables

        s_p = s_p + delta_lamda * np.sign(sig_pi_trial - X)
        z = z + delta_lamda
        alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)

        kappa = np.max(np.array([kappa, np.fabs(s[:, :, 1])]), axis=0)
        omega = self.omega(kappa)
        tau[:, :, 1] = (1 - omega) * self.E_b * (s[:, :, 1] - s_p)

        # Consistent tangent operator
        D_ed = -self.E_b / (self.E_b + self.K + self.gamma) * self.omega_derivative(kappa) * self.E_b * (s[:, :, 1] - s_p) \
            + (1 - omega) * self.E_b * (self.K + self.gamma) / \
            (self.E_b + self.K + self.gamma)

        D[:, :, 1, 1] = (1 - omega) * self.E_b * elas + D_ed * plas

        return tau, D, s_p, alpha, z, kappa, omega

    n_s = Constant(5)

    tree_view = View(
        VGroup(
            VGroup(
                Item('E_m', full_size=True, resizable=True),
                Item('E_f'),
                Item('E_b'),
                Item('gamma'),
                Item('K'),
                Item('tau_bar'),
            ),
            VGroup(
                Item('uncoupled_dp'),
                Item('s_0'),  # , enabled_when='uncoupled_dp'),
                Item('omega_fn_type'),
            ),
            UItem('omega_fn@')
        )
    )


class MATSBondSlipMultiLinear(MATSEval, BMCSTreeNode):

    def __init__(self, *args, **kw):
        super(MATSBondSlipMultiLinear, self).__init__(*args, **kw)
        self.bs_law.replot()

    E_m = Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    bs_law = Instance(MFnLineArray)

    def _bs_law_default(self):
        return MFnLineArray(
            xdata=[0, 0],
            ydata=[0, 1])

    n_s = Constant(5)

    def get_corr_pred(self, s, d_s, tau, t_n, t_n1,
                      s_p, alpha, z, kappa, omega):

        n_e, n_ip, n_s = s.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        d_tau = np.einsum('...st,...t->...s', D, d_s)
        tau += d_tau
        s = s[:, :, 1]
        shape = s.shape
        tau[:, :, 1] = self.bs_law(s.flatten()).reshape(*shape)

        D_tau = self.bs_law.diff(s.flatten()).reshape(*shape)

        D[:, :, 1, 1] = D_tau

        return tau, D, s_p, alpha, z, kappa, omega

    tree_view = View(
        VGroup(
            VGroup(
                Item('E_m', full_size=True, resizable=True),
                Item('E_f'),
            ),
            UItem('bs_law@')
        )
    )


if __name__ == '__main__':
    m = MATSBondSlipMultiLinear()
    m.configure_traits()

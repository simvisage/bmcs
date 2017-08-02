'''
Created on 05.12.2016

@author: abaktheer
'''

from os.path import join

from ibvpy.api import MATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from reporter.report_item import RInputRecord
from traits.api import  \
    Constant, Float, Tuple, List, on_trait_change, \
    Instance, Trait, Bool, Str, Button
from traitsui.api import View, VGroup, Item, UItem, Group
from view.ui import BMCSTreeNode

from mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    FRPDamageFn
import numpy as np


class MATSBondSlipFatigue(MATSEval, BMCSTreeNode, RInputRecord):

    node_name = 'bond model: bond fatigue'

    E_m = Float(30000, tooltip='Stiffness of the matrix [MPa]',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(200000, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = Float(12900,
                label="E_b",
                desc="Bond Stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = Float(55.0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = Float(11.0,
              label="K",
              desc="Isotropic harening",
              MAT=True,
              enter_set=True,
              auto_set=False)

    S = Float(0.00048,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              MAT=True,
              auto_set=False)

    r = Float(0.5,
              label="r",
              desc="Damage cumulation parameter",
              MAT=True,
              enter_set=True,
              auto_set=False)

    c = Float(2.8,
              label="c",
              desc="Damage cumulation parameter",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(4.2,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       MAT=True,
                       enter_set=True,
                       auto_set=False)

    pressure = Float(0,
                     label="Pressure",
                     desc="Lateral pressure",
                     MAT=True,
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              MAT=True,
              enter_set=True,
              auto_set=False)

    state_arr_shape = Tuple((4,))
    n_s = Constant(5)

    def get_cp(self, s, d_s, t_n, t_n1, state):
        tau, s_p, alpha, z, kappa, omega = state
        tau, D, s_p, alpha, z, kappa,  omega = \
            self.get_corr_pred(s, d_s, tau, t_n, t_n1,
                               s_p, alpha, z, kappa, omega)
        return tau, D, state

    def get_corr_pred2(self, s_n1, d_s, t_n, t_n1, sa_n):

        tau_n, s_p_n, alpha_n, z_n, omega_n = \
            sa_n[..., 0:3], sa_n[..., 3], sa_n[..., 4],\
            sa_n[..., 5], sa_n[..., 6]

        tau, D, s_p, alpha, z, omega = \
            self.get_corr_pred(s_n1, d_s, tau_n, t_n, t_n1,
                               s_p_n, alpha_n, z_n,  omega_n)

        sa_n1 = np.concatenate([tau,
                                s_p[..., np.newaxis],
                                alpha[..., np.newaxis],
                                z[..., np.newaxis],
                                omega[..., np.newaxis]], axis=-1)
        return tau, D, sa_n1

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, kappa, w):

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

        return sig, D, xs_pi, alpha, z, kappa, w

#     tree_view = View(
#         VGroup(
#             Item('E_m', full_size=True, resizable=True),
#             Item('E_f'),
#             Item('E_b'),
#             Item('gamma'),
#             Item('K'),
#             Item('S'),
#             Item('r'),
#             Item('c'),
#             Item('tau_pi_bar'),
#             Item('pressure'),
#             Item('a'))
#     )

    tree_view = View(VGroup(Group(Item('E_b'),
                                  Item('tau_pi_bar'), show_border=True, label='Bond Stiffness and reversibility limit'),
                            Group(Item('gamma'),
                                  Item('K'), show_border=True, label='Hardening parameters'),
                            Group(Item('S'),
                                  Item('r'), Item('c'), show_border=True, label='Damage cumulation parameters'),
                            Group(Item('pressure'),
                                  Item('a'), show_border=True, label='Lateral Pressure')))


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
                symbol="$E_\mathrm{b}$",
                unit='MPa',
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = Float(100.0,
                  symbol="$\gamma$",
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = Float(1000.0,
              symbol="$K$",
              desc="Isotropic hardening modulus",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5.0,
                    symbol=r'$\bar{\tau}$',
                    desc="Reversibility limit",
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    uncoupled_dp = Bool(False,
                        MAT=True,
                        label='Uncoupled d-p'
                        )

    s_0 = Float(MAT=True,
                desc='Elastic strain/displacement limit')

    def __init__(self, *args, **kw):
        super(MATSBondSlipDP, self).__init__(*args, **kw)
        self._update_s0()

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        if not self.uncoupled_dp:
            if self.E_b == 0:
                self.s_0 = 0
            else:
                self.s_0 = self.tau_bar / self.E_b
            self.omega_fn.s_0 = self.s_0

    omega_fn_type = Trait('li',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn,
                               FRP=FRPDamageFn,
                               ),
                          MAT=True,
                          )

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_(s_0=self.s_0)

    omega_fn = Instance(IDamageFn,
                        report=True)

    def _omega_fn_default(self):
        # return JirasekDamageFn()
        return LiDamageFn(alpha_1=1.,
                          alpha_2=100.
                          )

    state_arr_shape = Tuple((8,))

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def get_cp(self, s, d_s, t_n, t_n1, state):
        tau, s_p, alpha, z, kappa, omega = state
        tau, D, s_p, alpha, z, kappa, omega = \
            self.get_corr_pred(s, d_s, tau, t_n, t_n1,
                               s_p, alpha, z, kappa, omega)
        return tau, D, state

    def get_corr_pred2(self, s_n1, d_s, t_n, t_n1, sa_n):

        tau_n, s_p_n, alpha_n, z_n, kappa_n, omega_n = \
            sa_n[..., 0:3], sa_n[..., 3], sa_n[..., 4],\
            sa_n[..., 5], sa_n[..., 6], sa_n[..., 7]

        tau, D, s_p, alpha, z, kappa, omega = \
            self.get_corr_pred(s_n1, d_s, tau_n, t_n, t_n1,
                               s_p_n, alpha_n, z_n, kappa_n, omega_n)

        sa_n1 = np.concatenate([tau,
                                s_p[..., np.newaxis],
                                alpha[..., np.newaxis],
                                z[..., np.newaxis],
                                kappa[..., np.newaxis],
                                omega[..., np.newaxis]], axis=-1)
        return tau, D, sa_n1

    def get_corr_pred(self, s_n1, d_s, tau_n, t_n, t_n1,
                      s_p_n, alpha_n, z_n, kappa_n, omega_n):

        n_e, n_ip, n_s = s_n1.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        sig_pi_trial = self.E_b * (s_n1[:, :, 1] - s_p_n)

        Z = self.K * z_n

        # for handeling the negative values of isotropic hardening
        h_1 = self.tau_bar + Z
        pos_iso = h_1 > 1e-6

        X = self.gamma * alpha_n

        # for handeling the negative values of kinematic hardening (not yet)
        # h_2 = h * np.sign(sig_pi_trial - X) * \
        #    np.sign(sig_pi_trial) + X * np.sign(sig_pi_trial)
        #pos_kin = h_2 > 1e-6

        f = np.fabs(sig_pi_trial - X) - h_1 * pos_iso

        elas = f <= 1e-6
        plas = f > 1e-6

#         d_tau = np.einsum('...st,...t->...s', D, d_s)
#         tau += d_tau

        # @todo: change this to the calculation that does not need tau as input
        # -- tau is a derived variable.
#         print s.shape
#         print s_p.shape
        tau = np.einsum('...st,...t->...s', D, s_n1)

        # Return mapping
        delta_lamda = f / (self.E_b + self.gamma + np.fabs(self.K)) * plas
        # update all the state variables

        s_p_n1 = s_p_n + delta_lamda * np.sign(sig_pi_trial - X)
        z_n1 = z_n + delta_lamda
        alpha_n1 = alpha_n + delta_lamda * np.sign(sig_pi_trial - X)

        kappa_n1 = np.max(np.array([kappa_n, np.fabs(s_n1[:, :, 1])]), axis=0)
        omega_n1 = self.omega(kappa_n1)

        tau[:, :, 1] = (1 - omega_n1) * self.E_b * (s_n1[:, :, 1] - s_p_n1)

        domega_ds = self.omega_derivative(kappa_n1)

        # Consistent tangent operator
        D_ed = -self.E_b / (self.E_b + self.K + self.gamma) \
            * domega_ds * self.E_b * (s_n1[:, :, 1] - s_p_n1) \
            + (1 - omega_n1) * self.E_b * (self.K + self.gamma) / \
            (self.E_b + self.K + self.gamma)

        D[:, :, 1, 1] = (1 - omega_n1) * self.E_b * elas + D_ed * plas

        return tau, D, s_p_n1, alpha_n1, z_n1, kappa_n1, omega_n1

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

    node_name = "multilinear bond law"

    def __init__(self, *args, **kw):
        super(MATSBondSlipMultiLinear, self).__init__(*args, **kw)
        self.bs_law.replot()

    state_arr_shape = Tuple((0,))

    E_m = Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True, unit='MPa', symbol='$E_\mathrm{m}$',
                desc='E-modulus of the matrix',
                auto_set=True, enter_set=True)

    E_f = Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True, unit='MPa', symbol='$E_\mathrm{f}$',
                desc='E-modulus of the reinforcement',
                auto_set=False, enter_set=True)

    s_data = Str('', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='$s$',
                 desc='slip values',
                 auto_set=True, enter_set=False)

    tau_data = Str('', tooltip='Comma-separated list of stress values',
                   MAT=True, unit='MPa', symbol='$\\tau$',
                   desc='shear stress values',
                   auto_set=True, enter_set=False)

    update_bs_law = Button(label='update bond-slip law')

    def _update_bs_law_fired(self):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        tau_data = np.fromstring(self.tau_data, dtype=np.float_, sep=',')
        if len(s_data) != len(tau_data):
            raise ValueError, 's array and tau array must have the same size'
        self.bs_law.set(xdata=s_data,
                        ydata=tau_data)
        self.bs_law.replot()

    bs_law = Instance(MFnLineArray)

    def _bs_law_default(self):
        return MFnLineArray(
            xdata=[0.0, 1.0],
            ydata=[0.0, 1.0],)

    n_s = Constant(5)

    def get_corr_pred2(self, s_n1, d_s, t_n, t_n1, sa_n):

        sign_s = np.sign(s_n1)
        n_e, n_ip, n_s = s_n1.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        tau = np.einsum('...st,...t->...s', D, s_n1)
        s = s_n1[:, :, 1]
        shape = s.shape
        tau[:, :, 1] = self.bs_law(s.flatten()).reshape(*shape)

        D_tau = self.bs_law.diff(s.flatten()).reshape(*shape)

        D[:, :, 1, 1] = D_tau

        return tau, D, sa_n

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

    def write_figure(self, f, rdir, rel_path):
        fname = 'fig_' + self.node_name.replace(' ', '_') + '.pdf'
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % join(rel_path, fname))
        self.bs_law.replot()
        self.bs_law.savefig(join(rdir, fname))

    tree_view = View(
        VGroup(
            VGroup(
                Item('E_m', full_size=True, resizable=True),
                Item('E_f'),
                Item('s_data'),
                Item('tau_data'),
                UItem('update_bs_law')
            ),
            UItem('bs_law@')
        )
    )


class MATSBondSlipFRPDamage(MATSEval, BMCSTreeNode):

    node_name = 'bond model: FRP damage model'

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

    omega_fn_type = Trait('FRP',
                          dict(FRP=FRPDamageFn),
                          MAT=True,)

    @on_trait_change('omega_fn_type')
    def _reset_omega_fn(self):
        self.omega_fn = self.omega_fn_type_()

    omega_fn = Instance(IDamageFn,
                        # MAT=True, - not a parameter
                        report=True
                        )

    def _omega_fn_default(self):
        # return JirasekDamageFn()
        return FRPDamageFn(b=10.4,
                           Gf=1.19
                           )

    state_arr_shape = Tuple((8,))

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    def get_cp(self, s, d_s, t_n, t_n1, state):
        tau, s_p, alpha, z, kappa, omega = state
        tau, D, s_p, alpha, z, kappa, omega = \
            self.get_corr_pred(s, d_s, tau, t_n, t_n1,
                               s_p, alpha, z, kappa, omega)
        return tau, D, state

    def get_corr_pred2(self, s_n1, d_s, t_n, t_n1, sa_n):

        tau_n, s_p_n, alpha_n, z_n, kappa_n, omega_n = \
            sa_n[..., 0:3], sa_n[..., 3], sa_n[..., 4],\
            sa_n[..., 5], sa_n[..., 6], sa_n[..., 7]

        tau, D, s_p, alpha, z, kappa, omega = \
            self.get_corr_pred(s_n1, d_s, tau_n, t_n, t_n1,
                               s_p_n, alpha_n, z_n, kappa_n, omega_n)

        sa_n1 = np.concatenate([tau,
                                s_p[..., np.newaxis],
                                alpha[..., np.newaxis],
                                z[..., np.newaxis],
                                kappa[..., np.newaxis],
                                omega[..., np.newaxis]], axis=-1)
        return tau, D, sa_n1

    n_s = Constant(5)

    def get_corr_pred(self, s_n1, d_s, tau_n, t_n, t_n1,
                      s_p_n, alpha_n, z_n, kappa_n, omega_n):

        n_e, n_ip, n_s = s_n1.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f

        sig_pi_trial = self.omega_fn.E_b * (s_n1[:, :, 1])

        tau = np.einsum('...st,...t->...s', D, s_n1)

        # update all the state variables
        s_p_n1 = s_p_n
        z_n1 = z_n
        alpha_n1 = alpha_n

        kappa_n1 = np.max(np.array([kappa_n, np.fabs(s_n1[:, :, 1])]), axis=0)
        omega_n1 = self.omega(kappa_n1)

        tau[:, :, 1] = (1 - omega_n1) * \
            self.omega_fn.E_b * (s_n1[:, :, 1])

        domega_ds = self.omega_derivative(kappa_n1)

        #s_0 = self.omega_fn.s_0

        elas = omega_n1 <= 0.0
        plas = omega_n1 > 0.0

        # Consistent tangent operator
        D_ed = ((1 - omega_n1) - domega_ds * s_n1[:, :, 1]) * self.omega_fn.E_b

        # * elas + D_ed * plas
        D[:, :, 1, 1] = (1 - omega_n1) * self.omega_fn.E_b

        return tau, D, s_p_n1, alpha_n1, z_n1, kappa_n1, omega_n1

    tree_view = View(
        VGroup(
            VGroup(
                Item('E_m', full_size=True, resizable=True),
                Item('E_f'),
            ),
            UItem('omega_fn@')
        )
    )


if __name__ == '__main__':
    #m = MATSBondSlipMultiLinear()
    m = MATSBondSlipDP()
    m.configure_traits()

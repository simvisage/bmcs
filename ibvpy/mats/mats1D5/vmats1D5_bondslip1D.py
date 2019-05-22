'''
Created on 05.12.2016

@author: abaktheer
'''

from os.path import join

from ibvpy.api import MATSEval
from ibvpy.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn, \
    MultilinearDamageFn, \
    FRPDamageFn
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from simulator.api import \
    TLoopImplicit, TStepBC
from traits.api import  \
    Float, Tuple, List, on_trait_change, \
    Instance, Trait, Bool, Str, Button, Property

import numpy as np
import traitsui.api as ui


class MATSBondSlipMultiLinear(MATSEval):

    node_name = "multilinear bond law"

    # To use the model directly in the simulator specify the
    # time stepping classes
    tloop_type = TLoopImplicit
    tstep_type = TStepBC

    def __init__(self, *args, **kw):
        super(MATSBondSlipMultiLinear, self).__init__(*args, **kw)
        self.bs_law.replot()

    state_arr_shape = Tuple((0,))

    E_m = Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{m}',
                desc='E-modulus of the matrix',
                auto_set=True, enter_set=True)

    E_f = Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{f}',
                desc='E-modulus of the reinforcement',
                auto_set=False, enter_set=True)

    s_data = Str('', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='s',
                 desc='slip values',
                 auto_set=True, enter_set=False)

    tau_data = Str('', tooltip='Comma-separated list of stress values',
                   MAT=True, unit='MPa', symbol=r'\tau',
                   desc='shear stress values',
                   auto_set=True, enter_set=False)

    s_tau_table = Property

    def _set_s_tau_table(self, data):
        s_data, tau_data = data
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        self.bs_law.set(xdata=s_data,
                        ydata=tau_data)

    update_bs_law = Button(label='update bond-slip law')

    def _update_bs_law_fired(self):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        tau_data = np.fromstring(self.tau_data, dtype=np.float_, sep=',')
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        self.bs_law.set(xdata=s_data,
                        ydata=tau_data)
        self.bs_law.replot()

    bs_law = Instance(MFnLineArray)

    def _bs_law_default(self):
        return MFnLineArray(
            xdata=[0.0, 1.0],
            ydata=[0.0, 1.0],
            plot_diff=False
        )

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    U_var_shape = (1,)
    '''Shape of the primary variable required by the TStepState.
    '''

    state_var_shapes = {}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    node_name = 'multiply_linear bond'

    def get_corr_pred(self, s, t_n1):

        n_e, n_ip, _ = s.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        tau = np.einsum('...st,...t->...s', D, s)
        s = s[..., 1]
        shape = s.shape
        signs = np.sign(s.flatten())
        s_pos = np.fabs(s.flatten())
        tau[..., 1] = (signs * self.bs_law(s_pos)).reshape(*shape)
        D_tau = self.bs_law.diff(s_pos).reshape(*shape)
        D[..., 1, 1] = D_tau

        return tau, D

    def write_figure(self, f, rdir, rel_path):
        fname = 'fig_' + self.node_name.replace(' ', '_') + '.pdf'
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % join(rel_path, fname))
        self.bs_law.replot()
        self.bs_law.savefig(join(rdir, fname))

    def plot(self, ax, **kw):
        ax.plot(self.bs_law.xdata, self.bs_law.xdata, **kw)

    tree_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('E_m', full_size=True, resizable=True),
                ui.Item('E_f'),
                ui.Item('s_data'),
                ui.Item('tau_data'),
                ui.UItem('update_bs_law')
            ),
            ui.UItem('bs_law@')
        )
    )


class MATSBondSlipD(MATSEval):

    node_name = 'bond model: FRP damage model'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    def __init__(self, *args, **kw):
        super(MATSBondSlipD, self).__init__(*args, **kw)
        self._omega_fn_type_changed()

    E_m = Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(200000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = Float(10000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    omega_fn_type = Trait('FRP',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn,
                               FRP=FRPDamageFn,
                               multilinear=MultilinearDamageFn
                               ),
                          MAT=True,
                          )

    def _omega_fn_type_changed(self):
        self.omega_fn = self.omega_fn_type_(mats=self)

    omega_fn = Instance(IDamageFn,
                        report=True)

    def _omega_fn_default(self):
        return MultilinearDamageFn()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    state_var_shapes = dict(kappa_n=(), omega_n=())

    def get_corr_pred(self, eps_n1, t_n1, kappa_n, omega_n):
        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f
        s_n1 = eps_n1[..., 1]
        kappa_n[...] = np.max(np.array([kappa_n, np.fabs(s_n1)]), axis=0)
        omega_n[...] = self.omega(kappa_n)
        tau = np.einsum('...st,...t->...s', D, eps_n1)
        tau[..., 1] = (1 - omega_n) * self.E_b * s_n1
        domega_ds = self.omega_derivative(kappa_n)
        D[..., 1, 1] = ((1 - omega_n) - domega_ds * s_n1) * self.E_b
        return tau, D

    tree_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('E_m', full_size=True, resizable=True),
                ui.Item('E_f'),
                ui.Item('E_b'),
            ),
            ui.VGroup(
                ui.Item('omega_fn_type'),
            ),
            ui.UItem('omega_fn@')
        )
    )


class MATSBondSlipDP(MATSEval):

    node_name = 'bond model: damage-plasticity'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    E_m = Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                symbol='E_\mathrm{m}',
                unit='MPa',
                desc='Stiffness of the matrix',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(200000.0, tooltip='Stiffness of the reinforcement [MPa]',
                symbol='E_\mathrm{f}',
                unit='MPa',
                desc='Stiffness of the reinforcement',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = Float(12900.0,
                symbol="E_\mathrm{b}",
                unit='MPa',
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = Float(100.0,
                  symbol="\gamma",
                  unit='MPa',
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = Float(1000.0,
              symbol="K",
              unit='MPa',
              desc="Isotropic hardening modulus",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5.0,
                    symbol=r'\bar{\tau}',
                    unite='MPa',
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
        self._omega_fn_type_changed()
        self._update_s0()

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        if not self.uncoupled_dp:
            if self.E_b == 0:
                self.s_0 = 0
            else:
                self.s_0 = self.tau_bar / self.E_b
            self.omega_fn.s_0 = self.s_0

    omega_fn_type = Trait('multilinear',
                          dict(li=LiDamageFn,
                               jirasek=JirasekDamageFn,
                               abaqus=AbaqusDamageFn,
                               FRP=FRPDamageFn,
                               multilinear=MultilinearDamageFn
                               ),
                          MAT=True,
                          )

    def _omega_fn_type_changed(self):
        self.omega_fn = self.omega_fn_type_(mats=self, s_0=self.s_0)

    omega_fn = Instance(IDamageFn,
                        report=True)

    def _omega_fn_default(self):
        return MultilinearDamageFn()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    state_var_shapes = dict(s_p_n=(),
                            alpha_n=(),
                            z_n=(),
                            kappa_n=(),
                            omega_n=())

    def get_corr_pred(self, eps_n1, t_n1,
                      s_p_n, alpha_n, z_n, kappa_n, omega_n):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_n1 = eps_n1[..., 1]

        sig_pi_trial = self.E_b * (s_n1 - s_p_n)

        Z = self.K * z_n

        # for handeling the negative values of isotropic hardening
        h_1 = self.tau_bar + Z
        pos_iso = h_1 > 1e-6

        X = self.gamma * alpha_n

        # for handeling the negative values of kinematic hardening (not yet)
        # h_2 = h * np.sign(sig_pi_trial - X) * \
        #    np.sign(sig_pi_trial) + X * np.sign(sig_pi_trial)
        #pos_kin = h_2 > 1e-6

        f_trial = np.fabs(sig_pi_trial - X) - h_1 * pos_iso

        I = f_trial > 1e-6

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        # Return mapping
        delta_lamda_I = f_trial[I] / (self.E_b + self.gamma + np.fabs(self.K))

        # update all the state variables
        s_p_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        z_n[I] += delta_lamda_I
        alpha_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])

        kappa_n[I] = np.max(
            np.array([kappa_n[I], np.fabs(s_n1[I])]), axis=0)
        omega_n[I] = self.omega(kappa_n[I])

        tau[..., 1] = (1 - omega_n) * self.E_b * (s_n1 - s_p_n)

        domega_ds_I = self.omega_derivative(kappa_n[I])

        # Consistent tangent operator
        D_ed_I = -self.E_b / (self.E_b + self.K + self.gamma) \
            * domega_ds_I * self.E_b * (s_n1[I] - s_p_n[I]) \
            + (1 - omega_n[I]) * self.E_b * (self.K + self.gamma) / \
            (self.E_b + self.K + self.gamma)

        D[..., 1, 1] = (1 - omega_n) * self.E_b
        D[I, 1, 1] = D_ed_I

        return tau, D

    tree_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('E_m', full_size=True, resizable=True),
                ui.Item('E_f'),
                ui.Item('E_b'),
                ui.Item('gamma'),
                ui.Item('K'),
                ui.Item('tau_bar'),
            ),
            ui.VGroup(
                ui.Item('uncoupled_dp'),
                ui.Item('s_0'),  # , enabled_when='uncoupled_dp'),
                ui.Item('omega_fn_type'),
            ),
            ui.UItem('omega_fn@')
        )
    )


class MATSBondSlipEP(MATSEval):

    node_name = 'bond model: elasto-plasticity'

    E_m = Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                symbol='E_\mathrm{m}',
                unit='MPa',
                desc='Stiffness of the matrix',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = Float(200000.0, tooltip='Stiffness of the reinforcement [MPa]',
                symbol='E_\mathrm{f}',
                unit='MPa',
                desc='Stiffness of the reinforcement',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = Float(12900.0,
                symbol="E_\mathrm{b}",
                unit='MPa',
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = Float(100.0,
                  symbol="\gamma",
                  unit='MPa',
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = Float(1000.0,
              symbol="K",
              unit='MPa',
              desc="Isotropic hardening modulus",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = Float(5.0,
                    symbol=r'\bar{\tau}',
                    unite='MPa',
                    desc="Reversibility limit",
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    state_var_shapes = dict(s_p_n=(),
                            alpha_n=(),
                            z_n=())

    def get_corr_pred(self, eps_n1, t_n1,
                      s_p_n, alpha_n, z_n):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_n1 = eps_n1[..., 1]

        sig_pi_trial = self.E_b * (s_n1 - s_p_n)

        Z = self.K * z_n

        # for handling the negative values of isotropic hardening
        h_1 = self.tau_bar + Z
        pos_iso = h_1 > 1e-6

        X = self.gamma * alpha_n

        # for handling the negative values of kinematic hardening (not yet)
#         h_2 = h * np.sign(sig_pi_trial - X) * \
#             np.sign(sig_pi_trial) + X * np.sign(sig_pi_trial)
#         pos_kin = h_2 > 1e-6

        f_trial = np.fabs(sig_pi_trial - X) - h_1 * pos_iso

        I = f_trial > 1e-6

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        # Return mapping
        delta_lamda_I = f_trial[I] / (self.E_b + self.gamma + np.fabs(self.K))

        # update all the state variables
        s_p_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        z_n[I] += delta_lamda_I
        alpha_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])

        tau[..., 1] = self.E_b * (s_n1 - s_p_n)

        # Consistent tangent operator
        D_ed_I = (self.E_b * (self.K + self.gamma) /
                  (self.E_b + self.K + self.gamma)
                  )

        D[..., 1, 1] = self.E_b
        D[I, 1, 1] = D_ed_I

        return tau, D

    tree_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('E_m', full_size=True, resizable=True),
                ui.Item('E_f'),
                ui.Item('E_b'),
                ui.Item('gamma'),
                ui.Item('K'),
                ui.Item('tau_bar'),
            ),
        )
    )


class MATSBondSlipFatigue(MATSEval):

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

    state_var_shapes = dict(xs_pi=(),
                            alpha=(),
                            z=(),
                            kappa=(),
                            omega=()
                            )

    def get_corr_pred(self, eps_n1, t_n1, xs_pi, alpha, z, kappa, omega):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_n1 = eps_n1[..., 1]

        tau_pi_trial = self.E_b * (s_n1 - xs_pi)
        X = self.gamma * alpha
        Z = self.K * z

        f_trial = np.fabs(tau_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * self.pressure / 3

        I = np.where(f_trial > 1e-6)

        sig = np.einsum('...st,...t->...s', D, eps_n1)
        sig[..., 1] = tau_pi_trial

        omega_I = omega[I]

        # Return mapping
        delta_lamda_I = f_trial[I] / \
            (self.E_b / (1 - omega_I) + self.gamma + self.K)

        # update all the state variables
        xs_pi[I] += (delta_lamda_I *
                     np.sign(tau_pi_trial[I] - X[I]) / (1 - omega_I))
        z[I] += delta_lamda_I
        alpha[I] += delta_lamda_I * np.sign(tau_pi_trial[I] - X[I])

        Y_I = 0.5 * self.E_b * (s_n1[I] - xs_pi[I]) ** 2
        omega[I] += (
            delta_lamda_I *
            (1 - omega_I) ** self.c *
            (Y_I / self.S) ** self.r
        )
        sig[..., 1] = (1 - omega) * self.E_b * (s_n1 - xs_pi)
        omega_I = omega[I]
        O = np.where(np.fabs(1. - omega_I) > 1e-5)
        IO = tuple([I[o][O] for o in range(len(I))])
        omega_IO = omega[IO]
        D_ed_IO = (
            self.E_b * (1 - omega_IO) - ((1 - omega_IO) * self.E_b **
                                         2) / (self.E_b + (self.gamma + self.K) * (1 - omega_IO))
            - ((1 - omega_IO) ** self.c * (self.E_b ** 2) * ((Y_I[O] / self.S) ** self.r)
               * np.sign(tau_pi_trial[I][O] - X[I][O]) * (s_n1[I][O] - xs_pi[I][O])) / ((self.E_b / (1 - omega_IO)) + self.gamma + self.K)
        )
        D[..., 1, 1] = (1 - omega) * self.E_b
        IO11 = IO + (1, 1)
        D[IO11] = D_ed_IO

        return sig, D

    tree_view = ui.View(
        ui.VGroup(ui.Group(ui.Item('E_m'),
                           ui.Item('E_f'),
                           ui.Item('E_b'),
                           ui.Item('tau_pi_bar'),
                           show_border=True, label='Stiffnesses and reversibility limit'),
                  ui.Group(ui.Item('gamma'),
                           ui.Item('K'),
                           show_border=True, label='Hardening parameters'),
                  ui.Group(ui.Item('S'),
                           ui.Item('r'),
                           ui.Item('c'),
                           show_border=True, label='Damage cumulation parameters'),
                  ui.Group(ui.Item('pressure'),
                           ui.Item('a'), show_border=True, label='Lateral Pressure')))


def plot_stress_stress(m, s_max):
    s_r = np.linspace(0, s_max, 60)
    eps_m = np.zeros_like(s_r)
    eps_f = np.zeros_like(s_r)
    u_r = np.c_[eps_m, s_r, eps_f].reshape(60, -1, 3)
    state_arr = {var: np.zeros((1,) + var_shape, dtype=np.float_)
                 for var, var_shape in m.state_var_shapes.items()}
    tau_D = [
        m.get_corr_pred(u, 0.0, **state_arr)
        for u in u_r
    ]
    sig = np.array([tau for tau, _ in tau_D])
    D = np.array([D for _, D in tau_D])

    D1 = D[..., 0, 1, 1]
    tau = sig[..., 0, 1]
    delta_s = s_max / 100
    delta_sig = np.array([tau, tau + D1 * delta_s])
    delta_u = np.array([s_r, s_r + delta_s])

    import pylab as p
    p.plot(s_r, tau)
    p.plot(delta_u, delta_sig, color='red')
    p.show()


if __name__ == '__main__':
    tau_bar = 3.0
    s_max = 0.03
    E_T = 10000
    s_0 = tau_bar / E_T
    m = MATSBondSlipDP(omega_fn_type='jirasek')
    m.omega_fn.trait_set(s_0=s_0, s_f=100 * s_0)
    m = MATSBondSlipDP(omega_fn_type='li')
    m = MATSBondSlipD()
    m.omega_fn_type = 'FRP'
    m.omega_fn.trait_set(
        B=100.4,
        Gf=0.000019
    )
    # Parameters of the next model taken from the paper
    # Baktheer, A. and Chudoba, R., 2018.
    # Pressure-sensitive bond fatigue model with damage evolution
    # driven by cumulative slip: Thermodynamic formulation
    # and applications to steel-and FRP-concrete bond.
    # International Journal of Fatigue, 113, pp.277-289.
    m = MATSBondSlipFatigue(
        E_b=12900,
        tau_pi_bar=4.0,
        K=0.0,
        gamma=10.0,
        S=0.0025,
        r=1,
        c=1)
    plot_stress_stress(m, .2)

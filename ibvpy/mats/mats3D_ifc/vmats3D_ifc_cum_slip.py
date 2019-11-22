
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval, MATSEval
from numpy import array,\
    einsum, zeros_like, identity, sign,\
    sqrt
from traits.api import  \
    provides
from traits.api import Constant,\
    Float, Property, cached_property, Dict
from traitsui.api import \
    View, VGroup, Item

import numpy as np
import traits.api as tr
import traitsui.api as ui


@provides(IMATSEval)
class MATS3DIfcCumSlip(MATSEval):

    node_name = "Cumulative bond-slip law"

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

    state_var_shapes = Property(Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''
    @cached_property
    def _get_state_var_shapes(self):
        return dict(omega_w=(),
                    r_w=(),
                    omega_s=(),
                    z_s=(),
                    alpha_s_a=(2),
                    s_p_a=(2))

    def init(self, *args):
        r'''
        Initialize the state variables.
        '''
        for a in args:
            a[...] = 0

    algorithmic = tr.Bool(False)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------
    def get_corr_pred(self, u_b, t_n1,
                      omega_w, r_w, omega_s,
                      z_s, alpha_s_a, s_p_a):
        w = u_b[..., 0]
        s_a = u_b[..., 1:]
        # For normal - distinguish tension and compression
        T = w > 0.0
        sigma = self.E_n * w
        delta_lambda = 0.5 * self.E_n * (w * T)**2.0 * self.Ad * (1 + r_w)**2
        omega_w += delta_lambda
        r_w -= delta_lambda
        sigma[T] = (1.0 - omega_w[T]) * self.E_n * (w[T])
        # For tangential
        E_s = self.E_s
        tau_trial_a = E_s * (s_a - s_p_a)
        Z_s = self.K_T * z_s
        X_s_a = self.gamma_T * alpha_s_a
        q_a = tau_trial_a - X_s_a
        norm_q = sqrt(einsum('...a,...a->...', q_a, q_a))
        f = norm_q - self.tau_pi_bar - Z_s + self.a * sigma
        plas_1 = f > 1e-6
        elas_1 = f < 1e-6
        delta_lamda = (
            f / (E_s / (1 - omega_s) + self.gamma_T + self.K_T) * plas_1
        )
        norm_q_plas = 1.0 * elas_1 + sqrt(
            einsum('...a,...a->...', q_a, q_a)) * plas_1
        s_p_a += plas_1 * delta_lamda / (1.0 - omega_s) * q_a / norm_q_plas
        s_el_a = s_a - s_p_a
        Y = 0.5 * E_s * \
            einsum('...a,...a->...', s_el_a, s_el_a)
        omega_s += ((1.0 - omega_s) ** self.c_T) * \
            (delta_lamda * (Y / self.S_T) ** self.r_T)  # * \
        #(self.tau_pi_bar / (self.tau_pi_bar - self.a * sigma_kk / 3.0))
        alpha_s_a += plas_1 * delta_lamda * q_a / norm_q_plas
        z_s += delta_lamda
        tau_a = (1 - omega_s) * self.E_T * (s_a - s_p_a) * plas_1
        E_alg_T = (1 - omega_s) * self.E_T
        # Consistent tangent operator
        sig = np.zeros_like(u_b)
        sig[..., 0] = sigma
        sig[..., 1:] = tau_a
        E_TN = np.einsum('abEm->Emab',
                         np.array(
                             [
                                 #                                  [E_alg_N, np.zeros_like(E_alg_T)],
                                 #                                  [np.zeros_like(E_alg_N), E_alg_T]
                             ])
                         )
        return sig, E_TN

    def _get_var_dict(self):
        var_dict = super(MATS3DIfcCumSlip, self)._get_var_dict()
        var_dict.update(
            slip=self.get_slip,
            s_el=self.get_s_el,
            shear=self.get_shear,
            omega=self.get_omega,
            s_pi=self.get_s_pi,
            alpha=self.get_alpha,
            z=self.get_z
        )
        return var_dict

    def get_slip(self, u_r, tn1, **state):
        return self.get_eps(u_r, tn1)[..., 0]

    def get_shear(self, u_r, tn1, **state):
        return self.get_sig(u_r, tn1, **state)[..., 0]

    def get_omega(self, u_r, tn1, s_pi, alpha, z, omega):
        return omega

    def get_s_pi(self, u_r, tn1, s_pi, alpha, z, omega):
        return s_pi

    def get_alpha(self, u_r, tn1, s_pi, alpha, z, omega):
        return alpha

    def get_z(self, u_r, tn1, s_pi, alpha, z, omega):
        return z

    def get_s_el(self, u_r, tn1, **state):
        s = self.get_slip(u_r, tn1, **state)
        s_p = self.get_s_pi(u_r, tn1, **state)
        s_e = s - s_p
        return s_e

    tree_view = ui.View(
        ui.Item('E_N'),
        ui.Item('E_T'),
        ui.Item('gamma'),
        ui.Item('K'),
        ui.Item('S'),
        ui.Item('r'),
        ui.Item('c'),
        ui.Item('m'),
        ui.Item('tau_bar'),
        ui.Item('D_rs', style='readonly')
    )

    traits_view = tree_view

    #---------------------------------------
    # Tangential constitutive law parameters
    #---------------------------------------
    gamma_T = Float(5000.,
                    label="Gamma",
                    desc=" Tangential Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)

    K_T = Float(10.0,
                label="K",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S_T = Float(0.0001,
                label="S",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = Float(1.2,
                label="r",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = Float(1.0,
                label="c",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = Float(2.0,
                       label="Tau_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(0.0,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    #-------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    #-------------------------------------------
    Ad = Float(10000.0,
               label="Ad",
               desc="Brittleness coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = Float(0.0000002,
                  label="eps_0",
                  desc="Threshold strain",
                  enter_set=True,
                  auto_set=False)

    eps_f = Float(0.000002,
                  label="eps_f",
                  desc="Damage function shape",
                  enter_set=True,
                  auto_set=False)

    #-------------------------------------------
    # Normal_tension constitutive law parameters
    # (using cumulative normal plastic strain)
    #-------------------------------------------
    gamma_N_pos = Float(5000.,
                        label="Gamma N",
                        desc=" Tangential Kinematic hardening modulus",
                        enter_set=True,
                        auto_set=False)

    K_N_pos = Float(0.0,
                    label="K N",
                    desc="Tangential Isotropic harening",
                    enter_set=True,
                    auto_set=False)

    S_N = Float(0.005,
                label="S N",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_N = Float(1.0,
                label="r N",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_N = Float(1.0,
                label="c N",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    sigma_0_pos = Float(2.0,
                        label="sigma_0",
                        desc="Reversibility limit",
                        enter_set=True,
                        auto_set=False)

    #-----------------------------------------------
    # Normal_Compression constitutive law parameters
    #-----------------------------------------------
    K_N_neg = Float(10000.,
                    label="K N compression",
                    desc=" Normal isotropic harening",
                    enter_set=True,
                    auto_set=False)

    gamma_N_neg = Float(15000.,
                        label="gamma_compression",
                        desc="Normal kinematic hardening",
                        enter_set=True,
                        auto_set=False)

    sigma_0_neg = Float(20.,
                        label="sigma 0 compression",
                        desc="Yield stress in compression",
                        enter_set=True,
                        auto_set=False)

    traits_view = View(
        VGroup(
            VGroup(
                Item('E_s', full_size=True, resizable=True),
                Item('E_n', full_size=True, resizable=True),
                Item('nu'),
                label='Elastic parameters'
            ),
            VGroup(
                Item('gamma_T', full_size=True, resizable=True),
                Item('K_T'),
                Item('S_T'),
                Item('r_T'),
                Item('c_T'),
                Item('tau_pi_bar'),
                Item('a'),
                label='Tangential properties'
            ),
            VGroup(
                Item('Ad'),
                Item('eps_0', full_size=True, resizable=True),
                Item('eps_f'),
                label='Normal_Tension (no cumulative normal strain)'
            ),
            VGroup(
                Item('gamma_N_pos', full_size=True, resizable=True),
                Item('K_N_pos'),
                Item('S_N'),
                Item('r_N'),
                Item('c_N'),
                Item('sigma_0_pos'),
                label='Normal_Tension (cumulative normal plastic strain)',
            ),
            VGroup(
                Item('K_N_neg', full_size=True, resizable=True),
                Item('gamma_N_neg'),
                Item('sigma_0_neg'),
                label='Normal_compression parameters'
            )
        )
    )
    tree_view = traits_view

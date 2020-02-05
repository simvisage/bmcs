
from ibvpy.mats.mats_eval import \
    IMATSEval, MATSEval
from traitsui.api import \
    View, VGroup, Item

import numpy as np

from traits.api import Constant,\
    Float, Property, cached_property, Dict, provides, Bool


@provides(IMATSEval)
class MATS3DIfcCumSlip(MATSEval):

    node_name = "Cumulative bond-slip law"

    state_var_shapes = dict(
        omega_N=(),
        r_N=(),
        omega_T=(),
        z_T=(),
        alpha_T_a=(2,),
        s_p_a=(2,)
    )

    def init(self, *args):
        r'''
        Initialize the state variables.
        '''
        for a in args:
            a[...] = 0

    algorithmic = Bool(False)

    def get_corr_pred(self, u_b, t_n1, omega_N, r_N, omega_T,
                      z_T, alpha_T_a, s_p_a):

        w = u_b[..., 0]
        s_a = u_b[..., 1:]
        # For normal - distinguish tension and compression
        T = w > 0.0
        H_w_N = np.array(w <= 0.0, dtype=np.float_)
        E_alg_N = H_w_N * self.E_N
        sig_N = self.E_N * w
        delta_lambda = 0.5 * self.E_N * (w * T)**2.0 * self.Ad * (1 + r_N)**2
        omega_N += delta_lambda
        r_N -= delta_lambda
        sig_N[T] = (1.0 - omega_N[T]) * self.E_N * (w[T])

        E_T = self.E_T

        tau_a = E_T * (s_a - s_p_a)
        Z = self.K_T * z_T
        X_a = self.gamma_T * alpha_T_a
        q_a = tau_a - X_a
        norm_q = np.sqrt(np.einsum('...na,...na->...n', q_a, q_a))
        f_trial = norm_q - self.tau_pi_bar - Z + self.a * sig_N

        I = np.where(f_trial > 1e-6)

        omega_T_I = omega_T[I]
        q_a_I = q_a[I]
        norm_q_I = norm_q[I]
        delta_lambda_I = f_trial[I] / \
            (E_T / (1.0 - omega_T_I) + self.gamma_T + self.K_T)

        s_p_a[I, :] += (
            delta_lambda_I / (1.0 - omega_T_I) / norm_q_I
        )[:, np.newaxis] * q_a_I

        s_el_a_I = s_a[I] - s_p_a[I]
        Y_I = 0.5 * E_T * np.einsum('...a,...a->...', s_el_a_I, s_el_a_I)

        omega_T[I] += (
            (1.0 - omega_T[I]) ** self.c_T *
            delta_lambda_I * (Y_I / self.S_T) ** self.r_T
        )
        # * \
        #(self.tau_pi_bar / (self.tau_pi_bar - self.a * sig_N)
        alpha_T_a[I, :] += (
            (delta_lambda_I / norm_q_I)[:, np.newaxis] * q_a_I
        )
        z_T += delta_lambda

        tau_a[I, :] = (
            ((1 - omega_T[I]) * self.E_T)[:, np.newaxis] * s_el_a_I
        )
        # Secant stiffness
        E_alg_T = ((1 - omega_T) *
                   self.E_T)[:, np.newaxis, np.newaxis]
        # Consistent tangent operator
        sig = np.zeros_like(u_b)
        sig[..., 0] = sig_N
        sig[..., 1:] = tau_a
        N_ab = np.zeros((3, 3), dtype=np.float_)
        N_ab[0, 0] = 1
        T_ab = np.zeros((3, 3), dtype=np.float_)
        T_ab[(1, 2), (1, 2)] = 1
        E_NT = (np.einsum('ab,...->...ab', N_ab, E_alg_N) +
                np.einsum('ab,...ab->...ab', T_ab, E_alg_T))

        tau_a = E_T * (s_a - s_p_a)
        Z = self.K_T * z_T
        X_a = self.gamma_T * alpha_T_a
        q_a = tau_a - X_a
        norm_q = np.sqrt(np.einsum('...na,...na->...n', q_a, q_a))
        f_trial = norm_q - self.tau_pi_bar - Z + self.a * sig_N
        print('f_trial', f_trial[np.where(f_trial > 1e-6)])

        print('sig', sig)
        return sig, E_NT

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

    #---------------------------------------
    # Tangential constitutive law parameters
    #---------------------------------------
    E_T = Float(100, label='E_T',
                desc='Shear modulus of the interface',
                MAT=True,
                enter_set=True, auto_set=False)

    gamma_T = Float(5000.,
                    label="Gamma_T",
                    desc=" Tangential Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)

    K_T = Float(10.0,
                label="K_T",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S_T = Float(0.0001,
                label="S_T",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = Float(1.2,
                label="r_T",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = Float(1.0,
                label="c_T",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = Float(2.0,
                       label="Tau_bar_T",
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
    E_N = Float(100, label='E_N',
                desc='Normal stiffness of the interface',
                MAT=True,
                enter_set=True, auto_set=False)

    Ad = Float(10000.0,
               label="Ad",
               desc="Brittleness coefficient",
               enter_set=True,
               auto_set=False)

    traits_view = View(
        VGroup(
            VGroup(
                Item('E_T', full_size=True, resizable=True),
                Item('E_N', full_size=True, resizable=True),
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
                label='Normal_Tension (no cumulative normal strain)'
            ),
        )
    )
    tree_view = traits_view

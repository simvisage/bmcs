'''
Created on Jan 19, 2020

@author: rch

Test the case of a straight crack in the middle of a zone.
'''

from matplotlib.pyplot import tight_layout
from scipy.interpolate import interp1d
from scipy.optimize import root

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import traits.api as tr
import traitsui.api as ui
from .sz_rotation_kinematics import get_phi

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1
Z = np.array([0, 0, 1], dtype=np.float_)


class IMaterialModel(tr.Interface):

    get_sig_w = tr.Property(depends_on='+Param')
    get_tau_s = tr.Property(depends_on='+Param')


eps, f_c, E_c, L_c = sp.symbols(
    'epsilon, f_c, E_c, L_c'
)

gamma, E_s = sp.symbols(
    'gamma, E_s'
)

w, f_t, G_f, L = sp.symbols(
    'w, f_t, G_f, L'
)

s, tau_1, s_1, tau_2, s_2, tau_3, s_3 = sp.symbols(
    r's, tau_1, s_1, tau_2, s_2, tau_3, s_3 '
)

#=========================================================================
# Unkcracked concrete
#=========================================================================

sig_eps = sp.Piecewise(
    (f_c, E_c * eps < f_c),
    (E_c * eps, E_c * eps <= f_t),
    (f_t, E_c * eps > f_t)
)

d_sig_eps = sig_eps.diff(eps)

#=========================================================================
# Crack opening law
#=========================================================================

w_t = f_t / E_c * L_c

f_w = f_t * sp.exp(-f_t * (w - w_t) / G_f)

sig_w = sp.Piecewise(
    (f_c, E_c * w / L_c < f_c),
    (E_c * w / L_c, w <= w_t),
    (f_w, w > w_t)
)

d_sig_w = sig_w.diff(w)

#=========================================================================
# Bond-slip law
#=========================================================================

tau_s = sp.Piecewise(
    (tau_1 / s_1 * s, s < s_1),
    (tau_1 + (tau_2 - tau_1) / (s_2 - s_1) * (s - s_1), s <= s_2),
    (tau_2 + (tau_3 - tau_2) / (s_3 - s_2) * (s - s_2), s > s_2)
)
d_tau_s = tau_s.diff(s)


@tr.provides(IMaterialModel)
class MaterialModel(tr.HasStrictTraits):

    f_c = tr.Float(-80.0, PARAM=True)
    E_c = tr.Float(28000, PARAM=True)

    f_t = tr.Float(3.0, PARAM=True)
    G_f = tr.Float(0.5, PARAM=True)
    S_max = tr.Float(1e+9, PARAM=True)

    L_c = tr.Property

    def _get_L_c(self):
        return self.E_c * self.G_f / self.f_t**2

    L = tr.Float(100, PARAM=True)
    co_law_data = tr.Property(depends_on='+PARAM')

    @tr.cached_property
    def _get_co_law_data(self):
        return dict(f_t=float(self.f_t),
                    G_f=float(self.G_f),
                    S_max=float(self.S_max),
                    f_c=self.f_c,
                    E_c=self.E_c,
                    L_c=self.L_c,
                    L=self.L)

    get_sig_eps = tr.Property(depends_on='+Param')

    @tr.cached_property
    def _get_get_sig_eps(self):
        return sp.lambdify(eps, sig_eps.subs(self.co_law_data), 'numpy')

    get_d_sig_eps = tr.Property(depends_on='+Param')

    @tr.cached_property
    def _get_get_d_sig_eps(self):
        return sp.lambdify(eps, d_sig_eps.subs(self.co_law_data), 'numpy')

    #=========================================================================
    # Sig w
    #=========================================================================
    get_sig_w = tr.Property(depends_on='+Param')

    @tr.cached_property
    def _get_get_sig_w(self):
        return sp.lambdify(w, sig_w.subs(self.co_law_data), 'numpy')

    get_d_sig_w = tr.Property(depends_on='+Param')

    @tr.cached_property
    def _get_get_d_sig_w(self):
        return sp.lambdify(w, d_sig_w.subs(self.co_law_data), 'numpy')

    #=========================================================================
    #
    #=========================================================================
    tau_1 = tr.Float(1.0, PARAM=True)
    s_1 = tr.Float(0.000001, PARAM=True)
    tau_2 = tr.Float(1.0, PARAM=True)
    s_2 = tr.Float(0.02, PARAM=True)
    tau_3 = tr.Float(0.0, PARAM=True)
    s_3 = tr.Float(1.6, PARAM=True)

    bond_law_data = tr.Property(depends_on='+PARAM')

    @tr.cached_property
    def _get_bond_law_data(self):
        return dict(tau_1=self.tau_1, s_1=self.s_1,
                    tau_2=self.tau_2, s_2=self.s_2,
                    tau_3=self.tau_3, s_3=self.s_3)

    get_tau_s_plus = tr.Property(depends_on='+PARAM')

    @tr.cached_property
    def _get_get_tau_s_plus(self):
        return sp.lambdify(s, tau_s.subs(self.bond_law_data), 'numpy')

    get_d_tau_s_plus = tr.Property(depends_on='+PARAM')

    @tr.cached_property
    def _get_get_d_tau_s_plus(self):
        return sp.lambdify(s, d_tau_s.subs(self.bond_law_data), 'numpy')

    def get_tau_s(self, s):
        signs = np.sign(s)
        return signs * self.get_tau_s_plus(signs * s)

    def get_d_tau_s(self, s):
        signs = np.sign(s)
        return signs * self.get_d_tau_s_plus(signs * s)

    #=========================================================================
    # Plotting
    #=========================================================================

    def plot_sig_eps(self, ax1, ax2):
        eps_min = (f_c / E_c * 2).subs(self.co_law_data)
        eps_max = (f_t / E_c * 2).subs(self.co_law_data)
        eps_data = np.linspace(float(eps_min), float(eps_max), 100)
        ax1.plot(eps_data, self.get_sig_eps(eps_data), color='black')
        ax1.set_xlabel(r'$\varepsilon\;\;\mathrm{[-]}$')
        ax1.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')
        ax1.set_title('Concrete law')
        ax2.plot(eps_data, self.get_d_sig_eps(eps_data), color='black')
        ax2.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax2.set_ylabel(r'$\mathrm{d}\sigma/\mathrm{d}w\;\;\mathrm{[MPa/mm]}$')
        ax2.set_title('tangential stiffness')

    def plot_sig_w(self, ax1, ax2):
        w_min_expr = (f_c / E_c * L * 2).subs(self.co_law_data)

        w_max_expr = (sp.solve(f_w + f_w.diff(w) * w, w)
                      [0]).subs(self.co_law_data)
        w_max = np.float_(w_max_expr) * 10
        w_min = np.float_(w_min_expr) * 10
        w_data = np.linspace(w_min, w_max, 100)
        ax2.plot(w_data, self.get_d_sig_w(w_data), color='orange')
        ax2.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax2.set_ylabel(r'$\mathrm{d}\sigma/\mathrm{d}w\;\;\mathrm{[MPa/mm]}$')
        ax1.plot(w_data, self.get_sig_w(w_data), lw=2, color='black')
        ax1.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax1.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')
        ax1.set_title('crack opening law')

    def plot_tau_s(self, ax1, ax2):
        s_max = float(s_3.subs(self.bond_law_data))
        s_data = np.linspace(-s_max, s_max, 100)
        ax2.plot(s_data, self.get_d_tau_s(s_data), color='orange')
#         ax2.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
#         ax2.set_ylabel(r'$\mathrm{d}\tau/\mathrm{d}s\;\;\mathrm{[MPa/mm]}$')
        ax1.plot(s_data, self.get_tau_s(s_data), lw=2, color='green')
#         ax1.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
#         ax1.set_ylabel(r'$\tau\;\;\mathrm{[MPa]}$')
#         ax1.set_title('Bond-slip law')


class ShearZone(tr.HasStrictTraits):

    B = tr.Float(20, auto_set=False, enter_set=True)
    H = tr.Float(60, auto_set=False, enter_set=True)
    L = tr.Float(100, auto_set=False, enter_set=True)

    x_Ca = tr.Property(depends_on='H,L')
    '''Corner coordinates'''
    @tr.cached_property
    def _get_x_Ca(self):
        L = self.L
        H = self.H
        return np.array([[0, L, L, 0],
                         [0, 0, H, H]], dtype=np.float_).T

    C_Li = tr.Property(depends_on='H,L')
    '''Lines'''
    @tr.cached_property
    def _get_C_Li(self):
        return np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)

    initial_crack_position = tr.Float(400)

    x_t_Ia = tr.Array(dtype=np.float_, value=[])

    def _x_t_Ia_default(self):
        return np.array([[self.initial_crack_position, 0]],
                        dtype=np.float_)

    x_Ia = tr.Property  # (depends_on='state_changed')

    def _get_x_Ia(self):
        return np.vstack([self.x_t_Ia, self.x_tip_a[np.newaxis, :]])

    I_Li = tr.Property  # (depends_on='state_changed')
    '''Crack segments'''
#    @tr.cached_property

    def _get_I_Li(self):
        N_I = np.arange(len(self.x_Ia))
        I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
        return I_Li

    n_I0 = tr.Int(3)
    '''Initial number of nodes along the path
    '''

    n_J = tr.Int(10)
    '''Number of nodes along the uncracked zone
    '''

    x_Ja = tr.Property  # (depends_on='state_changed, x_Ca, n_J')
    '''Uncracked vertical section'''

    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.x_Ca[-1, 1], self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    xx_Ka = tr.Property  # (depends_on='state_changed, x_Ca, n_J')
    '''Integrated section'''

    def _get_xx_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    n_m = tr.Int(5)
    x_Ka = tr.Property  # (depends_on='state_changed, x_Ca, n_J')

    def _get_x_Ka(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.xx_Ka[1:] - self.xx_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.xx_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.xx_Ka[[-1], :]])

    K_Li = tr.Property  # (depends_on='state_changed')
    '''Crack segments'''

    def _get_K_Li(self):
        N_K = np.arange(len(self.x_Ka))
        K_Li = np.array([N_K[:-1], N_K[1:]], dtype=np.int_).T
        return K_Li

    x_Lb = tr.Property

    def _get_x_Lb(self):
        return np.sum(self.x_Ka[self.K_Li], axis=1) / 2

    #=========================================================================
    # Transformation relative to the crack path
    #=========================================================================

    norm_n_vec_L = tr.Property  # (depends_on='state_changed')
    '''Unit line vector
    '''
#    @tr.cached_property

    def _get_norm_n_vec_L(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

    T_Lab = tr.Property  # (depends_on='state_changed')
    '''Unit line vector
    '''
#    @tr.cached_property

    def _get_T_Lab(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        line_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        norm_line_vec_L = np.sqrt(np.einsum('...a,...a->...',
                                            line_vec_La, line_vec_La))
        normed_line_vec_La = np.einsum('...a,...->...a',
                                       line_vec_La, 1. / norm_line_vec_L)
        t_vec_La = np.einsum('ijk,...j,k->...i',
                             EPS[:-1, :-1, :], normed_line_vec_La, Z)
        T_bLa = np.array([t_vec_La, normed_line_vec_La])
        T_Lab = np.einsum('bLa->Lab', T_bLa)
        return T_Lab

    plot_scale = tr.Float(1.0, auto_set=False, enter_set=True)

    x_n_ejaM = tr.Property  # (depends_on='state_changed')
    '''Unit line vector
    '''
#    @tr.cached_property

    def _get_x_n_ejaM(self):
        K_Li = self.K_Li
        T_Lab = self.T_Lab
        x_Lia = self.x_Ka[K_Li]
        x_n0_Mea = np.einsum(
            'e,Ma->Mae', np.ones((2,)), np.sum(x_Lia, axis=1) / 2)
        x_n_jMea = np.array(
            [x_n0_Mea, x_n0_Mea + self.plot_scale * T_Lab]
        )
        return np.einsum('jMae->ejaM', x_n_jMea)

    def get_rot_mtx(self, phi):
        return np.array(
            [[np.cos(phi), -np.sin(phi)],
             [np.sin(phi), np.cos(phi)]], dtype=np.float_)

    x_rot_a = tr.Property  # (depends_on='state_changed')

#    @tr.cached_property
    def _get_x_rot_a(self):
        x_rot_0 = self.x_tip_a[0]
        x_rot_1 = self.x_rot_1
        return np.array([x_rot_0, x_rot_1], dtype=np.float_)

    def rotate(self, x_Ka, phi):
        rot_mtx = self.get_rot_mtx(phi)
        u_Ib = np.einsum(
            'ab,...b->...a',
            rot_mtx, x_Ka - self.x_rot_a
        )
        return u_Ib + self.x_rot_a

    #=========================================================================
    # Unknown variables
    #=========================================================================
    phi = tr.Property(tr.Float)
    '''Rotation of the right-hand part of the shear zone.
    '''

    def _get_phi(self):
        x_fps_a = self.x_fps_a
        w_f_t = self.w_f_t
        x_rot_a = self.x_rot_a
        n_tip = len(self.x_t_Ia) - 1
        n_m_tip = n_tip * self.n_m
        T_ab = self.T_Lab[n_m_tip, ...]
        phi = get_phi(T_ab, x_fps_a, x_rot_a, w_f_t)
        return phi

    x_rot_1 = tr.Property
    '''Vertical coordinate of the center of rotation.
    '''

    def _get_x_rot_1(self):
        return self.X[0]

    theta = tr.Property(tr.Float)
    '''Direction of the current crack propagation segment.
    '''

    def _get_theta(self):
        return self.X[1]

    def get_moved_Ka(self, x_Ka):
        return self.rotate(x_Ka=x_Ka, phi=self.phi)

    x1_Ia = tr.Property  # (depends_on='v, state_changed')

#    @tr.cached_property
    def _get_x1_Ia(self):
        return self.get_moved_Ka(self.x_Ia)

    x1_Ka = tr.Property  # (depends_on='v, state_changed')

#    @tr.cached_property
    def _get_x1_Ka(self):
        return self.get_moved_Ka(self.x_Ka)

    x1_Ca = tr.Property(depends_on='v')

    @tr.cached_property
    def _get_x1_Ca(self):
        return self.get_moved_Ka(self.x_Ca)

    #=========================================================================
    # Control and state variables
    #=========================================================================
    v = tr.Property
    '''Vertical displacement on the right hand side
    '''

    def _get_v(self):
        x_rot_0 = self.x_rot_a[0]
        phi = self.phi
        L_rot_distance = self.L - x_rot_0
        v = L_rot_distance * np.sin(phi)
        return v

    I = tr.Property
    '''Sectional moment of inertia
    '''

    def _get_I(self):
        H_uncracked = self.H  # - self.x_tip_a[1]
        return self.B * np.power(H_uncracked, 3) / 12.0

    Q = tr.Property
    '''Shear force'''

    def _get_Q(self):
        return self.M / (self.L - self.x_rot_a[0])

    def _get_xQ(self):
        return 12 * self.mm.E_c * self.I / np.power(self.L, 3) * self.v

    M_x_rot = tr.Property
    '''Bending moment at the center of rotation. 
    '''

    def _get_M_x_rot(self):
        return self.Q * (self.L - self.x_rot_a[0])

    tau_fps = tr.Property
    '''Shear stress in global xz coordinates in the fracture 
    process segment. Quadratic profile of shear stress assumed.
    '''

    def _get_tau_fps(self):
        z_fps = self.x_fps_a[1]
        Q = self.Q
        H = self.H
        B = self.B
        tau_fps = (6 * Q * z_fps / (H**2 + H * z_fps - 2 * z_fps**2))
        return tau_fps / B

    sig_fps_0 = tr.Property
    '''Normal stress component in global $x$ direction in the fracture .
    process segment.
    '''

    def _get_sig_fps_0(self):
        n_tip = len(self.x_t_Ia) - 1
        n_m_tip = n_tip * self.n_m
        S_a = self.S_La[n_m_tip, ...]
        return S_a[0] / self.B

    input_changed = tr.Event
    '''Register the input change event to trigger recalculation.
    '''

    state_changed = tr.Event
    '''Register the change of state to update derived interim variables.
    '''

    #=========================================================================
    # Kinematics
    #=========================================================================

    u_Lib = tr.Property  # (depends_on='v, state_changed')

#    @tr.cached_property
    def _get_u_Lib(self):
        K_Li = self.K_Li
        u_Ka = self.x1_Ka - self.x_Ka
        u_Lia = u_Ka[K_Li]
        T_Lab = self.T_Lab
        u_Lib = np.einsum(
            'Lia,Lab->Lib', u_Lia, T_Lab
        )
        return u_Lib

    u_Lb = tr.Property  # (depends_on='v, state_changed')

#    @tr.cached_property
    def _get_u_Lb(self):
        K_Li = self.K_Li
        u_Ka = self.x1_Ka - self.x_Ka
        u_Lia = u_Ka[K_Li]
        u_La = np.sum(u_Lia, axis=1) / 2
        T_Lab = self.T_Lab
        u_Lb = np.einsum(
            'La,Lab->Lb', u_La, T_Lab
        )
        return u_Lb

    get_u0 = tr.Property

    def _get_get_u0(self):
        return interp1d(self.x_Lb[:, 1], self.u_Lb[:, 0],
                        fill_value='extrapolate')

    x_u0_a = tr.Property  # (depends_on='state_changed')

#    @tr.cached_property
    def _get_x_u0_a(self):
        x0 = self.X[0]
        res = root(self.get_u0, x0, tol=1e-8)
        if res.success:
            return res.x

    u_ejaM = tr.Property  # (depends_on='v, state_changed')

#    @tr.cached_property
    def _get_u_ejaM(self):
        return self.get_f_ejaM(self.u_Lb)

    def get_f_ejaM(self, f_Lb, scale=1.0):
        K_Li = self.K_Li
        T_Lab = self.T_Lab
        x_Lia = self.x_Ka[K_Li]
        x_n0_Mea = np.einsum(
            'e,Ma->Mae', np.ones((2,)), np.sum(x_Lia, axis=1) / 2)
        f_base_Lab = np.einsum(
            'Lb, Lab->Lab', f_Lb, scale * T_Lab
        )
        x_n_jMea = np.array(
            [x_n0_Mea, x_n0_Mea + f_base_Lab]
        )
        return np.einsum('jMae->ejaM', x_n_jMea)

    #=========================================================================
    # Material characteristics
    #=========================================================================
    mm = tr.Instance(IMaterialModel)
    '''Material model containing the bond-slip law 
    concrete compression and crack softening law 
    '''

    eta = tr.Float(0.2, PARAM=True)
    '''Fraction of characteristic length to take as a propagation segment
    '''

    L_fps = tr.Property(tr.Float)
    '''Length of the fracture process segment.
    '''

    def _get_L_fps(self):
        return self.eta * self.mm.L_c

    #=========================================================================
    # Fracture process segment
    #=========================================================================
    T_fps_a = tr.Property(tr.Array)
    '''Orientation matrix of the crack propagation segment 
    '''

    def _get_T_fps_a(self):
        return np.array([-np.sin(self.theta), np.cos(self.theta)],
                        dtype=np.float_)

    x_tip_a = tr.Property(tr.Array)
    '''Unknown position of the crack tip. Depends on the sought 
    fracture process segment orientation $\theta$ 
    '''

    def _get_x_tip_a(self):
        return self.x_fps_a + self.L_fps * self.T_fps_a

    x_fps_a = tr.Property(tr.Array)
    '''Position of the starting point of the fracture process segment.
    '''

    def _get_x_fps_a(self):
        return self.x_t_Ia[-1, :]

    w_f_t = tr.Property(tr.Float)
    '''Critical crack opening at the level of tensile concrete strength.
    
    @todo: Check consistency, the crack opening is obtained as a 
    product of the critical strain at strength and the characteristic
    size of the localization zone.
    '''

    def _get_w_f_t(self):
        return self.mm.f_t / self.mm.E_c * self.mm.L_c

    #=========================================================================
    # Stress transformation and integration
    #=========================================================================

    S_Lb = tr.Property
    '''Stress returned by the material model
    '''

    def _get_S_Lb(self):
        u_a = self.u_Lb
        Sig_w = mm.get_sig_w(u_a[..., 0]) * self.B
        Tau_w = mm.get_tau_s(u_a[..., 1]) * self.B
#        Tau_w = mm.get_tau_s(np.fabs(u_a[..., 1])) * self.B
        return np.einsum('b...->...b', np.array([Sig_w, -Tau_w], dtype=np.float_))

    S_La = tr.Property
    '''Transposed stresses'''

    def _get_S_La(self):
        S_Lb = self.S_Lb
        S_La = np.einsum('Lb,Lab->La', S_Lb, self.T_Lab)
        return S_La

    F_La = tr.Property
    '''Integrated segment forces'''

    def _get_F_La(self):
        S_La = self.S_La
        F_La = np.einsum('La,L->La', S_La, self.norm_n_vec_L)
        return F_La

    z_f = tr.Array(np.float_, value=[1])
    E_f = tr.Array(np.float_, value=[210000])
    A_f = tr.Array(np.float_, value=[2 * np.pi * 8**2])

    F_f = tr.Property
    '''Get the discrete force in the reinforcement f
    @todo: Currently, uniform strain is assumed along the bar over the
    total length of the shear zone. Include a pullout
    '''

    def _get_F_f(self):
        u_f = self.get_u0(self.z_f)
        return self.E_f * self.A_f * u_f / self.L

    M = tr.Property
    '''Internal bending moment obtained by integrating the  
    normal stresses with the lever arm rooted at the height of the neutral 
    axis.
    '''

    def _get_M(self):
        x_Lia = self.x_Ka[self.K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = self.F_La
        M_L = (x_La[:, 1] - self.x_rot_a[1]) * F_La[:, 0]
        M = np.sum(M_L, axis=0)
        for y in self.z_f:
            M += (y - self.x_rot_a[1]) * self.F_f
        return -M

    F_a = tr.Property(depends_on='v, state_changed')
    '''Integrated normal and shear force
    '''

    def _get_F_a(self):
        F_La = self.F_La
        F_a = np.sum(F_La, axis=0)
        F_a[0] += np.sum(self.F_f)
        return F_a

    theta_bar = tr.Property
    '''Angle between the vertical direction and the orientation of 
    the principal stresses setting the orientation of the fracture
    process segment in the next iteration step.
    '''

    def _get_theta_bar(self):
        tau_fps = self.tau_fps
        sig_x = self.sig_fps_0
        tan_theta = 2 * tau_fps / (
            sig_x + np.sqrt(4 * tau_fps**2 + sig_x**2))
        return np.arctan(tan_theta)

    #=========================================================================
    # Nonlinear solver
    #=========================================================================

    _X = tr.Array(np.float_)
    '''Primary unknown variables subject to the iteration process.
    - center of rotation
    - inclination angle of a new crack segment
    '''

    def __X_default(self):
        return [self.H / 2.0,
                0.0, ]

    X = tr.Property()

    def _get_X(self):
        return self._X

    def _set_X(self, value):
        self.state_changed = True
        self._X[:] = value

    def get_R(self):
        '''Residuum checking the lack-of-fit 
        - of the normal force equilibrium in the cross section
        - of the orientation of the principal stress and of the fracture
          process segment (FPS) 
        '''
        N, _ = self.F_a
        R_M = (self.theta - self.theta_bar) * 100
        R = np.array([N,  R_M], dtype=np.float_)
        return R

    X_sol = tr.Property
    '''Get X vector satisfying the conditions given in the residuum.
    '''

    def _get_X_sol(self):
        X0 = np.copy(self.X[:])

        def get_R_X(X):
            self.X[:] = X[:]
            return self.get_R()
        res = root(get_R_X, X0, tol=1e-5)
        if res.success == False:
            raise ValueError('no solution found')
        return res.x

    def solve(self):
        self.X[:] = self.X_sol
        print(self.get_R())

    def next_crack_segment(self):
        self.x_t_Ia = np.vstack([self.x_t_Ia, self.x_tip_a[np.newaxis, :]])
        self.X[1] = 0

    record_traits = tr.List(
        ['M', 'Q', 'M_x_rot', 'v', 'phi',
         'tau_fps', 'theta', 'theta_bar',
         'sig_fps_0', 'v', 'phi']
    )

    response = tr.Property(depends_on='input_changed')

    n_seg = tr.Int(5, auto_set=False, enter_set=True)
    '''Number of crack propagation steps.
    '''

    @tr.cached_property
    def _get_response(self):
        n_seg = self.n_seg
        record = {key: [0] for key in self.record_traits}
        for seg in range(n_seg):
            print('seg', seg)
            try:
                self.solve()
            except ValueError:
                print('No convergence')
                self.X[1] = 0
                break
            record_vals = self.trait_get(*self.record_traits)
            for key, val in record.items():
                val.append(record_vals[key])
            self.next_crack_segment()
        rarr = {key: np.array(vals, dtype=np.float_)
                for key, vals in record.items()}
        return rarr

    #=========================================================================
    # Plotting methods
    #=========================================================================

    def plot_x_Ka(self, ax):
        x_Ka = self.x_Ka
        x_La = np.sum(x_Ka[self.K_Li], axis=1) / 2
        x_aL = x_La.T
        ax.plot(*x_aL, color='brown', markersize=2)

    def plot_x_tip_a(self, ax):
        '''Show the current crack tip.
        '''
        x, y = self.x_tip_a
        ax.plot(x, y, 'bo', color='red', markersize=10)

    def plot_x_rot_a(self, ax):
        '''Show the current center of rotation.
        '''
        x, y = self.x_rot_a

        ax.annotate('center of rotation', xy=(x, y), xytext=(x + 50, y + 20),
                    arrowprops=dict(facecolor='black', width=1, shrink=0.1),
                    )

        ax.plot([0, self.L], [y, y],
                color='black', linestyle='--')
        ax.plot(x, y, 'bo', color='blue', markersize=10)

    def plot_x_fps_a(self, ax):
        x, y = self.x_fps_a
        ax.plot(x, y, 'bo', color='green', markersize=10)

    def plot_crack_orientation(self, ax1, ax2):
        '''Visualize the stress components affecting the 
        crack orientation in the next step.
        '''
        z = self.x_Ia[:-1, 1]
        R = self.response
        ax1.set_title(r'Fracture propagation segment')
        ax1.plot(R['theta'], z, color='orange',
                 label='theta')
        ax1.plot(R['theta_bar'], z, color='black',
                 linestyle='dashed', label='theta_bar')
        ax1.set_xlabel(r'Orientation $\theta$ [$\pi^{-1}$]')
        ax1.set_ylabel(r'Vertical position $z$ [mm]')
        ax1.legend(loc='lower left')
        ax2.plot(R['tau_fps'], z, color='red', label='tau_fps')
        ax2.fill_betweenx(z, R['tau_fps'], 0, color='red', alpha=0.1)
        ax2.plot(R['sig_fps_0'], z, color='black', label='sig_x')
        ax2.fill_betweenx(z, R['sig_fps_0'], 0, color='black', alpha=0.1)
        ax2.set_xlabel(r'$\sigma_x$ and $\tau_{xz}$ [MPa]')
        ax2.legend(loc='upper left')

    def plot_sz0(self, ax):
        x_Ia = self.x_Ia
        x_Ca = self.x_Ca
        x_aI = x_Ia.T
        x_LL = x_Ca[0]
        x_LU = x_Ca[3]
        x_RL = self.x_Ka[0]
        x_RU = self.x_Ka[-1]
        x_Da = np.array([x_LL, x_RL, x_RU, x_LU])
        D_Li = np.array([[0, 1], [2, 3], [3, 0]], dtype=np.int_)
        x_aiD = np.einsum('Dia->aiD', x_Da[D_Li])
        ax.plot(*x_aiD, color='black')
        ax.plot(*x_aI, lw=2, color='black')

    def plot_sz1(self, ax):
        x_Ia = self.x1_Ia
        x_Ca = self.x1_Ca
        x_aI = x_Ia.T
        x_LL = self.x1_Ka[0]
        x_LU = self.x1_Ka[-1]
        x_RL = x_Ca[1]
        x_RU = x_Ca[2]
        x_Da = np.array([x_LL, x_RL, x_RU, x_LU])
        D_Li = np.array([[0, 1], [1, 2], [2, 3], ], dtype=np.int_)
        x_aiD = np.einsum('Dia->aiD', x_Da[D_Li])
        ax.plot(*x_aiD, color='black')
        ax.set_title(r'Simulated crack path')
        ax.set_xlabel(r'Horizontal position $x$ [mm]')
        ax.set_ylabel(r'Vertical position $z$ [mm]')
        ax.plot(*x_aI, lw=2, color='black')

    def plot_sz_fill(self, ax):
        x_Ca = self.x1_Ca
        x_Da = np.vstack([
            x_Ca[:1],
            self.x_Ia,
            self.x1_Ia[::-1],
            x_Ca[1:, :],
        ])
        ax.fill(*x_Da.T, color='gray', alpha=0.2)

    def plot_reinf(self, ax):
        for z in self.z_f:
            # left part
            ax.plot([0, self.L], [z, z], color='maroon', lw=5)

    def plot_T_Lab(self, ax):
        K_Li = self.K_Li
        x_n_ejaM = self.x_n_ejaM
        x_n_eajM = np.einsum('ejaM->eajM', x_n_ejaM)
        ax.plot(*x_n_eajM.reshape(-1, 2, len(K_Li)), color='orange', lw=3)

    def plot_u_T_Lab(self, ax):
        u_ejaM = self.u_ejaM
        u_eajM = np.einsum('ejaM->eajM', u_ejaM)
        ax.plot(*u_eajM.reshape(-1, 2, len(self.K_Li)), color='orange', lw=3)

    def x_plot_S_Lb(self, ax):
        S_max = np.max(self.S_Lb)
        G_min = np.min([self.L, self.H]) * 0.2
        S_ejaM = self.get_f_ejaM(self.S_Lb, scale=G_min / S_max)
        S_eajM = np.einsum('ejaM->eajM', S_ejaM)
        ax.plot(*S_eajM.reshape(-1, 2, len(self.K_Li)), color='orange', lw=3)
        return

    def plot_hlines(self, ax, h_min, h_max):
        _, y_tip = self.x_tip_a
        _, y_rot = self.x_rot_a
        _, z_fps = self.x_fps_a
        ax.plot([h_min, h_max], [y_tip, y_tip],
                color='black', linestyle=':')
        ax.plot([h_min, h_max], [y_rot, y_rot],
                color='black', linestyle='--')
        ax.plot([h_min, h_max], [z_fps, z_fps],
                color='black', linestyle='-.')

    def plot_u_Lc(self, ax, u_Lc, idx=0, color='red', label='w'):
        x_La = self.x_Lb
        u_Lb_min = np.min(u_Lc[:, idx])
        u_Lb_max = np.max(u_Lc[:, idx])
        self.plot_hlines(ax, u_Lb_min, u_Lb_max)
        ax.plot(u_Lc[:, idx], x_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_La[:, 1], u_Lc[:, idx], 0, color=color, alpha=0.1)
        ax.legend()

    def plot_S_Lc(self, ax, S_Lc, idx=0,
                  title='Normal stress profile',
                  label=r'$\sigma_{xx}$',
                  color='green',
                  ylabel=r'Vertical position [mm]',
                  xlabel=r'Stress [MPa]'):
        x_La = self.x_Lb
        S_La_min = np.min(S_Lc[:, idx])
        S_La_max = np.max(S_Lc[:, idx])
        self.plot_hlines(ax, S_La_min, S_La_max)
        ax.set_title(title)
        ax.plot(S_Lc[:, idx], x_La[:, 1],
                color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.fill_betweenx(x_La[:, 1], S_Lc[:, idx], 0,
                         color=color, alpha=0.2)
        ax.legend()

    def _align_xaxis(self, ax1, ax2):
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = (ax1, ax2)
        extrema = [ax.get_xlim() for ax in axes]
        tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [list(reversed(l))
                                   for l in (axes, extrema, tops)]

        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]

        b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
        t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
        axes[0].set_xlim(extrema[0][0], b_new_t)
        axes[1].set_xlim(t_new_b, extrema[1][1])

    traits_view = ui.View(
        ui.Item('L'),
        ui.Item('H')
    )


if __name__ == '__main__':
    mm = MaterialModel(G_f=0.09, f_c=-30)

    '''Development history:
    
    1) Test the correctness of the calculation for 
    given values in 
    2) Integration over the height
    '''
    sz = ShearZone(
        B=100, H=200, L=800,
        n_J=10,
        n_m=8,
        n_seg=31,
        eta=0.02,
        initial_crack_position=250,
        z_f=[20],  # 10],
        A_f=[2 * np.pi * 8**2],
        plot_scale=1,
        mm=mm
    )
    if True:
        print('x_fps')
        print(sz.x_fps_a)
        print('x_tip')
        print(sz.x_tip_a)
        print('x_Ia')
        print(sz.x_Ia)
        print('L_c')
        print(sz.mm.L_c)
        print('L_fps')
        print(sz.L_fps)
        print('X')
        print(sz.X)
        print('phi')
        print(sz.phi)
        print('F_a')
        print(sz.F_a)
        print('x_rot_a')
        print(sz.x_rot_a)
        print('w_tip_a')
        print(sz.w_tip)
        print('w_f_t')
        print(sz.w_f_t)
        print('X')
        print(sz.X)
        print('Q')
        print(sz.Q)
        R = sz.response
    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sz.plot_sz0(ax)
        sz.plot_sz1(ax)
        plt.show()
    if False:
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(
            2, 2, figsize=(12, 8), tight_layout=True)
        mm.plot_sig_w(ax11, ax21)
        mm.plot_tau_s(ax12, ax22)
        plt.show()
    if False:
        fig, ax = plt.subplots(
            1, 1, figsize=(7, 2), tight_layout=True
        )
        ax.axis('equal')
        sz.plot_sz1(ax)
        sz.plot_sz0(ax)

        sz.plot_x_tip_a(ax)
        sz.plot_x_rot_a(ax)
        sz.plot_x_fps_a(ax)
        sz.plot_reinf(ax)
        plt.show()
    if True:
        fig, ((ax11, ax12, ax13, ax14),
              (ax21, ax22, ax23, ax24)) = plt.subplots(
            2, 4, figsize=(20, 18), tight_layout=True
        )
        ax11.axis('equal')
        x_La = np.sum(sz.x_Ka[sz.K_Li], axis=1) / 2

        sz.plot_sz1(ax11)
        sz.plot_sz0(ax11)
        sz.plot_sz_fill(ax11)
        sz.plot_x_tip_a(ax11)
        sz.plot_x_rot_a(ax11)
        sz.plot_x_fps_a(ax11)
        sz.plot_reinf(ax11)
        ax11.set_xlim(sz.x_rot_a[0] - 50,
                      sz.initial_crack_position + 40)

        ax21b = ax21.twiny()
        sz.plot_crack_orientation(ax21, ax21b)

        ax12.plot(R['v'], R['Q'], color='black')
        ax12.set_ylabel(r'Shear force $Q$ [kN]')
        ax12.set_xlabel(r'Displacement $v$ [mm]')

        mm.plot_sig_w(ax22, ax22.twinx())
        mm.plot_tau_s(ax22, ax22.twinx())

        ax13a = ax13.twiny()
        sz.plot_u_Lc(ax13a, sz.u_Lb, 1)
        sz.plot_S_Lc(ax13, sz.S_Lb, 1)
        sz._align_xaxis(ax13, ax13a)

        ax23a = ax23.twiny()
        sz.plot_u_Lc(ax23a, sz.u_Lb, 0)
        sz.plot_S_Lc(ax23, sz.S_Lb, 0)
        sz._align_xaxis(ax23, ax23a)

        ax14a = ax14.twiny()
        sz.plot_u_Lc(ax14a, sz.u_Lb, 1)
        sz.plot_S_Lc(ax14, sz.S_La, 1)
        sz._align_xaxis(ax14, ax14a)

        ax24a = ax24.twiny()
        sz.plot_u_Lc(ax24a, sz.u_Lb, 0)
        sz.plot_S_Lc(ax24, sz.S_La, 0)
        sz._align_xaxis(ax24, ax24a)

        plt.show()

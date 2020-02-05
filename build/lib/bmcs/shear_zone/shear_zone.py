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

tau_s = 0 * sp.Piecewise(
    (tau_1 / s_1 * s, s < s_1),
    (tau_1 + (tau_2 - tau_1) / (s_2 - s_1) * (s - s_1), s <= s_2),
    (tau_2 + (tau_3 - tau_2) / (s_3 - s_2) * (s - s_2), s > s_2)
)
d_tau_s = tau_s.diff(s)


@tr.provides(IMaterialModel)
class MaterialModel(tr.HasStrictTraits):

    f_c = tr.Float(-30.0, PARAM=True)
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
    tau_1 = tr.Float(2.0, PARAM=True)
    s_1 = tr.Float(0.000001, PARAM=True)
    tau_2 = tr.Float(1.0, PARAM=True)
    s_2 = tr.Float(0.4, PARAM=True)
    tau_3 = tr.Float(0.8, PARAM=True)
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
        ax1.set_title('concrete law')
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
        ax2.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax2.set_ylabel(r'$\mathrm{d}\tau/\mathrm{d}s\;\;\mathrm{[MPa/mm]}$')
        ax1.plot(s_data, self.get_tau_s(s_data), lw=2, color='black')
        ax1.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax1.set_ylabel(r'$\tau\;\;\mathrm{[MPa]}$')
        ax1.set_title('bond-slip law')


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

    x_t_Ia = tr.Array(dtype=np.float_, value=[])

    def _x_t_Ia_default(self):
        x1_Ia = np.linspace(0, self.initial_crack_length, self.n_I0)
        x0_Ia = np.ones_like(x1_Ia) * self.initial_crack_position
        return np.c_[x0_Ia, x1_Ia]

    x_Ia = tr.Property(depends_on='state_changed')

    def _get_x_Ia(self):
        return np.vstack([self.x_t_Ia, self.X_tip_a[np.newaxis, :]])

    I_Li = tr.Property(depends_on='state_changed')
    '''Crack segments'''
    @tr.cached_property
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

    x_Ja = tr.Property(depends_on='state_changed, x_Ca, n_J')
    '''Uncracked vertical section'''
    @tr.cached_property
    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.x_Ca[-1, 1], self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    x_Ka = tr.Property(depends_on='state_changed, x_Ca, n_J')
    '''Integrated section'''
    #@tr.cached_property

    def _get_x_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    n_m = tr.Int(5)
    x_Kma = tr.Property(depends_on='state_changed, x_Ca, n_J')

    def _get_x_Kma(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.x_Ka[1:] - self.x_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.x_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.x_Ka[[-1], :]])

    K_Li = tr.Property(depends_on='state_changed')
    '''Crack segments'''
    @tr.cached_property
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

    norm_n_vec_L = tr.Property(depends_on='state_changed')
    '''Unit line vector
    '''
    @tr.cached_property
    def _get_norm_n_vec_L(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

    o_base_Lab = tr.Property(depends_on='state_changed')
    '''Unit line vector
    '''
    @tr.cached_property
    def _get_o_base_Lab(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        norm_n_vec_L = np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))
        normed_n_vec_La = np.einsum('...a,...->...a',
                                    n_vec_La, 1. / norm_n_vec_L)
        t_vec_La = np.einsum('ijk,...j,k->...i',
                             EPS[:-1, :-1, :], normed_n_vec_La, Z)
        o_base_bLa = np.array([t_vec_La, normed_n_vec_La])
        o_base_Lab = np.einsum('bLa->Lab', o_base_bLa)
        return o_base_Lab

    plot_scale = tr.Float(1.0, auto_set=False, enter_set=True)

    x_n_ejaM = tr.Property(depends_on='state_changed')
    '''Unit line vector
    '''
    @tr.cached_property
    def _get_x_n_ejaM(self):
        K_Li = self.K_Li
        o_base_Lab = self.o_base_Lab
        x_Lia = self.x_Ka[K_Li]
        x_n0_Mea = np.einsum(
            'e,Ma->Mae', np.ones((2,)), np.sum(x_Lia, axis=1) / 2)
        x_n_jMea = np.array(
            [x_n0_Mea, x_n0_Mea + self.plot_scale * o_base_Lab]
        )
        return np.einsum('jMae->ejaM', x_n_jMea)

    #=========================================================================
    # Field manipulation functions - translate and rotate
    #=========================================================================
    def translate(self, x_Ka, u_a):
        return x_Ka[:, :] + u_a[np.newaxis, :]

    def get_rot_mtx(self, phi):
        return np.array(
            [[np.cos(phi), -np.sin(phi)],
             [np.sin(phi), np.cos(phi)]], dtype=np.float_)

    ROT = tr.Property(depends_on='state_changed')

    @tr.cached_property
    def _get_ROT(self):
        x_ROT = self.X_tip_a[0]
        y_ROT = self.X[0]
        return np.array([x_ROT, y_ROT], dtype=np.float_)

    def rotate(self, x_Ka, phi):
        rot_mtx = self.get_rot_mtx(phi)
        u_Ib = np.einsum(
            'ab,...b->...a',
            rot_mtx, x_Ka - self.ROT
        )
        return u_Ib + self.ROT

    phi = tr.Property()

    def _get_phi(self):
        return np.arctan(self.v / (self.L - self.X_tip_a[0]))

    def get_moved_Ka(self, x_Ka):
        return self.rotate(x_Ka=x_Ka, phi=self.phi)

    x1_Ka = tr.Property(depends_on='v, state_changed')

    @tr.cached_property
    def _get_x1_Ka(self):
        return self.get_moved_Ka(self.x_Ka)

    x1_Ca = tr.Property(depends_on='v')

    @tr.cached_property
    def _get_x1_Ca(self):
        return self.get_moved_Ka(self.x_Ca)

    #=========================================================================
    # Initial state
    #=========================================================================
    initial_crack_position = tr.Float()

    def _initial_crack_position_default(self):
        return self.L * 0.5

    initial_crack_length = tr.Float()

    def _initial_crack_length_default(self):
        return self.H * 0.05

    #=========================================================================
    # Control and state variables
    #=========================================================================
    v = tr.Float(0)
    '''Control vertical displacement
    '''
    state_changed = tr.Event

    _X = tr.Array(np.float_)

    def __X_default(self):
        return [self.H / 2,
                self.initial_crack_position,
                self.initial_crack_length]

    X = tr.Property()

    def _get_X(self):
        return self._X

    def _set_X(self, value):
        self.state_changed = True
        self._X[:] = value

    #=========================================================================
    # Kinematics
    #=========================================================================

    u_Lib = tr.Property(depends_on='v, state_changed')

    @tr.cached_property
    def _get_u_Lib(self):
        K_Li = self.K_Li
        u_Ka = self.x1_Ka - self.x_Ka
        u_Lia = u_Ka[K_Li]
        o_base_Lab = self.o_base_Lab
        u_Lib = np.einsum(
            'Lia,Lab->Lib', u_Lia, o_base_Lab
        )
        return u_Lib

    u_Lb = tr.Property(depends_on='v, state_changed')

    @tr.cached_property
    def _get_u_Lb(self):
        K_Li = self.K_Li
        u_Ka = self.x1_Ka - self.x_Ka
        u_Lia = u_Ka[K_Li]
        u_La = np.sum(u_Lia, axis=1) / 2
        o_base_Lab = self.o_base_Lab
        u_Lb = np.einsum(
            'La,Lab->Lb', u_La, o_base_Lab
        )
        return u_Lb

    get_u0 = tr.Property

    def _get_get_u0(self):
        return interp1d(self.x_Lb[:, 1], self.u_Lb[:, 0],
                        fill_value='extrapolate')

    x_u0_a = tr.Property(depends_on='state_changed')

    @tr.cached_property
    def _get_x_u0_a(self):
        x0 = self.X[0]
        res = root(self.get_u0, x0, tol=1e-8)
        if res.success:
            return res.x

    u_ejaM = tr.Property(depends_on='v, state_changed')

    @tr.cached_property
    def _get_u_ejaM(self):
        return self.get_f_ejaM(self.u_Lb)

    def get_f_ejaM(self, f_Lb, scale=1.0):
        K_Li = self.K_Li
        o_base_Lab = self.o_base_Lab
        x_Lia = self.x_Ka[K_Li]
        x_n0_Mea = np.einsum(
            'e,Ma->Mae', np.ones((2,)), np.sum(x_Lia, axis=1) / 2)
        f_base_Lab = np.einsum(
            'Lb, Lab->Lab', f_Lb, scale * o_base_Lab
        )
        x_n_jMea = np.array(
            [x_n0_Mea, x_n0_Mea + f_base_Lab]
        )
        return np.einsum('jMae->ejaM', x_n_jMea)

    mm = tr.Instance(IMaterialModel)

    #=========================================================================
    # Stress transformation and integration
    #=========================================================================

    S_Lb = tr.Property
    '''Stress returned by the material model
    '''

    def _get_S_Lb(self):
        u_a = self.u_Lb
        Sig_w = mm.get_sig_w(u_a[..., 0]) * self.B
        Tau_w = mm.get_tau_s(np.fabs(u_a[..., 1])) * self.B
        return np.einsum('b...->...b', np.array([Sig_w, Tau_w], dtype=np.float_))

    S_La = tr.Property
    '''Transposed stresses'''

    def _get_S_La(self):
        S_Lb = self.S_Lb
        S_La = np.einsum('Lb,Lab->La', S_Lb, self.o_base_Lab)
        return S_La

    F_La = tr.Property
    '''Integrated segment forces'''

    def _get_F_La(self):
        S_La = self.S_La
        F_La = np.einsum('La,L->La', S_La, self.norm_n_vec_L)
        return F_La

    y_f = tr.Array(np.float_, value=[1])
    E_f = tr.Array(np.float_, value=[210000])
    A_f = tr.Array(np.float_, value=[2 * np.pi * 8**2])

    F_f = tr.Property

    def _get_F_f(self):
        u_f = self.get_u0(self.y_f)
        return self.E_f * self.A_f * u_f / self.L

    FM_a = tr.Property(depends_on='v, state_changed')

    def _get_FM_a(self):
        x_Lia = self.x_Ka[self.K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = self.F_La
        M_L = x_La[:, 0] * F_La[:, 1]
        F_a = np.sum(F_La, axis=0)
        F_a[0] += np.sum(self.F_f)
        M = np.sum(M_L, axis=0)
#        M -= np.sum(self.F_f * self.y_f)
        return np.hstack([F_a, M])

    #=========================================================================
    # Plotting methods
    #=========================================================================

    def plot_x_Ka(self, ax):
        x_Ka = self.x_Ka
        x_La = np.sum(x_Ka[self.K_Li], axis=1) / 2
        x_aL = x_La.T
        ax.plot(*x_aL, 'bo')

    def plot_X_tip_a(self, ax):
        x, y = self.X_tip_a
        ax.plot(x, y, 'bo', color='red')

    def plot_sz0(self, ax):
        x_Ka = self.x_Ka
        x_Ca = self.x_Ca
        x_aK = x_Ka.T
        x_aiC = np.einsum('Cia->aiC', x_Ca[self.C_Li])
        ax.plot(*x_aiC, color='black')
        ax.plot(*x_aK, lw=2, color='blue')

    def plot_sz1(self, ax):
        x_Ka = self.x1_Ka
        x_Ca = self.x1_Ca
        x_aK = x_Ka.T
        x_aiC = np.einsum('Cia->aiC', x_Ca[self.C_Li])
        ax.plot(*x_aiC, color='black')
        ax.plot(*x_aK, lw=2, color='blue')

    def plot_o_base_Lab(self, ax):
        K_Li = self.K_Li
        x_n_ejaM = self.x_n_ejaM
        x_n_eajM = np.einsum('ejaM->eajM', x_n_ejaM)
        ax.plot(*x_n_eajM.reshape(-1, 2, len(K_Li)), color='orange', lw=3)

    def plot_uo_base_Lab(self, ax):
        u_ejaM = self.u_ejaM
        u_eajM = np.einsum('ejaM->eajM', u_ejaM)
        ax.plot(*u_eajM.reshape(-1, 2, len(self.K_Li)), color='orange', lw=3)

    def plot_S_Lb(self, ax):
        S_max = np.max(self.S_Lb)
        G_min = np.min([self.L, self.H]) * 0.2
        S_ejaM = self.get_f_ejaM(self.S_Lb, scale=G_min / S_max)
        S_eajM = np.einsum('ejaM->eajM', S_ejaM)
        ax.plot(*S_eajM.reshape(-1, 2, len(self.K_Li)), color='orange', lw=3)
        return

    X_tip_a = tr.Property()

    def _get_X_tip_a(self):
        return self.X[1:]

    w_tip = tr.Property()

    def _get_w_tip(self):
        n_tip = len(self.x_Ia[-1])
        return self.u_Lib[n_tip, 1, 0]
#        print('N_TIP', n_tip)
#        n_X_I = len(self.x_Ia)
#        return self.u_Lb[n_X_I, 0]

    w_f_t = tr.Property()

    def _get_w_f_t(self):
        return self.mm.f_t / self.mm.E_c * self.mm.L_c

    traits_view = ui.View(
        ui.Item('L'),
        ui.Item('H')
    )


if __name__ == '__main__':
    mm = MaterialModel(G_f=0.08)

    crack_length = 4
    n_crack_segs = 6

    sz = ShearZone(
        B=100, H=200, L=500,
        n_J=10,
        initial_crack_length=20,
        initial_crack_position=400,
        y_f=[5],
        plot_scale=1,
        x_t_Ia=[[400, 0],
                #                [400, 5],
                [400, 10],
                ],
        v=0.2,
        mm=mm
    )

    print(sz.x_Kma)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXx')
    print(sz.x_Ia)
    print(sz.FM_a)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(sz.X)
    print(sz.phi)
    print(sz.ROT)
    print('sz.FM_a')
    print(sz.FM_a)

    def resid(X):
        sz.X = X
        N, Q, M = sz.FM_a
        W = (sz.w_tip - sz.w_f_t)
        return np.array([W, N, M - Q * sz.L], dtype=np.float_)

    if True:
        X0 = np.copy(sz.X)
        res = root(lambda X: resid(X),
                   X0, tol=1e-6)
        print('Success')
        print(res.success)
        sz.X = res.x
        print('X')
        print(sz.X)
        print('X_tip_a')
        print(sz.X_tip_a)
        print('FM_a')
        print(sz.FM_a)
        print('R')
        print(resid(res.x))
        print('W_f_t')
        print(sz.w_f_t)
        print('w_tip')
        print(sz.w_tip)
        print('sigma')
    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sz.plot_x_Ka(ax)
        sz.plot_sz0(ax)
        sz.plot_sz1(ax)
#        sz.plot_S_Lb(ax)
        plt.show()
    if False:
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(
            2, 2, figsize=(12, 8), tight_layout=True)
        # sz.plot_o_base_Lab(ax)
        mm.plot_sig_w(ax11, ax21)
        mm.plot_tau_s(ax12, ax22)
        plt.show()
    if True:
        fig, ((ax11, ax12, ax13),
              (ax21, ax22, ax23)) = plt.subplots(
            2, 3, figsize=(20, 18), tight_layout=True
        )
        x_La = np.sum(sz.x_Ka[sz.K_Li], axis=1) / 2

        sz.plot_x_Ka(ax11)
        sz.plot_sz0(ax11)
        sz.plot_sz1(ax11)
        sz.plot_X_tip_a(ax11)

        mm.plot_sig_w(ax12, ax12.twinx())
        mm.plot_tau_s(ax22, ax22.twinx())

        _, y_tip = sz.X_tip_a
        u_Lb_min = np.min(sz.u_Lb[:, 0])
        u_Lb_max = np.max(sz.u_Lb[:, 0])
        ax13.plot([u_Lb_min, u_Lb_max], [y_tip, y_tip],
                  color='black')
        ax13.plot(sz.u_Lb[:, 0], x_La[:, 1], color='red')
        ax13.twiny().plot(sz.u_Lb[:, 1], x_La[:, 1], color='orange')
        ax23.plot(sz.S_Lb[:, 0], x_La[:, 1], color='red')
        ax23.twiny().plot(sz.S_Lb[:, 1], x_La[:, 1], color='orange')
        ax23.plot(sz.S_La[:, 0], x_La[:, 1], color='green')
        ax23.twiny().plot(sz.S_La[:, 1], x_La[:, 1], color='blue')
#         ax2.plot(x_La[1], sz.F_La[0])
#         ax2.plot(x_La[1], sz.F_La[1])
        plt.show()

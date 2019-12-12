from scipy.optimize import newton, brentq, root

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import traits.api as tr

###### Sympy symbols definition ######
E_ct, E_cc, eps_cr, eps_tu, mu = sp.symbols(
    r'E_ct, E_cc, varepsilon_cr, varepsilon_tu, mu',
    real=True, nonnegative=True
)
eps_cy, eps_cu = sp.symbols(
    r'varepsilon_cy, varepsilon_cu', real=True, nonpositive=True
)
kappa = sp.Symbol('kappa', real=True, nonpositive=True)
eps_top = sp.symbols('varepsilon_top', real=True, nonpositive=True)
eps_bot = sp.symbols('varepsilon_bot', real=True, nonnegative=True)
b, h, z = sp.symbols('b, h, z', nonnegative=True)
eps_sy, E_s = sp.symbols('varepsilon_sy, E_s')
eps = sp.Symbol('varepsilon', real=True)

# Linear profile of strain over the cross section height
eps_z = eps_bot + z * (eps_top - eps_bot) / h

# Express varepsilon_top as a function of kappa and varepsilon_bot
curvature_definition = kappa + eps_z.diff(z)
subs_eps = {eps_top: sp.solve(curvature_definition, eps_top)[0]}
# to return eps on a value z when (kappa, eps_bot) are given
#get_eps_z = sp.lambdify((kappa, eps_bot, z), eps_z.subs(subs_eps), 'numpy')

###### Concrete constitutive law ######
sig_c_eps = sp.Piecewise(
    (0, eps < eps_cu),
    (E_cc * eps_cy, eps < eps_cy),
    (E_cc * eps, eps < 0),
    (E_ct * eps, eps < eps_cr),
    (mu * E_ct * eps_cr, eps < eps_tu),
    (0, eps >= eps_tu)
)

# Stress over the cross section height
sig_c_z = sig_c_eps.subs(eps, eps_z)

# Substitute eps_top to get sig as a function of (kappa, eps_bot, z)
sig_c_z_lin = sig_c_z.subs(subs_eps)

###### Reinforcement constitutive law ######
sig_s_eps = sp.Piecewise(
    (-E_s * eps_sy, eps < -eps_sy),
    (E_s * eps, eps < eps_sy),
    (E_s * eps_sy, eps >= eps_sy)
)

# defining a variable b
z = sp.symbols('z', real=True)
b, b_w, b_f, h_w = sp.symbols('b, b_w, b_f, h_w ', real=True)

b_z = sp.Piecewise(
    (b_w, z < h_w),
    (b_f, z >= h_w)
)

b_data = {
    b_w: 50,
    b_f: 500,
    h_w: 600  # zeta = 0.1
}


class MomentCurvature(tr.HasStrictTraits):
    r'''Class returning the moment curvature relationship.
    '''
    get_b_z = tr.Property(depends_on='z_m')

    @tr.cached_property
    def _get_get_b_z(self):
        return sp.lambdify((z), b_z.subs(b_data), 'numpy')

    h = tr.Float
    b = tr.Float

    model_params = tr.Dict({
        E_ct: 24000, E_cc: 25000,
        eps_cr: 0.001,
        eps_cy: -0.003,
        eps_cu: -0.01,
        mu: 0.33,
        eps_tu: 0.003
    })

    # Number of material points along the height of the cross section
    n_m = tr.Int(200)

    # Reinforcement
    z_j = tr.Array(np.float_, value=[10])
    A_j = tr.Array(np.float_, value=[[np.pi * (16 / 2.)**2]])
    E_j = tr.Array(np.float_, value=[[210000]])
    eps_sy_j = tr.Array(np.float_, value=[[500. / 210000.]])

    z_m = tr.Property(depends_on='n_m, h')

    @tr.cached_property
    def _get_z_m(self):
        return np.linspace(0, self.h, self.n_m)

    kappa_range = tr.Tuple(-0.001, 0.001, 101)

    kappa_t = tr.Property(tr.Array(np.float_), depends_on='kappa_range')

    @tr.cached_property
    def _get_kappa_t(self):
        return np.linspace(*self.kappa_range)

    get_eps_z = tr.Property(depends_on='model_params_items')

    @tr.cached_property
    def _get_get_eps_z(self):
        return sp.lambdify(
            (kappa, eps_bot, z), eps_z.subs(subs_eps), 'numpy'
        )

    get_sig_c_z = tr.Property(depends_on='model_params_items')

    @tr.cached_property
    def _get_get_sig_c_z(self):
        return sp.lambdify(
            (kappa, eps_bot, z), sig_c_z_lin.subs(self.model_params), 'numpy'
        )

    get_sig_s_eps = tr.Property(depends_on='model_params_items')

    @tr.cached_property
    def _get_get_sig_s_eps(self):
        return sp.lambdify((eps, E_s, eps_sy), sig_s_eps, 'numpy')

    # Normal force

    def get_N_s_tj(self, kappa_t, eps_bot_t):
        eps_z_tj = self.get_eps_z(
            kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis],
            self.z_j[np.newaxis, :]
        )
        sig_s_tj = self.get_sig_s_eps(eps_z_tj, self.E_j, self.eps_sy_j)
        return np.einsum('j,tj->tj', self.A_j, sig_s_tj)

    def get_N_c_t(self, kappa_t, eps_bot_t):
        z_tm = self.z_m[np.newaxis, :]
        # @TODO - handle a variable cross section
#         N_z_tm = self.b * self.get_sig_c_z(
#             kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis], z_tm
#         )
        b_z_m = self.get_b_z(z_tm)
        N_z_tm = b_z_m * self.get_sig_c_z(
            kappa_t[:, np.newaxis], eps_bot_t[:, np.newaxis], z_tm
        )
        return np.trapz(N_z_tm, x=z_tm, axis=-1)

    def get_N_t(self, kappa_t, eps_bot_t):
        N_s_t = np.sum(self.get_N_s_tj(kappa_t, eps_bot_t), axis=-1)
        return self.get_N_c_t(kappa_t, eps_bot_t) + N_s_t

    # SOLVER: Get eps_bot to render zero force

    eps_bot_t = tr.Property()
    r'''Resolve the tensile strain to get zero normal force 
    for the prescribed curvature
    '''

    def _get_eps_bot_t(self):
        res = root(lambda eps_bot_t: self.get_N_t(self.kappa_t, eps_bot_t),
                   np.zeros_like(self.kappa_t), tol=1e-10)
        return res.x

    # POSTPROCESSING

    eps_cr = tr.Property()

    def _get_eps_cr(self):
        return np.array([self.model_params[eps_cr]], dtype=np.float_)

    kappa_cr = tr.Property()

    def _get_kappa_cr(self):
        res = root(lambda kappa: self.get_N_t(kappa, self.eps_cr),
                   0.0000001 + np.zeros_like(self.eps_cr), tol=1e-10)
        return res.x

    # Bending moment

    M_s_t = tr.Property()

    def _get_M_s_t(self):
        eps_z_tj = self.get_eps_z(
            self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis],
            self.z_j[np.newaxis, :]
        )
        sig_z_tj = self.get_sig_s_eps(
            eps_z_tj, self.E_j, self.eps_sy_j)
        return -np.einsum('j,tj,j->t', self.A_j, sig_z_tj, self.z_j)

    M_c_t = tr.Property()

    def _get_M_c_t(self):
        z_tm = self.z_m[np.newaxis, :]
#         N_z_tm = self.b * self.get_sig_c_z(
#             self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis], z_tm
#         )
        b_z_m = self.get_b_z(z_tm)
        N_z_tm = b_z_m * self.get_sig_c_z(
            self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis], z_tm
        )
        return -np.trapz(N_z_tm * z_tm, x=z_tm, axis=-1)

    M_t = tr.Property()

    def _get_M_t(self):
        return self.M_c_t + self.M_s_t

    N_s_tj = tr.Property()

    def _get_N_s_tj(self):
        return self.get_N_s_tj(self.kappa_t, self.eps_bot_t)

    eps_tm = tr.Property()

    def _get_eps_tm(self):
        return self.get_eps_z(
            self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis],
            self.z_m[np.newaxis, :],
        )

    sig_tm = tr.Property()

    def _get_sig_tm(self):
        return self.get_sig_c_z(
            self.kappa_t[:, np.newaxis], self.eps_bot_t[:, np.newaxis],
            self.z_m[np.newaxis, :],
        )

    idx = tr.Int(0)

    M_norm = tr.Property()

    def _get_M_norm(self):
        W = (self.b * self.h**2) / 6  # Section modulus
        sig_cr = self.model_params[E_ct] * self.model_params[eps_cr]
        return W * sig_cr

    kappa_norm = tr.Property()

    def _get_kappa_norm(self):
        return self.kappa_cr

    def plot_norm(self, ax1, ax2):
        idx = self.idx
        ax1.plot(self.kappa_t / self.kappa_norm, self.M_t / self.M_norm)
        ax1.plot(self.kappa_t[idx] / self.kappa_norm,
                 self.M_t[idx] / self.M_norm, marker='o')
        ax2.barh(self.z_j, self.N_s_tj[idx, :], color='red', align='center')
        #ax2.fill_between(eps_z_arr[idx,:], z_arr, 0, alpha=0.1);
        ax3 = ax2.twiny()
#        ax3.plot(self.eps_tm[idx, :], self.z_m, color='k', linewidth=0.8)
        ax3.plot(self.sig_tm[idx, :], self.z_m)
        ax3.axvline(0, linewidth=0.8, color='k')
        ax3.fill_betweenx(self.z_m, self.sig_tm[idx, :], 0, alpha=0.1)
        self._align_xaxis(ax2, ax3)

    def plot(self, ax1, ax2):
        idx = self.idx
        ax1.plot(self.kappa_t, self.M_t / (1e6))
        ax1.plot(self.kappa_t[idx], self.M_t[idx] / (1e6), marker='o')
        ax2.barh(self.z_j, self.N_s_tj[idx, :], color='red', align='center')
        #ax2.fill_between(eps_z_arr[idx,:], z_arr, 0, alpha=0.1);
        ax3 = ax2.twiny()
#        ax3.plot(self.eps_tm[idx, :], self.z_m, color='k', linewidth=0.8)
        ax3.plot(self.sig_tm[idx, :], self.z_m)
        ax3.axvline(0, linewidth=0.8, color='k')
        ax3.fill_betweenx(self.z_m, self.sig_tm[idx, :], 0, alpha=0.1)
        self._align_xaxis(ax2, ax3)

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


# Mobasher paper

# For info (left is paper notation, right is notation in this file):
# ------------------------------------------------------------------
# E = E_ct
# E_c = E_cc
# E_s = E_j
# eps_cr = eps_cr
# eps_cu = eps_cu
# eps_tu = eps_tu
# eps_cy = eps_cy
# mu = mu

# gamma = E_cc/E_ct
# omega = eps_cy/eps_cr
# lambda_cu = eps_cu/eps_cr
# beta_tu = eps_tu/eps_cr
# psi = eps_sy_j/eps_cr
# n = E_j/E_ct
# alpha = z_j/h

# r = A_s_c/A_s_t

# rho_g = A_j[0]/A_c # where A_j[0] must be tension steel area
# ------------------------------------------------------------------


# Substituting:
# -----------------------
# Values from parametric study in paper p.11 to draw Fig. 7:

E_ct_ = 24000
eps_cr_ = 0.000125
gamma_ = 1
omega_ = 8.5
lambda_cu_ = 28
beta_tu_ = 160
psi_ = 16
n_ = 8.75
alpha_ = 0.9

mu_ = 0.33
rho_g_ = 0.003

# not given in paper
b_ = 500
h_ = 666
r_ = 0.0

# Substituting formulas:
E_cc_ = gamma_ * E_ct_
eps_cy_ = omega_ * eps_cr_
eps_cu_ = lambda_cu_ * eps_cr_
eps_tu_ = beta_tu_ * eps_cr_
eps_sy_j_ = np.array([psi_ * eps_cr_, psi_ * eps_cr_])
E_j_ = np.array([n_ * E_ct_, n_ * E_ct_])
z_j_ = np.array([h_ * (1 - alpha_), alpha_ * h_])

# concrete section area (TODO use an integration in case of b(z))
A_c_ = 63000  # b_ * h_
A_s_t_ = rho_g_ * A_c_
A_s_c_ = r_ * A_s_t_
A_j_ = np.array([A_s_t_, A_s_c_])  # A_j[0] must be tension steel area

# Creating MomentCurvature object

mc = MomentCurvature(idx=10, n_m=100)
mc.b = b_
mc.h = h_
mc.model_params = {E_ct: E_ct_,
                   E_cc: E_cc_,
                   eps_cr: eps_cr_,
                   eps_cy: eps_cy_,
                   eps_cu: eps_cu_,
                   mu: mu_,
                   eps_tu: eps_tu_}
mc.z_j = z_j_
mc.A_j = A_j_
mc.E_j = E_j_

# If plot_norm is used, use the following:
# mc.kappa_range = (0, mc.kappa_cr * 100, 100)
# Otherwise:
mc.kappa_range = (0, 0.00002, 100)


# print('XXX', mc.kappa_cr)

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
mc.plot(ax1, ax2)
plt.show()

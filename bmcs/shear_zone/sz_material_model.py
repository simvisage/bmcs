'''
Created on Mar 2, 2020

@author: rch
'''

from reporter import RInputRecord
from view import BMCSLeafNode, Vis2D

import numpy as np
import sympy as sp
import traits.api as tr
import traitsui.api as ui


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
# Steel
#=========================================================================

L_f, E_f, f_s_t = sp.symbols('L_f, E_f, f_s_t')

sig_w_f = sp.Piecewise(
    (E_f * w / L_f, E_f * w / L_f <= f_s_t),
    (f_s_t, E_f * w / L_f > f_s_t)
)

d_sig_w_f = sig_w_f.diff(eps)

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
class MaterialModel(BMCSLeafNode, RInputRecord, Vis2D):

    node_name = 'material model'

    f_c = tr.Float(-80.0,
                   MAT=True,
                   unit=r'$\mathrm{MPa}$',
                   symbol=r'f_\mathrm{c}',
                   auto_set=False, enter_set=True,
                   desc='concrete strength')
    E_c = tr.Float(28000,
                   MAT=True,
                   unit=r'$\mathrm{MPa}$',
                   symbol=r'E_\mathrm{c}',
                   auto_set=False, enter_set=True,
                   desc='concrete material stiffness')

    f_t = tr.Float(3.0, MAT=True)
    G_f = tr.Float(0.5, MT=True)

    L_fps = tr.Float(50, MAT=True)
    L_c = tr.Property

    def _get_L_c(self):
        return self.E_c * self.G_f / self.f_t**2

    traits_view = ui.View(
        ui.Item('f_t'),
        ui.Item('f_c'),
        ui.Item('E_c'),
        ui.Item('G_f'),
        ui.Item('L'),
        ui.Item('L_c', style='readonly'),
    )

    tree_view = traits_view

    L = tr.Float(100, PARAM=True)
    co_law_data = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_co_law_data(self):
        return dict(f_t=float(self.f_t),
                    G_f=float(self.G_f),
                    f_c=self.f_c,
                    E_c=self.E_c,
                    L_c=self.L_c,
                    L=self.L)

    get_sig_eps = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_sig_eps(self):
        return sp.lambdify(eps, sig_eps.subs(self.co_law_data), 'numpy')

    get_d_sig_eps = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_d_sig_eps(self):
        return sp.lambdify(eps, d_sig_eps.subs(self.co_law_data), 'numpy')

    #=========================================================================
    # Sig w
    #=========================================================================
    get_sig_w = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_sig_w(self):
        return sp.lambdify(w, sig_w.subs(self.co_law_data), 'numpy')

    get_d_sig_w = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_d_sig_w(self):
        return sp.lambdify(w, d_sig_w.subs(self.co_law_data), 'numpy')

    #=========================================================================
    #
    #=========================================================================
    tau_1 = tr.Float(1.0, MAT=True)
    s_1 = tr.Float(0.000001, MAT=True)
    tau_2 = tr.Float(1.0, MAT=True)
    s_2 = tr.Float(0.02, MAT=True)
    tau_3 = tr.Float(0.0, MAT=True)
    s_3 = tr.Float(1.6, MAT=True)

    bond_law_data = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_bond_law_data(self):
        return dict(tau_1=self.tau_1, s_1=self.s_1,
                    tau_2=self.tau_2, s_2=self.s_2,
                    tau_3=self.tau_3, s_3=self.s_3)

    get_tau_s_plus = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_tau_s_plus(self):
        return sp.lambdify(s, tau_s.subs(self.bond_law_data), 'numpy')

    get_d_tau_s_plus = tr.Property(depends_on='+MAT')

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
    # Steel sig_eps
    #=========================================================================
    L_f = tr.Float(200.0, MAT=True)
    E_f = tr.Float(210000, MAT=True)
    f_s_t = tr.Float(500, MAT=True)

    steel_law_data = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_steel_law_data(self):
        return dict(L_f=float(self.L_f),
                    E_f=float(self.E_f),
                    f_s_t=self.f_s_t)

    get_sig_w_f = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_sig_w_f(self):
        return sp.lambdify(w, sig_w_f.subs(self.steel_law_data), 'numpy')

    get_d_sig_w_f = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_d_sig_eps_f(self):
        return sp.lambdify(w, d_sig_w_f.subs(self.steel_law_data), 'numpy')

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

    def plot_sig_w(self, ax, vot=1.0):

        w_min_expr = (f_c / E_c * L * 2).subs(self.co_law_data)

        w_max_expr = (sp.solve(f_w + f_w.diff(w) * w, w)
                      [0]).subs(self.co_law_data)
        w_max = np.float_(w_max_expr) * 5
        w_min = np.float_(w_min_expr) * 0.5
        w_data = np.linspace(w_min, w_max, 100)
        ax.plot(w_data, self.get_sig_w(w_data), lw=2, color='red')
        ax.fill_between(w_data, self.get_sig_w(w_data),
                        color='red', alpha=0.2)
        ax.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')
        ax.set_title('crack opening law')

    def plot_sig_w_f(self, ax, vot=1.0):

        w_max_expr = (f_s_t / E_f * L_f * 2).subs(self.steel_law_data)
        w_min_expr = 0
        w_max = np.float_(w_max_expr)
        w_min = np.float_(w_min_expr)
        w_data = np.linspace(w_min, w_max, 50)
        ax.plot(w_data, self.get_sig_w_f(w_data), lw=2, color='darkred')
        ax.fill_between(w_data, self.get_sig_w_f(w_data),
                        color='darkred', alpha=0.2)
        ax.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')
        ax.set_title('crack opening law')

    def plot_d_sig_w(self, ax2, vot=1.0):
        w_min_expr = (f_c / E_c * L * 2).subs(self.co_law_data)

        w_max_expr = (sp.solve(f_w + f_w.diff(w) * w, w)
                      [0]).subs(self.co_law_data)
        w_max = np.float_(w_max_expr) * 10
        w_min = np.float_(w_min_expr) * 10
        w_data = np.linspace(w_min, w_max, 100)
        ax2.plot(w_data, self.get_d_sig_w(w_data), color='orange')
        ax2.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax2.set_ylabel(r'$\mathrm{d}\sigma/\mathrm{d}w\;\;\mathrm{[MPa/mm]}$')

    def plot_tau_s(self, ax, vot=1.0):
        s_max = float(s_3.subs(self.bond_law_data))
        s_data = np.linspace(-s_max, s_max, 100)
        ax.plot(s_data, self.get_tau_s(s_data), lw=2, color='blue')
        ax.fill_between(
            s_data, self.get_tau_s(s_data), color='blue', alpha=0.2
        )
        ax.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\tau\;\;\mathrm{[MPa]}$')
        ax.set_title('crack interface law')

    def plot_d_tau_s(self, ax2, vot=1.0):
        s_max = float(s_3.subs(self.bond_law_data))
        s_data = np.linspace(-s_max, s_max, 100)
        ax2.plot(s_data, self.get_d_tau_s(s_data), color='orange')

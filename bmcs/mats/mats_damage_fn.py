
from os.path import join

from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from reporter import RInputRecord
from scipy.optimize import newton
from traits.api import \
    Instance, \
    Float, on_trait_change,\
    HasStrictTraits, Interface, implements, Range, Property
from traitsui.api import \
    View, Item, UItem, VGroup
from view.ui import BMCSLeafNode

import numpy as np


class PlottableFn(RInputRecord):

    plot_min = Float(0.0, input=True,
                     enter_set=True, auto_set=False)
    plot_max = Float(1.0, input=True,
                     enter_set=True, auto_set=False)

    fn = Instance(MFnLineArray)

    def _fn_default(self):
        return MFnLineArray()

    def __init__(self, *args, **kw):
        super(PlottableFn, self).__init__(*args, **kw)
        self.update()

    @on_trait_change('+input')
    def update(self):
        n_vals = 200
        xdata = np.linspace(self.plot_min, self.plot_max, n_vals)
        ydata = self.__call__(xdata)
        self.fn.set(xdata=xdata, ydata=ydata)
        self.fn.replot()

    def write_figure(self, f, rdir, fname):
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % fname)
        self.fn.savefig(join(rdir, fname))

    traits_view = View(UItem('fn'))


class IDamageFn(Interface):

    def __call__(self, k):
        '''get the value of the function'''

    def diff(self, k):
        '''get the first derivative of the function'''


class DamageFn(BMCSLeafNode, PlottableFn):
    #     L_s = Float(1.0,
    #                 MAT=True,
    #                 input=True,
    #                 label="s_0",
    #                 desc="length of the strain softening zone",
    #                 enter_set=True,
    #                 auto_set=False)

    s_0 = Float(0.0004,
                MAT=True,
                input=True,
                symbol="$s_0$",
                desc="elastic strain limit",
                enter_set=True,
                auto_set=False)

    def diff(self, k):
        return self.fn.diff(k)


class JirasekDamageFn(DamageFn):

    node_name = 'Jirasek damage function'

    implements(IDamageFn)

    s_f = Float(0.001,
                MAT=True,
                input=True,
                symbol="$s_f$",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    plot_max = 1e-2

    def __call__(self, kappa):
        s_0 = self.s_0
        s_f = self.s_f
        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]
        omega[d_idx] = 1. - s_0 / k * np.exp(-1 * (k - s_0) / (s_f - s_0))
        return omega

    def diff(self, kappa):

        nz_ix = np.where(kappa != 0.0)[0]
        s_0 = self.s_0
        s_f = self.s_f

        domega_dkappa = np.zeros_like(kappa)
        kappa_nz = kappa[nz_ix]
        domega_dkappa[nz_ix] = ((s_0 / (kappa_nz * s_f)) -
                                (s_0 / kappa_nz**2)) * np.exp(-(kappa_nz - s_0) / (s_f))

        return domega_dkappa

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly', full_size=True, resizable=True),
                Item('s_f'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class LiDamageFn(DamageFn):

    node_name = 'Li damage function'

    implements(IDamageFn)

    alpha_1 = Range(value=1., low=0.0, high=1.0,
                    MAT=True,
                    input=True,
                    symbol=r'$\alpha_1$',
                    desc="parameter controls the damage function",
                    enter_set=True,
                    auto_set=False)

    alpha_2 = Float(2000.,
                    MAT=True,
                    input=True,
                    symbol=r'$\alpha_2$',
                    desc="parameter controls the damage function",
                    enter_set=True,
                    auto_set=False)

    plot_max = 1e-2

    def __call__(self, kappa):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        s_0 = self.s_0
        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]
        omega[d_idx] = 1. / \
            (1. + np.exp(-1. * alpha_2 * (k - s_0) + 6.)) * alpha_1
        return omega

    def diff(self, kappa):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        s_0 = self.s_0
        return ((alpha_1 * alpha_2 *
                 np.exp(-1. * alpha_2 * (kappa - s_0) + 6.)) /
                (1 + np.exp(-1. * alpha_2 * (kappa - s_0) + 6.)) ** 2)

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly', full_size=True, resizable=True),
                Item('alpha_1'),
                Item('alpha_2'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class AbaqusDamageFn(DamageFn):

    node_name = 'Abaqus damage function'

    implements(IDamageFn)

    s_u = Float(0.003,
                MAT=True,
                input=True,
                symbol="$s_u$",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    alpha = Float(0.1,
                  MAT=True,
                  input=True,
                  symbol="$\alpha$",
                  desc="parameter controlling the slop of damage",
                  enter_set=True,
                  auto_set=False)

    plot_max = 1e-3

    def __call__(self, kappa):
        s_0 = self.s_0
        s_u = self.s_u
        alpha = self.alpha

        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa > s_0)[0]
        k = kappa[d_idx]

        sk = (k - s_0) / (s_u - s_0)
        frac = (1 - np.exp(-alpha * sk)) / (1 - np.exp(-alpha))

        omega[d_idx] = 1 - s_0 / k * (1 - frac)
        omega[np.where(omega > 1.0)] = 1.0
        return omega

    def diff(self, kappa):
        s_0 = self.s_0
        s_u = self.s_u
        alpha = self.alpha
        d_g_eps = (- s_0 * np.exp(alpha * (kappa - s_0) /
                                  (s_u - s_0)) * ((s_u - s_0) *
                                                  np.exp(alpha * (kappa - s_0) /
                                                         (s_u - s_0)) + np.exp(alpha) * (alpha * kappa - s_u + s_0))
                   / ((np.exp(alpha) - 1) * kappa**2 * (s_u - s_0)))
        d_g_eps[np.where(kappa - s_0 < 0)] = 0.0
        return d_g_eps

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly',
                     full_size=True, resizable=True),
                Item('s_u'),
                Item('alpha'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


class FRPDamageFn(DamageFn):

    node_name = 'FRP damage function'

    implements(IDamageFn)

    b = Float(10.4,
              MAT=True,
              input=True,
              symbol="$b$",
              desc="parameter controls the damage function",
              enter_set=True,
              auto_set=False)

    Gf = Float(1.19,
               MAT=True,
               input=True,
               symbol="$G_\mathrm{f}$",
               desc="fracture energy",
               enter_set=True,
               auto_set=False)

    plot_max = 0.5

    def __init__(self, *args, **kw):
        super(FRPDamageFn, self).__init__(*args, **kw)
        self._update_dependent_params()

    E_b = Float

    @on_trait_change('b, Gf')
    def _update_dependent_params(self):

        self.E_b = 1.734 * self.Gf * self.b ** 2.0
        # calculation of s_0, implicit function solved using Newton method

        def f_s(s_0): return s_0 / \
            (np.exp(- self.b * s_0) - np.exp(-2.0 * self.b * s_0)) - \
            2.0 * self.b * self.Gf / self.E_b
        self.s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

    def __call__(self, kappa):

        b = self.b
        Gf = self.Gf
        Eb = self.E_b  # 1.734 * Gf * b**2
        s_0 = self.s_0
        # calculation of s_0, implicit function solved using Newton method

#         def f_s(s_0): return s_0 / \
#             (np.exp(-b * s_0) - np.exp(-2.0 * b * s_0)) - 2.0 * b * Gf / Eb
#         s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]

        omega[d_idx] = 1 - \
            (2.0 * b * Gf * (np.exp(-b * k) - np.exp(-2.0 * b * k))) / (k * Eb)

        return omega

    def diff(self, kappa):

        nz_ix = np.where(kappa != 0.0)[0]

        b = self.b
        Gf = self.Gf
        Eb = 1.734 * Gf * b**2

        domega_dkappa = np.zeros_like(kappa)
        kappa_nz = kappa[nz_ix]
        domega_dkappa[nz_ix] = (
            (2.0 * b * Gf *
             (np.exp(-b * kappa_nz) -
              np.exp(-2.0 * b * kappa_nz))
             ) / (Eb * kappa_nz**2.0) -
            (2.0 * b * Gf *
             (-b * np.exp(-b * kappa_nz) +
              2.0 * b * np.exp(-2.0 * b * kappa_nz))) /
            (Eb * kappa_nz)
        )
        return domega_dkappa

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly', full_size=True, resizable=True),
                Item('E_b', style='readonly', full_size=True, resizable=True),
                Item('b'),
                Item('Gf'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


if __name__ == '__main__':
    #ld = LiDamageFn()
    #ld = JirasekDamageFn()
    ld = FRPDamageFn()
    ld.configure_traits()

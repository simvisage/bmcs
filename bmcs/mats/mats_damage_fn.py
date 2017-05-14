
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import \
    Instance, \
    Float, on_trait_change,\
    HasStrictTraits, Interface, implements, Range
from traitsui.api import \
    View, Item, UItem, VGroup
from view.ui import BMCSLeafNode

import numpy as np


class PlottableFn(HasStrictTraits):

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

    traits_view = View(UItem('fn'))


class IDamageFn(Interface):

    def __call__(self, k):
        '''get the value of the function'''

    def diff(self, k):
        '''get the first derivative of the function'''


class DamageFn(BMCSLeafNode, PlottableFn):
    s_0 = Float(0.0004,
                MAT=True,
                input=True,
                label="s_0",
                desc="parameter controls the damage function",
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
                label="s_f",
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
        s_0 = self.s_0
        s_f = self.s_f
        return ((s_0 / (kappa * s_f)) - (s_0 / kappa)) * np.exp(-(kappa - s_0) / (s_f))

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
                    label="alpha_1",
                    desc="parameter controls the damage function",
                    enter_set=True,
                    auto_set=False)

    alpha_2 = Float(2000.,
                    MAT=True,
                    input=True,
                    label="alpha_2",
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
        return ((self.alpha_1 * self.alpha_2 *
                 np.exp(-1. * self.alpha_2 * kappa + 6.)) /
                (1 + np.exp(-1. * self.alpha_2 * kappa + 6.)) ** 2)

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
                label="s_u",
                desc="parameter controls the damage function",
                enter_set=True,
                auto_set=False)

    alpha = Float(0.1,
                  MAT=True,
                  input=True,
                  label="alpha",
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
        return (- s_0 * np.exp(alpha * (kappa - s_0) /
                               (s_u - s_0)) * ((s_u - s_0) *
                                               np.exp(alpha * (kappa - s_0) /
                                                      (s_u - s_0)) + np.exp(alpha) * (alpha * kappa - s_u + s_0))
                / ((np.exp(alpha) - 1) * kappa**2 * (s_u - s_0)))

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

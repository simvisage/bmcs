
from os.path import join

from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from reporter import RInputRecord
from scipy.optimize import newton
from traits.api import \
    Instance, Str, \
    Float, on_trait_change,\
    Interface, provides, Range, Property, Button, \
    Array, WeakRef
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

    def plot(self, ax, **kw):
        n_vals = 200
        xdata = np.linspace(self.plot_min, self.plot_max, n_vals)
        ydata = np.zeros_like(xdata)
        f_idx = self.get_f_trial(xdata)
        ydata[f_idx] = self.__call__(xdata[f_idx])
        color = kw.pop('color', 'green')
        ax.plot(xdata, ydata, color=color, **kw)

    @on_trait_change('+input')
    def update(self):
        n_vals = 200
        xdata = np.linspace(self.plot_min, self.plot_max, n_vals)
        ydata = np.zeros_like(xdata)
        f_idx = self.get_f_trial(xdata)
        if len(f_idx) > 0:
            ydata[f_idx] = self.__call__(xdata[f_idx])
        self.fn.set(xdata=xdata, ydata=ydata)
        self.fn.replot()

    def write_figure(self, f, rdir, rel_study_path):
        fname = 'fig_' + self.node_name.replace(' ', '_') + '.pdf'
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % join(rel_study_path, fname))
        self.fn.savefig(join(rdir, fname))

    traits_view = View(UItem('fn'))


class IDamageFn(Interface):

    def __call__(self, k):
        '''get the value of the function'''

    def diff(self, k):
        '''get the first derivative of the function'''

    def get_f_trial(self, k):
        '''get the map of indexes with inelastic behavior'''


class DamageFn(BMCSLeafNode, PlottableFn):

    mats = WeakRef

    s_0 = Float(0.0004,
                MAT=True,
                input=True,
                symbol="s_0",
                desc="elastic strain limit",
                unit='mm',
                enter_set=True,
                auto_set=False)

    def diff(self, k):
        return self.fn.diff(k)

    latex_eq = Str(None)

    def _repr_latex_(self):
        return self.latex_eq + super(DamageFn, self)._repr_latex_()


class GfDamageFn(DamageFn):
    '''Class defining the damage function coupled with the fracture 
    energy of a cohesive crack model.
    '''
    node_name = 'damage function Gf'

    L_s = Float(1.0,
                MAT=True,
                input=True,
                label="L_s",
                desc="Length of the softening zone",
                enter_set=True,
                auto_set=False)

    E_ = Float(34000.0,
               MAT=True,
               input=True,
               label="E",
               desc="Young's modulus",
               enter_set=True,
               auto_set=False)

    E = Property()

    def _get_E(self):
        if self.mats:
            return self.mats.E
        else:
            return self.E_

    f_t = Float(4.5,
                MAT=True,
                input=True,
                label="f_t",
                desc="Tensile strength",
                enter_set=True,
                auto_set=False)

    f_t_Em = Array(np.float_, value=None)

    G_f = Float(0.004,
                MAT=True,
                input=True,
                label="G_f",
                desc="Fracture energy",
                enter_set=True,
                auto_set=False)

    eps_0 = Property()

    def _get_eps_0(self):
        return self.f_t / self.E

    eps_ch = Property()

    def _get_eps_ch(self):
        return self.G_f / self.f_t

    plot_max = Property(depends_on='G_f,f_t,E')

    def _get_plot_max(self):
        return self.eps_ch * self.L_s * 3.0

    def get_f_trial(self, eps_eq_Em):
        f_t = self.f_t
        if len(self.f_t_Em) > 0:
            f_t = self.f_t * self.f_t_Em
        eps_0 = f_t / self.E
        return np.where(eps_eq_Em - eps_0 > 0)

    def __call__(self, kappa):
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        eps_0 = self.eps_0
        return (
            1 - f_t * np.exp(-f_t * (kappa - eps_0) * L_s / G_f)
            / (E * kappa)
        )

    def diff(self, kappa):
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        eps_0 = self.eps_0
        return (
            f_t * np.exp(L_s * (eps_0 - kappa) * f_t / G_f)
            / (E * G_f * kappa**2) * (G_f + L_s * kappa * f_t)
        )

    traits_view = View(
        VGroup(
            VGroup(
                Item('f_t'),
                Item('G_f'),
                Item('E', style='readonly'),
                Item('L_s', style='readonly'),
                Item('s_0', style='readonly',
                     full_size=True, resizable=True),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


@provides(IDamageFn)
class JirasekDamageFn(DamageFn):

    node_name = 'Jirasek damage function'

    s_f = Float(0.001,
                MAT=True,
                input=True,
                symbol="s_\mathrm{f}",
                unit='mm/mm',
                desc="derivative of the damage function at the onset of damage",
                enter_set=True,
                auto_set=False)

    plot_max = 1e-2

    def get_f_trial(self, eps_eq_Em):
        eps_0 = self.s_0
        return np.where(eps_eq_Em - eps_0 > 0)

    def __call__(self, kappa):
        s_0 = self.s_0
        s_f = self.s_f
        omega = np.zeros_like(kappa, dtype=np.float_)
        I = np.where(kappa >= s_0)
        k_I = kappa[I]
        omega[I] = 1. - s_0 / k_I * np.exp(-1 * (k_I - s_0) / (s_f - s_0))
        return omega

    def diff(self, kappa):
        s_0 = self.s_0
        s_f = self.s_f
        I = np.where(kappa >= s_0)
        k_I = kappa[I]
        domega_dkappa = np.zeros_like(kappa)
        domega_dkappa[I] = (
            s_0 * (-k_I + s_0 - s_f) * np.exp((k_I - s_0) /
                                              (s_0 - s_f)) / (k_I**2 * (s_0 - s_f))
        )
        return domega_dkappa

    latex_eq = Str(r'''Damage function (Jirasek)
        \begin{align}
        \omega &= g(\kappa) 
        = 1 - \left[\frac{s_0}{\kappa} \exp \left(- \frac{\kappa 
        - s_0}{s_f - s_0}\right)\right]
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        ''')

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', full_size=True, resizable=True),
                Item('s_f'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


@provides(IDamageFn)
class LiDamageFn(DamageFn):

    node_name = 'Li damage function'

    latex_eq = Str(r'''Damage function (Li)
        \begin{align}
        \omega = g(\kappa) = \frac{\alpha_1}{1 + \exp(-\alpha_2 \kappa + 6 )}
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        ''')

    alpha_1 = Range(value=1., low=0.0, high=1.0,
                    MAT=True,
                    input=True,
                    symbol=r'\alpha_1',
                    unit='-',
                    desc="parameter controlling the shape of the damage function",
                    enter_set=True,
                    auto_set=False)

    alpha_2 = Float(2000.,
                    MAT=True,
                    input=True,
                    symbol=r'\alpha_2',
                    unit='-',
                    desc="parameter controlling the shape of the damage function",
                    enter_set=True,
                    auto_set=False)

    plot_max = 1e-2

    def get_f_trial(self, eps_eq_Em):
        eps_0 = self.s_0
        return np.where(eps_eq_Em - eps_0 > 0)

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


@provides(IDamageFn)
class AbaqusDamageFn(DamageFn):

    node_name = 'Abaqus damage function'

    latex_eq = Str(r'''Damage function (Abaqus)
        \begin{align}
        \omega = g(\kappa) = 1 -\left(\frac{s_0}{\kappa}\right)\left[ 1 - \frac{1 - \exp(- \alpha(\frac{\kappa - s_0}{s_u - s_0})}{1 - \exp(-\alpha)}  \right]
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        ''')

    s_u = Float(0.003,
                MAT=True,
                input=True,
                symbol="s_u",
                unit='mm',
                desc="parameter of the damage function",
                enter_set=True,
                auto_set=False)

    alpha = Float(0.1,
                  MAT=True,
                  input=True,
                  symbol=r"\alpha",
                  desc="parameter controlling the slope of damage",
                  unit='-',
                  enter_set=True,
                  auto_set=False)

    plot_max = 1e-3

    def get_f_trial(self, kappa):
        s_0 = self.s_0
        return np.where(kappa > s_0)[0]

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
        denom = ((np.exp(alpha) - 1) * kappa**2 * (s_u - s_0))
        denom[np.where(np.fabs(denom) < 1e-5)] = 1e-8
        d_g_eps = (
            - s_0 * np.exp(alpha * (kappa - s_0) / (s_u - s_0))
            * (
                (s_u - s_0) * np.exp(alpha * (kappa - s_0) / (s_u - s_0))
                + np.exp(alpha) * (alpha * kappa - s_u + s_0)
            )
            / denom
        )
        d_g_eps[np.where(kappa - s_0 < 0)] = 0.0
        return d_g_eps

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0',
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


@provides(IDamageFn)
class FRPDamageFn(DamageFn):

    node_name = 'FRP damage function'

    B = Float(10.4,
              MAT=True,
              input=True,
              symbol="B",
              unit='mm$^{-1}$',
              desc="parameter controlling the damage maximum stress level",
              enter_set=True,
              auto_set=False)

    Gf = Float(1.19,
               MAT=True,
               input=True,
               symbol="G_\mathrm{f}",
               unit='N/mm',
               desc="fracture energy",
               enter_set=True,
               auto_set=False)

    plot_max = 0.5

    def __init__(self, *args, **kw):
        super(FRPDamageFn, self).__init__(*args, **kw)
        self._update_dependent_params()

    E_bond = Float(0.0)

    E_b = Property(Float)

    def _get_E_b(self):
        return self.mats.E_b

    def _set_E_b(self, value):
        self.E_bond = value
        self.mats.E_b = value

    @on_trait_change('B, Gf')
    def _update_dependent_params(self):
        self.E_b = 1.734 * self.Gf * self.B ** 2.0
        # calculation of s_0, implicit function solved using Newton method

        def f_s(s_0): return s_0 / \
            (np.exp(- self.B * s_0) - np.exp(-2.0 * self.B * s_0)) - \
            2.0 * self.B * self.Gf / self.E_b
        self.s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

    def get_f_trial(self, eps_eq_Em):
        eps_0 = self.s_0
        return np.where(eps_eq_Em - eps_0 > 0)

    def __call__(self, kappa):

        b = self.B
        Gf = self.Gf
        Eb = self.E_b  # 1.734 * Gf * b**2
        s_0 = self.s_0
        # calculation of s_0, implicit function solved using Newton method

#         def f_s(s_0): return s_0 / \
#             (np.exp(-b * s_0) - np.exp(-2.0 * b * s_0)) - 2.0 * b * Gf / Eb
#         s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

        omega = np.zeros_like(kappa, dtype=np.float_)
        I = np.where(kappa >= s_0)[0]
        kappa_I = kappa[I]

        omega[I] = 1 - \
            (2.0 * b * Gf * (np.exp(-b * kappa_I)
                             - np.exp(-2.0 * b * kappa_I))) / (kappa_I * Eb)

        return omega

    def diff(self, kappa):

        nz_ix = np.where(kappa != 0.0)[0]

        b = self.B
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

    latex_eq = r'''Damage function (FRP)
        \begin{align}
        \omega = g(\kappa) = 
        1 - {\frac {{\exp(-2\,Bs)}-{\exp(-Bs)}}{Bs}}
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        '''

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly', full_size=True, resizable=True),
                Item('E_bond', style='readonly',
                     full_size=True, resizable=True),
                Item('B'),
                Item('Gf'),
                Item('plot_max'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


@provides(IDamageFn)
class MultilinearDamageFn(DamageFn):

    node_name = 'Multilinear damage function'

    s_data = Str('0,1', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='s',
                 desc='slip values',
                 auto_set=True, enter_set=False)

    omega_data = Str('0,0', tooltip='Comma-separated list of damage values',
                     MAT=True, unit='-', symbol=r'\omega',
                     desc='shear stress values',
                     auto_set=True, enter_set=False)

    def get_f_trial(self, eps_eq_Em):
        eps_0 = self.damage_law.xdata[1]
        return np.where(eps_eq_Em - eps_0 > 0)[0]

    s_omega_table = Property

    def _set_s_omega_table(self, data):
        s_data, omega_data = data
        if len(s_data) != len(omega_data):
            raise ValueError('s array and tau array must have the same size')
        self.damage_law.set(xdata=s_data,
                            ydata=omega_data)

    update_damage_law = Button(label='update bond-slip law')

    def _update_damage_law_fired(self):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        omega_data = np.fromstring(self.omega_data, dtype=np.float_, sep=',')
        if len(s_data) != len(omega_data):
            raise ValueError('s array and tau array must have the same size')
        self.damage_law.set(xdata=s_data,
                            ydata=omega_data)
        self.damage_law.replot()

    damage_law = Instance(MFnLineArray)

    def _damage_law_default(self):
        return MFnLineArray(
            xdata=[0.0, 1.0],
            ydata=[0.0, 0.0],
            plot_diff=False)

    def __call__(self, kappa):
        shape = kappa.shape
        return self.damage_law(kappa.flatten()).reshape(*shape)

    def diff(self, kappa):
        shape = kappa.shape
        return self.damage_law.diff(kappa.flatten()).reshape(*shape)

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_data', full_size=True, resizable=True),
                Item('omega_data'),
                UItem('update_damage_law')
            ),
            UItem('damage_law@')
        )
    )

    tree_view = traits_view


if __name__ == '__main__':
    #ld = LiDamageFn()
    #ld = GfDamageFn()
    #mld = MultilinearDamageFn()
    #mld = FRPDamageFn(Gf=100)
    mld = JirasekDamageFn()
    mld.configure_traits()

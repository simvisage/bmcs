from bmcs.mats.mats_damage_fn import \
    IDamageFn, LiDamageFn, JirasekDamageFn, AbaqusDamageFn,\
    PlottableFn, DamageFn
from traits.api import Float, Property
from traitsui.api import View, Item, VGroup, UItem

import numpy as np
import pylab as p


class GfDamageFn(DamageFn):
    '''Class defining the damage function coupled with the fracture 
    energy of a cohesive crack model.
    '''
    L_s = Float(1.0,
                MAT=True,
                input=True,
                label="L_s",
                desc="Length of the softening zone",
                enter_set=True,
                auto_set=False)

    E = Float(34000.0,
              MAT=True,
              input=True,
              label="E",
              desc="Young's modulus",
              enter_set=True,
              auto_set=False)

    f_t = Float(4.5,
                MAT=True,
                input=True,
                label="f_t",
                desc="Tensile strength",
                enter_set=True,
                auto_set=False)

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

    def __call__(self, eps):
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        eps_0 = self.eps_0
        g_eps = 1 - f_t * np.exp(-f_t * (eps - eps_0) * L_s / G_f) \
            / (E * eps)
        g_eps[np.where(eps - eps_0 < 0)] = 0
        return g_eps

    def diff(self, eps):
        L_s = self.L_s
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        eps_0 = self.eps_0
        d_g_eps = (f_t * np.exp(L_s * (eps_0 - eps) * f_t / G_f)
                   / (E * G_f * eps**2) *
                   (G_f + L_s * eps * f_t))

        d_g_eps[np.where(eps - eps_0 < 0)] = 0.0
        return d_g_eps

    traits_view = View(
        VGroup(
            VGroup(
                Item('s_0', style='readonly',
                     full_size=True, resizable=True),
                Item('f_t'),
                Item('G_f'),
                Item('E'),
            ),
            VGroup(
                UItem('fn@', height=300)
            )
        )
    )

    tree_view = traits_view


f_t = 2.4
G_f = 0.090
E = 20000.0
eps_0 = f_t / E
eps_ch = G_f / f_t
L_ch = E * G_f / f_t**2

print('L_ch', L_ch)

L = 45.0
u_max = 0.15
eps_max = eps_ch
eps = np.linspace(0, eps_max, 100)

omega_fn_abaqus = AbaqusDamageFn(s_0=eps_0, s_u=0.03)
omega_fn_gf = GfDamageFn(G_f=G_f, f_t=f_t, E=E)

omega_fn = omega_fn_gf

n_T = 2000
K_max = 200

for N in [1.00001, 2, 3, 4, 5, 9]:
    sig_t = []
    eps_t = []
    L_el = (N - 1.0) / N * L
    L_s = 1.0 / N * L
    eps_s = 0.0
    eps_s_arr = np.array([eps_s], dtype=np.float_)
    u_t = np.linspace(0, u_max, n_T)
    omega_fn.L_s = L_s
    for u in u_t:
        for K in range(K_max):
            R = 1 / L_el * (u - eps_s_arr * L_s) - \
                (1 - omega_fn(eps_s_arr)) * eps_s_arr
            if np.fabs(R) < 1e-8:
                break
            dR = -L_s / L_el + omega_fn.diff(eps_s_arr) * \
                eps_s_arr - (1 - omega_fn(eps_s_arr))
            d_eps_s = -R / dR
            eps_s_arr += d_eps_s
            if K == K_max - 1:
                raise ValueError('No convergence')
        sig = ((1.0 - omega_fn(eps_s_arr)) * E * eps_s_arr)[0]
        sig_t.append(sig)
        eps_t.append(u / L)
    p.plot(eps_t, sig_t)
p.show()

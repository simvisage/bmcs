'''
Created on May 30, 2018

@author: rch
'''

from scipy import interpolate as ip
from view.plot2d import Viz2D, Vis2D

import numpy as np
import traits.api as tr


class Vis2DEnergy(Vis2D):

    U_bar_t = tr.List()

    def setup(self):
        self.U_bar_t = []

    def update(self):
        mats = self.sim.mats
        tstep = self.sim.tstep
        xdomain = self.sim.xdomain
        U = tstep.U_k
        t = tstep.t_n1
        fets = xdomain.fets
        n_c = fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[xdomain.I_Ei]
        d = xdomain.integ_factor  # thickness
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            xdomain.B_Eimabc, U_Eia
        )
        sig_Emab, _ = mats.get_corr_pred(
            eps_Emab, t, **tstep.fe_domain[0].state_n)
        w_m = fets.w_m
        det_J_Em = xdomain.det_J_Em
        U_bar = d / 2.0 * np.einsum(
            'm,Em,Emab,Emab',
            w_m, det_J_Em, sig_Emab, eps_Emab
        )
        self.U_bar_t.append(U_bar)

    def get_t(self):
        return self.sim.hist.t

    def get_w(self):
        _, w = self.sim.hist['Pw'].Pw
        return w

    def get_W_t(self):
        P, w = self.sim.hist['Pw'].Pw
        return [
            np.trapz(P[:i + 1], w[:i + 1])
            for i, _ in enumerate(w)
        ]

    def get_G_t(self):
        U_bar_t = np.array(self.U_bar_t, dtype=np.float_)
        W_t = self.get_W_t()
        G = W_t - U_bar_t
        return G

    def get_dG_t(self):
        t = self.get_t()
        G = self.get_G_t()
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)


class Viz2DEnergy(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot,
             label_U='U(t)', label_W='W(t)',
             color_U='blue', color_W='red'):
        t = self.vis2d.get_t()
        U_bar_t = self.vis2d.U_bar_t
        W_t = self.vis2d.get_W_t()
        ax.plot(t, W_t, color=color_W, label=label_W)
        ax.plot(t, U_bar_t, color=color_U, label=label_U)
        ax.fill_between(t, W_t, U_bar_t, facecolor='gray', alpha=0.5,
                        label='G(t)')
        ax.set_ylabel('energy [Nmm]')
        ax.set_xlabel('control displacement [mm]')
        ax.legend()


class Viz2DEnergyReleasePlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'released energy'

    def plot(self, ax, vot, *args, **kw):
        w = self.vis2d.get_w()
        G_t = self.vis2d.get_G_t()
        ax.plot(w, G_t, color='black', linewidth=2, label='G')
        ax.fill_between(w, 0, G_t, facecolor='gray', alpha=0.5)
        ax.legend()
#         dG_ax = ax  # ax.twinx()
#         dG_t = self.vis2d.get_dG_t()
#         dG_ax.plot(t, dG_t, color='black', label='dG/dt')
#         dG_ax.fill_between(t, 0, dG_t, facecolor='blue', alpha=0.2)

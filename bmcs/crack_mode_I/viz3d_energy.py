'''
Created on May 30, 2018

@author: rch
'''

from scipy import interpolate as ip
from view.plot2d import Viz2D, Vis2D

import numpy as np
import traits.api as tr


class Vis2DEnergy(Vis2D):

    model = tr.WeakRef
    tloop = tr.WeakRef

    def setup(self, tloop):
        self.tloop = tloop

    U_bar_t = tr.List()

    def update(self, U, t):
        ts = self.tloop.ts
        mats = ts.mats
        fets = ts.fets
        n_c = fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[ts.I_Ei]
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab', ts.B_Eimabc, U_Eia
        )
        deps_Emab = np.zeros_like(eps_Emab)
        D_Emabef, sig_Emab = mats.get_corr_pred(
            eps_Emab, deps_Emab, t, t, False, False,
            **ts.state_arrays
        )
        w_m = fets.ip_weights
        det_J_Em = ts.det_J_Em
        d = ts.integ_factor  # thickness
        U_bar = d / 2.0 * np.einsum('m,Em,Emab,Emab',
                                    w_m, det_J_Em, sig_Emab, eps_Emab)
        self.U_bar_t.append(U_bar)

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    def get_W_t(self):
        P, W = self.model.get_PW()
        W_t = []
        for i, w in enumerate(W):
            W_t.append(np.trapz(P[:i + 1], W[:i + 1]))
        return W_t

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
        ax.set_xlabel('time [-]')
        ax.legend()


class Viz2DEnergyRatesPlot(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'line plot'

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()
        G_t = self.vis2d.get_G_t()
        ax.plot(t, G_t, color='black', label='G')
        ax.fill_between(t, 0, G_t, facecolor='orange', alpha=0.2)
        ax.legend()
#         dG_ax = ax  # ax.twinx()
#         dG_t = self.vis2d.get_dG_t()
#         dG_ax.plot(t, dG_t, color='black', label='dG/dt')
#         dG_ax.fill_between(t, 0, dG_t, facecolor='blue', alpha=0.2)

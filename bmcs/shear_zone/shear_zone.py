'''
Created on Jan 19, 2020

@author: rch

Test the case of a straight crack in the middle of a zone.

Design issues

- Simulator contains a time loop with a timestepper.
- Time stepper represents the corrector predictor  
  scheme running along the time line from 0 to 1
- History records the state of the time stepper.

'''

import copy

from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import root
from simulator.api import \
    TStep, Hist, TLoop
from view import BMCSLeafNode, Vis2D, BMCSWindow, PlotPerspective
from view.ui.bmcs_tree_node import BMCSTreeNode

import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr
import traitsui.api as ui

from .sz_material_model import IMaterialModel, MaterialModel
from .sz_rotation_kinematics import get_phi
from .sz_theta import get_theta_0, get_theta_f


EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1
Z = np.array([0, 0, 1], dtype=np.float_)


class TLoopSZ(TLoop):

    n_seg = tr.Int(5, auto_set=False, enter_set=True)
    '''Number of crack propagation steps.
    '''

    def eval(self):
        ts = self.tstep
        n_seg = self.n_seg
        delta_t = 1 / (n_seg)
        t_n1 = 0
        for seg in range(n_seg):
            print('seg', seg, t_n1)
            try:
                ts.make_iter()
            except ValueError:
                print('No convergence')
                ts.X[1] = 0
                break
            t_n1 += delta_t
            self.tstep.t_n1 = t_n1
            ts.make_incr()
            self.tline.val = min(t_n1, self.tline.max)


class XDomain(BMCSLeafNode, Vis2D):
    '''Represent the current crack state.
    This objects represents the discretization of the domain using 
    segments / finite elements
    '''
    node_name = 'crack domain'

    ts = tr.WeakRef

    #=========================================================================
    # Primary state variables
    #=========================================================================
    x_rot_1 = tr.Float(50, state_changed=True)
    theta = tr.Float(0, state_changed=True)

    @tr.on_trait_change('+state_changed')
    def set_state_changed(self):
        self.state_changed = True
    #=========================================================================
    # Discretization parameters
    #=========================================================================
    n_J = tr.Int(10)
    '''Number of nodes along the uncracked zone
    '''
    n_m = tr.Int(8)
    '''Number of integration points within a segment
    '''

    tree_view = ui.View(
        ui.Item('L_fps', style='readonly'),
        ui.Item('n_m', style='readonly'),
        ui.Item('n_J', style='readonly'),
        ui.Item('theta', style='readonly'),
        ui.Item('x_tip_a', style='readonly'),
        ui.Item('x_rot_a', style='readonly'),
    )
    eta = tr.DelegatesTo('ts')
    #=========================================================================
    # Geometry - used for visualization and constraits of crack propagation.
    #=========================================================================
    x_Ca = tr.Property(depends_on='input_changed')
    '''Corner coordinates'''
    @tr.cached_property
    def _get_x_Ca(self):
        L = self.ts.L
        H = self.ts.H
        return np.array([[0, L, L, 0],
                         [0, 0, H, H]], dtype=np.float_).T

    C_Li = tr.Property(depends_on='input_changed')
    '''Lines'''
    @tr.cached_property
    def _get_C_Li(self):
        return np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)

    #=========================================================================
    # Discretization / state domain
    #=========================================================================
    state_changed = tr.Event
    '''Register the state change event to trigger recalculation.
    '''

    x_t_Ia = tr.Array(dtype=np.float_, value=[])

    def _x_t_Ia_default(self):
        return np.array([[self.ts.initial_crack_position, 0]],
                        dtype=np.float_)

    x_Ia = tr.Property(depends_on='state_changed')
    '''Nodes along the crack path including the fps segment'''
    @tr.cached_property
    def _get_x_Ia(self):
        return np.vstack([self.x_t_Ia, self.x_tip_a[np.newaxis, :]])

    I_Li = tr.Property(depends_on='state_changed')
    '''Crack segments'''
    @tr.cached_property
    def _get_I_Li(self):
        N_I = np.arange(len(self.x_Ia))
        I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
        return I_Li

    x_Ja = tr.Property(depends_on='state_changed')
    '''Uncracked vertical section'''
    @tr.cached_property
    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.x_Ca[-1, 1], self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    xx_Ka = tr.Property(depends_on='state_changed')
    '''Integrated section'''
    @tr.cached_property
    def _get_xx_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    x_Ka = tr.Property(depends_on='state_changed')
    '''Integration points'''
    @tr.cached_property
    def _get_x_Ka(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.xx_Ka[1:] - self.xx_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.xx_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.xx_Ka[[-1], :]])

    K_Li = tr.Property(depends_on='state_changed')
    '''Crack segments'''
    @tr.cached_property
    def _get_K_Li(self):
        N_K = np.arange(len(self.x_Ka))
        K_Li = np.array([N_K[:-1], N_K[1:]], dtype=np.int_).T
        return K_Li

    x_Lb = tr.Property(depends_on='state_changed')
    '''Midpoints'''
    @tr.cached_property
    def _get_x_Lb(self):
        return np.sum(self.x_Ka[self.K_Li], axis=1) / 2

    L_fps = tr.Property(tr.Float)
    '''Length of the fracture process segment.
    '''

    def _get_L_fps(self):
        return self.ts.mm.L_fps

    #=========================================================================
    # Fracture process segment
    #=========================================================================
    T_fps_a = tr.Property(tr.Array, depends_on='state_changed')
    '''Orientation matrix of the crack propagation segment
    '''
    @tr.cached_property
    def _get_T_fps_a(self):
        return np.array([-np.sin(self.theta), np.cos(self.theta)],
                        dtype=np.float_)

    x_tip_a = tr.Property(tr.Array, depends_on='state_changed')
    '''Unknown position of the crack tip. Depends on the sought
    fracture process segment orientation $\theta$
    '''
    @tr.cached_property
    def _get_x_tip_a(self):
        return self.x_fps_a + self.L_fps * self.T_fps_a

    da_fps = tr.Property(tr.Float, depends_on='input_changed')
    '''Crack length increment contoling the calculation
    '''
    @tr.cached_property
    def _get_da_fps(self):
        return self.eta * self.L_fps

    x1_fps_a = tr.Property(tr.Array, depends_on='state_changed')
    '''Shifted position of the fracture process hot spot. This 
    is the point that is required to achieve the tensile strength
    '''
    @tr.cached_property
    def _get_x1_fps_a(self):
        x1_fps_a = self.x_fps_a + self.da_fps * self.T_fps_a
        x1_fps_a[1] = np.min([x1_fps_a[1], self.ts.H])
        return x1_fps_a

    x_fps_a = tr.Property(tr.Array, depends_on='state_changed')
    '''Position of the starting point of the fracture process segment.
    '''
    @tr.cached_property
    def _get_x_fps_a(self):
        return self.x_t_Ia[-1, :]

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

    T_Lab = tr.Property(depends_on='state_changed')
    '''Unit line vector
    '''
    @tr.cached_property
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

    x_n_ejaM = tr.Property(depends_on='state_changed')
    '''Unit line vector
    '''
    @tr.cached_property
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

    x_rot_a = tr.Property(depends_on='state_changed')
    '''Center of rotation.
    '''
    @tr.cached_property
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

    w_f_t = tr.DelegatesTo('sim')

    #=========================================================================
    # Unknown variables
    #=========================================================================
    phi = tr.Property(tr.Float, depends_on='state_changed')
    '''Rotation of the right-hand part of the shear zone.
    '''
    @tr.cached_property
    def _get_phi(self):
        x_fps_a = self.x_fps_a
        w_f_t = self.ts.w_f_t
        x_rot_a = self.x_rot_a
        n_fps = len(self.x_t_Ia) - 1
        n_m_fps = n_fps * (self.n_m - 1) + 3
        T_ab1 = self.T_Lab[n_m_fps, ...]
        theta = self.theta
        T_ab = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
        phi = get_phi(T_ab, x_fps_a, x_rot_a, w_f_t)
        return phi

    def get_moved_Ka(self, x_Ka):
        return self.rotate(x_Ka=x_Ka, phi=self.phi)

    x1_Ia = tr.Property(depends_on='state_changed')
    '''Displaced segment nodes'''
    @tr.cached_property
    def _get_x1_Ia(self):
        return self.get_moved_Ka(self.x_Ia)

    x1_Ka = tr.Property(depends_on='state_changed')
    '''Displaced integration points'''
    @tr.cached_property
    def _get_x1_Ka(self):
        return self.get_moved_Ka(self.x_Ka)

    x1_Ca = tr.Property(depends_on='state_changed')
    '''Diplaced corner nodes'''
    @tr.cached_property
    def _get_x1_Ca(self):
        return self.get_moved_Ka(self.x_Ca)

    #=========================================================================
    # Control and state variables
    #=========================================================================
    v = tr.Property(depends_on='state_changed')
    '''Vertical displacement on the right hand side
    '''
    @tr.cached_property
    def _get_v(self):
        x_rot_0 = self.x_rot_a[0]
        phi = self.phi
        L_rot_distance = self.ts.L - x_rot_0
        v = L_rot_distance * np.sin(phi)
        return v

    #=========================================================================
    # Kinematics
    #=========================================================================

    u_Lib = tr.Property(depends_on='state_changed')
    '''Displacement of lines in local coordinates'''
    @tr.cached_property
    def _get_u_Lib(self):
        K_Li = self.K_Li
        u_Ka = self.x1_Ka - self.x_Ka
        u_Lia = u_Ka[K_Li]
        T_Lab = self.T_Lab
        u_Lib = np.einsum(
            'Lia,Lab->Lib', u_Lia, T_Lab
        )
        return u_Lib

    u_Lb = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
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

    get_u0 = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning horizontal displacement 
    component for a specified vertical coordinate of a ligmant.
    '''
    @tr.cached_property
    def _get_get_u0(self):
        return interp1d(self.x_Lb[:, 1], self.u_Lb[:, 0],
                        fill_value='extrapolate')

    u_ejaM = tr.Property(depends_on='state_changed')
    '''Transformed displacement at the line midpoints '''
    @tr.cached_property
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

    cmod = tr.Property(depends_on='state_changed')
    '''Crack mouth opening displacement .
    '''
    @tr.cached_property
    def _get_cmod(self):
        return self.u_Lb[0, 0]

    xn_m_fps = tr.Property(depends_on='state_changed')
    '''Get the index of the material point at which the orientation is 
    evaluated
    '''
    @tr.cached_property
    def _get_xn_m_fps(self):
        n_tip = len(self.x_t_Ia) - 1
        n_m_tip = n_tip * (self.n_m - 1) + int(self.n_m * 2)
        return n_m_tip

    x_sig_fps_1 = tr.Property(depends_on='state_changed')
    '''Get the vertical position of the orientation segment 
    '''
    @tr.cached_property
    def _get_x_sig_fps_1(self):
        return self.x_tip_a[1]

    #=========================================================================
    # Plot current state
    #=========================================================================
    def plot_x_Ka(self, ax):
        x_aK = self.x_Ka.T
        ax.plot(*x_aK, 'bo', color='blue', markersize=8)

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

        ax.plot([0, self.ts.L], [y, y],
                color='black', linestyle='--')
        ax.plot(x, y, 'bo', color='blue', markersize=10)

    def plot_x_fps_a(self, ax):
        x, y = self.x_fps_a
        ax.plot(x, y, 'bo', color='green', markersize=10)

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
        for z in self.ts.z_f:
            # left part
            ax.plot([0, self.ts.L], [z, z], color='maroon', lw=5)

    def plot_sz_state(self, ax, vot=1.0):
        self.plot_sz1(ax)
        self.plot_sz0(ax)
        self.plot_sz_fill(ax)
        self.plot_x_tip_a(ax)
        self.plot_x_rot_a(ax)
        self.plot_x_fps_a(ax)
        self.plot_reinf(ax)
        ax.axis('off')
        ax.axis('equal')

    def plot_T_Lab(self, ax):
        K_Li = self.K_Li
        x_n_ejaM = self.x_n_ejaM
        x_n_eajM = np.einsum('ejaM->eajM', x_n_ejaM)
        ax.plot(*x_n_eajM.reshape(-1, 2, len(K_Li)), color='orange', lw=3)

    def plot_u_T_Lab(self, ax):
        u_ejaM = self.u_ejaM
        u_eajM = np.einsum('ejaM->eajM', u_ejaM)
        ax.plot(*u_eajM.reshape(-1, 2, len(self.K_Li)), color='orange', lw=3)

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

    def plot_u_Lc(self, ax, u_Lc, idx=0, color='black', label=r'$w$ [mm]'):
        x_La = self.x_Lb
        u_Lb_min = np.min(u_Lc[:, idx])
        u_Lb_max = np.max(u_Lc[:, idx])
        self.plot_hlines(ax, u_Lb_min, u_Lb_max)
        ax.plot(u_Lc[:, idx], x_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_La[:, 1], u_Lc[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    def plot_u_L0(self, ax, vot=1):
        self.plot_u_Lc(ax, self.u_Lb, 0,
                       label=r'$w$ [mm]')
        ax.set_xlabel(r'effective horizontal COD $w$ [mm]')


class ShearZoneHist(Hist):
    '''History class
    '''
    record_traits = tr.List(
        ['M', 'Q',
         'v',
         # 'cmod', 'phi',
         'z_fps',
         'tau_fps',
         'theta',
         'theta_bar',
         'sig_fps_0']
    )

    x_t_Ia = tr.List
    record_t = tr.Dict

    def init_state(self):
        super(ShearZoneHist, self).init_state()
        self.x_t_Ia.append(np.copy(self.tstep_source.xd.x_t_Ia))
        for rt in self.record_traits:
            self.record_t[rt] = [0]

    def record_timestep(self, t, U, F,
                        state_vars=None):
        super(ShearZoneHist, self).record_timestep(t, U, F, state_vars)
        self.x_t_Ia.append(np.copy(self.tstep_source.xd.x_t_Ia))
        for rt in self.record_traits:
            self.record_t[rt].append(getattr(self.tstep_source, rt))

    #=========================================================================
    # History browsering
    #=========================================================================
    #
    # Time slider might be an
    vot = tr.Float

    def _vot_changed(self):
        idx = self.get_time_idx(self.vot)
        # update the primary variables of the model
        # the internal variables and the geometrical
        # representation of the model.
        self.ts.xd.x_t_Ia = self.x_t_Ia[idx]
        self.ts.X = self.U_t[idx, :]

    ts = tr.Property

    @tr.cached_property
    def _get_ts(self):
        return copy.deepcopy(self.tstep_source)

    response = tr.Property

    def _get_response(self):
        rarr = {key: np.array(vals, dtype=np.float_)
                for key, vals in self.record_t.items()}
        return rarr

    #=========================================================================
    # Energetic characteristics
    #=========================================================================

    G = tr.Property  # (depends_on='input_changed, state_')
    '''Dissipated energy
    '''
#    @tr.cached_property

    def _get_G(self):
        Q = self.response['Q']
        v = self.response['v']
        W = cumtrapz(Q, v, initial=0)
        U = Q * v / 2.0
        return W - U

    d_G = tr.Property  # (depends_on='input_changed')
    '''Energy release rate
    '''
#    @tr.cached_property

    def _get_d_G(self):
        G = self.G
        G0 = G[:-1]
        G1 = G[1:]
        dG = np.hstack([0, G1 - G0])
        return dG / self.ts.xd.da_fps / self.ts.B

    def plot_dGv(self, ax, vot=1):
        R = self.response
        ax.plot(R['v'], self.d_G, color='magenta')
        ax.fill_between(R['v'], self.d_G, color='magenta', alpha=0.05)

    def plot_Qv(self, ax, vot=1):
        ax.set_title('Structural response')
        R = self.response
        ax.plot(R['v'], R['Q'] * 1e-3, color='black')
        ax.set_ylabel(r'Shear force $Q$ [kN]')
        ax.set_xlabel(r'Displacement $v$ [mm]')

    def plot_Mw(self, ax, vot=1):
        ax.set_title('Structural response')
        R = self.response
        ax.plot(R['cmod'], R['M'], color='black')
        ax.set_ylabel(r'Moment $M$ [Nmm]')
        ax.set_xlabel(r'CMOD $w$ [mm]')

    def plot_fps_stress(self, ax, vot=1):
        '''Visualize the stress components affecting the
        crack orientation in the next step.
        '''
        R = self.response
        z_fps = R['z_fps']
        tau_fps = R['theta_bar']
        sig_fps_0 = R['sig_fps_0']
        ax.plot(tau_fps, z_fps, color='blue',
                label=r'$\tau^{\mathrm{fps}}_{xz}$')
        ax.fill_betweenx(z_fps, tau_fps, 0, color='blue', alpha=0.2)
        ax.plot([0, self.ts.sig_fps_0],
                [self.ts.xd.x_sig_fps_1, self.ts.xd.x_sig_fps_1],
                lw=3, color='blue')
        ax.plot(sig_fps_0, z_fps, color='red',
                label=r'$\sigma^{\mathrm{fps}}_{xx}$')
        ax.fill_betweenx(z_fps, sig_fps_0, 0, color='red', alpha=0.2)
        ax.set_xlabel(r'$\sigma_x$ and $\tau_{xz}$ [MPa]')
        ax.legend(loc='center left')

    def plot_fps_orientation(self, ax1, vot=1):
        '''Show the lack-of-fit between the direction of principal stresses
        at the process zone and the converged crack inclination angle.
        '''
        R = self.response
        z_fps = R['z_fps']
        theta = R['theta_bar']
        theta_bar = R['theta_bar']
        ax1.set_title(r'Fracture propagation segment')
        ax1.plot(theta, z_fps, color='orange',
                 label=r'$\theta$')
        ax1.plot(theta_bar, z_fps, color='black',
                 linestyle='dashed', label=r'$\bar{\theta}$')
        ax1.set_xlabel(r'Orientation $\theta$ [$\pi^{-1}$]')
        ax1.set_ylabel(r'Vertical position $z$ [mm]')
        ax1.legend(loc='lower left')


class ShearZoneModel(TStep, BMCSTreeNode, Vis2D):
    '''Time step representing the shear zone at a particular 
    time t_n1. It provides the mapping between the vector of 
    primary unknowns U and the residuum.
    '''
    hist_type = ShearZoneHist

    def _tree_node_list_default(self):
        return [
            self.mm,
            self.xd,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.mm,
            self.xd,
        ]

    xd = tr.Instance(XDomain)

    def _xd_default(self):
        return XDomain(ts=self)

    tree_view = ui.View(
        ui.Group(
            ui.Item('B', resizable=True, full_size=True),
            ui.Item('L', resizable=True, full_size=True),
            ui.Item('H', resizable=True, full_size=True),
        )
    )

    B = tr.Float(20, GEO=True, auto_set=False, enter_set=True)
    H = tr.Float(60, GEO=True, auto_set=False, enter_set=True)
    L = tr.Float(100, GEO=True, auto_set=False, enter_set=True)
    initial_crack_position = tr.Float(
        400, GEO=True, auto_set=False, enter_set=True)
    eta = tr.Float(0.2, MAT=True)
    '''Fraction of characteristic length to take as a propagation segment
    '''

    input_changed = tr.Event
    '''Register the input change event to trigger recalculation.
    '''

    #=========================================================================
    # Material characteristics
    #=========================================================================
    mm = tr.Instance(IMaterialModel)
    '''Material model containing the bond-slip law
    concrete compression and crack softening law
    '''

    w_f_t = tr.Property(tr.Float)
    '''Critical crack opening at the level of tensile concrete strength.

    @todo: Check consistency, the crack opening is obtained as a
    product of the critical strain at strength and the characteristic
    size of the localization zone.
    '''

    def _get_w_f_t(self):
        return self.mm.f_t / self.mm.E_c * self.mm.L_c

    z_f = tr.Array(np.float_, value=[30])
    A_f = tr.Array(np.float_, value=[2 * np.pi * 8**2])

    theta = tr.Property

    def _get_theta(self):
        return self.xd.theta

    v = tr.Property

    def _get_v(self):
        return self.xd.v

    t_n = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Fundamental state time.
    '''
    t_n1 = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Target value of the control variable.
    '''
    U_n = tr.Array(np.float_,
                   value=[0.0, 0.0], auto_set=False, enter_set=True)
    '''Current fundamental value of the primary variable.
    '''
    U_k = tr.Array(np.float_,
                   value=[0.0, 0.0], auto_set=False, enter_set=True)
    '''Primary unknown variables subject to the iteration process.
    - center of rotation
    - inclination angle of a new crack segment
    '''

    def init_state(self):
        '''Initialize state.
        '''
        self.t_n1 = 0.0
        self.U_n[:] = 0.0
        self.U_k = [self.H / 2.0,
                    0.0, ]
        self.xd.x_rot_1 = self.U_k[0]

    xtol = tr.Float(1e-3, auto_set=False, enter_set=True)
    maxfev = tr.Int(1000, auto_set=False, enter_set=True)

    def make_iter(self):
        '''Perform one iteration
        '''
        X0 = np.copy(self.X[:])

        def get_R_X(X):
            self.X = X
            R = self.get_R()
            return R
        res = root(get_R_X, X0, method='lm',
                   options={'xtol': self.xtol,
                            })
        self.X[:] = res.x
        self.U_n[:] = self.U_k[:]
        R_k = self.get_R()
        nR_k = np.linalg.norm(R_k)
        print('R_k', nR_k, 'Success', res.success)
        if res.success == False:
            raise StopIteration('no solution found')
        self.xd.state_changed = True
        self.state_changed = True
        return res.x

    def make_incr(self):
        '''Update the control, primary and state variabrles..
        '''
        R_k = self.get_R()
        self.hist.record_timestep(self.t_n1, self.U_k, R_k)
        self.t_n = self.t_n1
        self.xd.x_t_Ia = np.vstack(
            [self.xd.x_t_Ia, self.xd.x1_fps_a[np.newaxis, :]])
        self.xd.state_changed = True
        self.state_changed = True

    X = tr.Property()

    def _get_X(self):
        return self.U_k

    def _set_X(self, value):
        self.U_k[:] = value
        self.xd.x_rot_1 = value[0]
        self.xd.theta = value[1]
        self.xd.state_changed = True
        self.state_changed = True

    def get_R(self):
        '''Residuum checking the lack-of-fit
        - of the normal force equilibrium in the cross section
        - of the orientation of the principal stress and of the fracture
          process segment (FPS)
        '''
        N, _ = self.F_a
        R_M = (self.xd.theta - self.theta_bar)
        R = np.array([N,  R_M], dtype=np.float_)
        return R

    state_changed = tr.Event

    Q = tr.Property(depends_on='state_changed')
    '''Shear force'''
    @tr.cached_property
    def _get_Q(self):
        L_x = self.L
        Q_single = self.M / (L_x - self.xd.x_rot_a[0])
        x_ = self.xd.x_rot_a[0]
        Q_line = 2 * self.M * x_ / (L_x**2 - x_**2)
        #print(Q_single, Q_line)
        return Q_single

    z_fps = tr.Property(depends_on='state_changed')
    '''Vertical coordinate of the crack tip
    '''
    @tr.cached_property
    def _get_z_fps(self):
        #z_fps = self.x_fps_a[1]
        return self.xd.x_sig_fps_1

    tau_fps = tr.Property(depends_on='state_changed')
    '''Shear stress in global xz coordinates in the fracture
    process segment. Quadratic profile of shear stress assumed.
    '''
    @tr.cached_property
    def _get_tau_fps(self):
        z_fps = self.z_fps
        Q = self.Q
        H = self.H
        B = self.B
        tau_fps = (6 * Q * z_fps / (H**2 + H * z_fps - 2 * z_fps**2))
        return tau_fps / B

    sig_fps_0 = tr.Property(depends_on='state_changed')
    '''Normal stress component in global $x$ direction in the fracture .
    process segment.
    '''
    @tr.cached_property
    def _get_sig_fps_0(self):
        z_fps = self.z_fps
        return self.get_sig_fps_0(z_fps)
#         S_a = self.S_La[self.xd.n_m_fps, ...]
#         return S_a[0] / self.sim.B

    get_sig_fps_0 = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning horizontal stress 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_sig_fps_0(self):
        return interp1d(self.xd.x_Lb[:, 1], self.S_La[:, 0] / self.B,
                        fill_value='extrapolate')

    #=========================================================================
    # Stress transformation and integration
    #=========================================================================

    S_Lb = tr.Property(depends_on='state_changed')
    '''Stress returned by the material model
    '''
    @tr.cached_property
    def _get_S_Lb(self):
        u_a = self.xd.u_Lb
        Sig_w = self.mm.get_sig_w(u_a[..., 0]) * self.B
        Tau_w = self.mm.get_tau_s(u_a[..., 1]) * self.B
        return np.einsum('b...->...b', np.array([Sig_w, -Tau_w], dtype=np.float_))

    S_La = tr.Property(depends_on='state_changed')
    '''Transposed stresses'''
    @tr.cached_property
    def _get_S_La(self):
        S_Lb = self.S_Lb
        S_La = np.einsum('Lb,Lab->La', S_Lb, self.xd.T_Lab)
        return S_La

    F_La = tr.Property(depends_on='state_changed')
    '''Integrated segment forces'''
    @tr.cached_property
    def _get_F_La(self):
        S_La = self.S_La
        F_La = np.einsum('La,L->La', S_La, self.xd.norm_n_vec_L)
        return F_La

    F_f = tr.Property(depends_on='state_changed')
    '''Get the discrete force in the reinforcement f
    '''
    @tr.cached_property
    def _get_F_f(self):
        u_f = self.xd.get_u0(self.z_f)
        F_new = self.A_f * self.mm.get_sig_w_f(u_f)
        return F_new

    M = tr.Property(depends_on='state_changed')
    '''Internal bending moment obtained by integrating the
    normal stresses with the lever arm rooted at the height of the neutral
    axis.
    '''
    @tr.cached_property
    def _get_M(self):
        x_Lia = self.xd.x_Ka[self.xd.K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = self.F_La
        M_L = (x_La[:, 1] - self.xd.x_rot_a[1]) * F_La[:, 0]
        M = np.sum(M_L, axis=0)
        for y in self.z_f:
            M += (y - self.xd.x_rot_a[1]) * self.F_f
        return -M

    F_a = tr.Property(depends_on='state_changed')
    '''Integrated normal and shear force
    '''
    @tr.cached_property
    def _get_F_a(self):
        F_La = self.F_La
        F_a = np.sum(F_La, axis=0)
        F_a[0] += np.sum(self.F_f)
        return F_a

    theta_bar = tr.Property(depends_on='state_changed')
    '''Angle between the vertical direction and the orientation of
    the principal stresses setting the orientation of the fracture
    process segment in the next iteration step.
    '''
    @tr.cached_property
    def _get_theta_bar(self):
        tau_fps = self.tau_fps
        sig_x = self.sig_fps_0

        theta_0 = get_theta_0(tau_fps, sig_x, self.mm.f_t)
        return theta_0
        theta_f = get_theta_f(tau_fps, sig_x, self.mm.f_t)
        if sig_x > 0:
            return theta_0
        else:
            return theta_f

        tan_theta = 2 * tau_fps / (
            sig_x + np.sqrt(4 * tau_fps**2 + sig_x**2))
        return np.arctan(tan_theta)

    #=========================================================================
    # Plotting methods
    #=========================================================================

    def plot_sig_fps_0(self, ax, vot=1.0):
        ax.plot([0, self.sig_fps_0 * self.B], [self.xd.x_sig_fps_1,
                                               self.xd.x_sig_fps_1],
                lw=3, color='blue')

    def plot_S_Lc(self, ax, S_Lc, idx=0,
                  title='Normal stress profile',
                  label=r'$\sigma_{xx}$',
                  color='red',
                  ylabel=r'Vertical position [mm]',
                  xlabel=r'Stress [MPa]'):
        x_La = self.xd.x_Lb
        S_La_min = np.min(S_Lc[:, idx])
        S_La_max = np.max(S_Lc[:, idx])
        self.xd.plot_hlines(ax, S_La_min, S_La_max)
        ax.set_title(title)
        ax.plot(S_Lc[:, idx], x_La[:, 1],
                color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.fill_betweenx(x_La[:, 1], S_Lc[:, idx], 0,
                         color=color, alpha=0.2)
        ax.legend()

    def plot_S_L0(self, ax, vot=1):
        self.plot_S_Lc(ax, self.S_La, idx=0,
                       title='Normal stress profile',
                       label=r'$\sigma_{xx}$',
                       color='red',
                       ylabel=r'Vertical position [mm]',
                       xlabel=r'Stress [MPa]')
        self.plot_sig_fps_0(ax, vot)

    def plot_fps_sig_w(self, ax, vot=1):
        '''Plot the stress profile in the fracture process zone
        '''
        get_ux_z = interp1d(self.xd.x_Lb[:, 1], self.xd.u_Lb[:, 0],
                            fill_value='extrapolate')
        z_ = np.linspace(self.xd.x_fps_a[1], self.xd.x_tip_a[1], 30)
        w = get_ux_z(z_)
        sig = self.mm.get_sig_w(w)
        ax.plot(z_, sig)

    tloop_type = TLoopSZ

    a_t = tr.Property

    def _get_a_t(self):
        n_a_t = len(self.xd.x_t_Ia)
        return np.linspace(0, n_a_t, n_a_t) * self.xd.L_fps

    def get_window(self):
        hist = self.hist
        fw_geo = hist.ts.xd.plt('plot_sz_state')
        u_x = hist.ts.xd.plt('plot_u_L0')
        fps_sig_w = hist.ts.plt('plot_fps_sig_w')
        Qv = hist.plt('plot_Qv')
        fps_angle = hist.plt('plot_fps_orientation')
        fps_state = hist.plt('plot_fps_stress')
        dGv = hist.plt('plot_dGv')
        S_x = hist.ts.plt('plot_S_L0')
        Mw = hist.ts.plt('plot_Mw')

        pp1 = PlotPerspective(
            name='main view',
            viz2d_list=[fw_geo,
                        fps_angle,
                        u_x,
                        Qv
                        ],
            positions=[211,
                       234, 235, 236
                       ],
            twiny=[(fps_angle, fps_state, False),
                   (u_x, S_x, True)
                   ],
            twinx=[(Qv, dGv, False)]
        )

        mm_sig_w = self.mm.plt('plot_sig_w')
        mm_tau_s = self.mm.plt('plot_tau_s')
        mm_sig_w_f = self.mm.plt('plot_sig_w_f')
        pp2 = PlotPerspective(
            name='material view',
            viz2d_list=[mm_sig_w_f, fps_sig_w, mm_sig_w, mm_tau_s],
            positions=[221, 222, 223, 224],
            twiny=[],
            twinx=[]
        )

        win = BMCSWindow(model=self)
        win.viz_sheet.monitor_chunk_size = 10
        win.viz_sheet.pp_list = [pp1, pp2]
        win.viz_sheet.selected_pp = pp1
        win.viz_sheet.tight_layout = True
        win.viz_sheet.hist = hist
        return win


if __name__ == '__main__':
    material_model = MaterialModel(G_f=0.015, L_fps=50, f_c=-30.3)

    '''Development history:

    1) Test the correctness of the calculation for
    given values
    2) Integration over the height
    '''
    L_q = 3850
    shear_zone = ShearZoneModel(
        B=250, H=600, L=L_q,
        eta=0.07,
        initial_crack_position=L_q * 0.2,  # (1 - 0.25 * 0.8),
        z_f=[50],  # [20],
        A_f=[2 * np.pi * 14**2],
        mm=material_model
    )

    shear_zone.sim.tloop.n_seg = 230
    shear_zone.xtol = 1e-3
    shear_zone.maxfev = 1000
    if False:
        shear_zone.n_seg = 8
        shear_zone.run()
        print('timeline')
        print(shear_zone.hist.timesteps)
        print(shear_zone.hist.t)
        shear_zone.hist.vot = 1
        print(shear_zone.hist.ts.xd.x_Ia)
        shear_zone.hist.vot = 7
        print(shear_zone.hist.ts.xd.x_Ia)
    if True:
        win = shear_zone.get_window()
        # shear_zone.run()
        win.configure_traits()
    if False:
        xd = shear_zone.xd
        print('x_fps')
        print(xd.x_fps_a)
        print('x_tip')
        print(xd.x_tip_a)
        print('x_Ia')
        print(xd.x_Ia)
        print('L_fps')
        print(xd.L_fps)
        print('phi')
        print(xd.phi)
        print('x_rot_a')
        print(xd.x_rot_a)
        print('w_f_t')
        print(xd.w_f_t)
    if False:
        print('L_c')
        print(shear_zone.mm.L_c)
        print('U')
        print(shear_zone.tstep.X)
        print('F_a')
        print(shear_zone.F_a)
        print('Q')
        print(shear_zone.Q)
    if False:
        xd = shear_zone.xd
        fig, ax = plt.subplots(
            1, 1, figsize=(20, 18), tight_layout=True
        )
        ax.axis('equal')
        xd.plot_sz_state(ax)
        shear_zone.plot_reinf(ax)
        plt.show()
    if False:
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(
            2, 2, figsize=(12, 8), tight_layout=True)
        material_model.plot_sig_w(ax11)
        material_model.plot_tau_s(ax12)
        material_model.plot_sig_w_f(ax21)
        plt.show()
    if False:
        shear_zone.run()
        R = shear_zone.response
        fig, ((ax11, ax12, ax13, ax14),
              (ax21, ax22, ax23, ax24)) = plt.subplots(
            2, 4, figsize=(20, 18), tight_layout=True
        )
        ax11.axis('equal')
        xd = shear_zone.xd
        ts = shear_zone.tstep
        x_La = np.sum(xd.x_Ka[xd.K_Li], axis=1) / 2

        xd.plot_sz_state(ax11)

        ax21b = ax21.twiny()
        ts.plot_fps_stress(ax21)
        ts.plot_fps_orientation(ax21b)

        ax12b = ax12.twinx()
        ts.plot_Qv(ax12)
        ax12b.plot(R['v'], R['M'], color='black', linestyle='dashed')
        ax12b.set_ylabel(r'Bending moment $M(x_\mathrm{rot})$ [kNm]')

        ax22.plot(shear_zone.a_t, R['G'], label=r'$G$', color='black',
                  linestyle='dashed')
        ax22b = ax22.twinx()
        ax22b.plot(shear_zone.a_t, R['d_G'], color='magenta')
#         mm.plot_sig_w(ax22, ax22.twinx())
#         mm.plot_tau_s(ax22, ax22.twinx())

        ax13a = ax13.twiny()
        xd.plot_u_Lc(ax13a, xd.u_Lb, 1)
        ts.plot_S_Lc(ax13, ts.S_Lb, 1)
        #shear_zone._align_xaxis(ax13, ax13a)

        ax23a = ax23.twiny()
        xd.plot_u_Lc(ax23a, xd.u_Lb, 0)
        ts.plot_S_Lc(ax23, ts.S_Lb, 0)
        #shear_zone._align_xaxis(ax23, ax23a)

        ax14a = ax14.twiny()
        xd.plot_u_Lc(ax14a, xd.u_Lb, 1)
        ts.plot_S_Lc(ax14, ts.S_La, 1)
        #shear_zone._align_xaxis(ax14, ax14a)

        ax24a = ax24.twiny()
        xd.plot_u_Lc(ax24a, xd.u_Lb, 0)
        ts.plot_S_Lc(ax24, ts.S_La, 0)
        #shear_zone._align_xaxis(ax24, ax24a)

        plt.show()

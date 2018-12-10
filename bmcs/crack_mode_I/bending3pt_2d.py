'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from scipy import interpolate as ip
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item, UItem, VGroup

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    IMATSEval, TLine, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.vtloop import TimeLoop
from ibvpy.dots.vdots_grid import DOTSGrid
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage, MATS2DMplCSDEEQ  # , MATS2DMplCSDODF
from ibvpy.mats.mats3D.mats3D_sdamage.viz3d_sdamage import Vis3DSDamage,\
    Viz3DSDamage
from ibvpy.mats.mats3D.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from ibvpy.mats.mats3D.viz3d_stress_field import \
    Vis3DStressField, Viz3DStressField
import numpy as np
import traits.api as tr
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow

from .viz3d_energy import Viz2DEnergy, Vis2DEnergy, Viz2DEnergyReleasePlot


class Viz2DForceDeflection(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        P, W = self.vis2d.get_PW()
        ymin, ymax = np.min(P), np.max(P)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(W), np.max(W)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        color = kw.get('color', 'black')
        linewidth = kw.get('linewidth', 2)
        label = kw.get('label', 'P(w)')
        ax.plot(W, P, linewidth=linewidth, color=color, alpha=0.4,
                label=label)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('Force P [N]')
        ax.set_xlabel('Deflection w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        P, W = self.vis2d.get_PW()
        idx = self.vis2d.tloop.get_time_idx(vot)
        P, w = P[idx], W[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Vis2DCrackBand(Vis2D):

    model = tr.WeakRef
    tloop = tr.Property

    def _get_tloop(self):
        return self.model.tloop

    X_E = tr.Array(np.float_)
    eps_t = tr.List
    sig_t = tr.List
    a_t = tr.List

    def setup(self, tl):
        self.X_E = []
        self.eps_t = []
        self.sig_t = []
        self.a_t = []

    def update(self, U, t):
        bt = self.model
        tl = self.tloop
        ts = tl.ts
        fe_grid = ts.mesh
        mats = ts.mats
        fets = ts.fets
        n_c = fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[ts.I_Ei]
        eps_Enab = np.einsum(
            'Einabc,Eic->Enab', ts.B_Einabc, U_Eia
        )
        deps_Emab = np.zeros_like(eps_Enab)
        D_Enabef, sig_Enab = mats.get_corr_pred(
            eps_Enab, deps_Emab, t, t, False, False,
            ** ts.state_arrays
        )
        crack_band = fe_grid[0, bt.n_a + 1:]
        E = crack_band.elems
        X = crack_band.dof_X
        eps_Ey = eps_Enab[E, :, 0, 0]
        eps_E1 = np.average(eps_Ey, axis=1)
        sig_Ey = sig_Enab[E, :, 0, 0]
        sig_E1 = np.average(sig_Ey, axis=1)
        X_E = np.average(X[:, :, 1], axis=1)
        eps_0 = mats.omega_fn.eps_0
        eps_thr = eps_E1 - eps_0
        a_idx = np.argwhere(eps_thr < 0)
        if len(a_idx) > 0:
            x_1 = X_E[0]
            x_a = X_E[a_idx[0][0]]
            self.a_t.append(x_a - x_1)
        else:
            self.a_t.append(0)

        self.X_E = X_E
        self.eps_t.append(eps_E1)
        self.sig_t.append(sig_E1)

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    def get_a_x(self):
        return self.X_E

    def get_eps_t(self, vot):
        t_idx = self.tloop.get_time_idx(vot)
        return self.eps_t[t_idx]

    def get_sig_t(self, vot):
        t_idx = self.tloop.get_time_idx(vot)
        return self.sig_t[t_idx]

    def get_a_t(self):
        return np.array(self.a_t, dtype=np.float_)

    def get_da_dt(self):
        a = self.get_a_t()
        t = self.get_t()
        tck = ip.splrep(t, a, s=0, k=1)
        return ip.splev(t, tck, der=1)


def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


class Viz2DStrainInCrack(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'strain in crack'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    ax_sig = tr.Any

    def plot(self, ax, vot, *args, **kw):
        eps = self.vis2d.get_eps_t(vot)
        sig = self.vis2d.get_sig_t(vot)
        a_x = self.vis2d.get_a_x()
        ax.plot(a_x, eps, linewidth=3, color='red', alpha=0.4,
                label='P(w;x=L)')
        ax.fill_between(a_x, 0, eps, facecolor='orange', alpha=0.2)
        ax.set_ylabel('Strain [-]')
        ax.set_xlabel('Position [mm]')
        if self.ax_sig:
            self.ax_sig.clear()
        else:
            self.ax_sig = ax.twinx()
        self.ax_sig.plot(a_x, sig, linewidth=2, color='blue')
        self.ax_sig.fill_between(a_x, 0, sig, facecolor='blue', alpha=0.2)
        align_yaxis_np(ax, self.ax_sig)
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DStressInCrack(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'stress in crack'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sig = self.vis2d.get_sig_t(vot)
        a_x = self.vis2d.get_a_x()
        ax.plot(a_x, sig, linewidth=3, color='red', alpha=0.4,
                label='P(w;x=L)')
        ax.fill_between(a_x, 0, sig, facecolor='orange', alpha=0.2)
        ax.set_ylabel('Stress [-]')
        ax.set_xlabel('Position [mm]')
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DTA(Viz2D):

    '''Plot adaptor for bending test simulator.
    '''
    label = 'crack length'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()
        a = self.vis2d.get_a_t()
        ax.plot(t, a, linewidth=3, color='blue', alpha=0.4,
                label='Crack length')
        ax.fill_between(t, 0, a, facecolor='blue', alpha=0.2)
        ax.set_xlabel('time')
        ax.set_ylabel('a')
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DdGdA(Viz2D):
    label = 'Energy release per unit crack length'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    vis2d_cb = tr.WeakRef

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d_cb.get_t()
        G_t = self.vis2d.get_G_t()
        a_t = self.vis2d_cb.get_a_t()
        b = self.vis2d_cb.model.cross_section.b

        tck = ip.splrep(a_t * b, G_t, s=0, k=1)
        dG_da = ip.splev(a_t, tck, der=1)

#         ax.plot(a_t, dG_da, linewidth=3, color='blue', alpha=0.4,
#                 label='dG/da')
#         ax.fill_between(a_t, 0, dG_da, facecolor='blue', alpha=0.2)
#         ax.set_xlabel('time')
#         ax.set_ylabel('dG_da')

        tck = ip.splrep(t, G_t, s=0, k=1)
        dG_dt = ip.splev(t[:-1], tck, der=1)
        tck = ip.splrep(t, a_t, s=0, k=1)
        da_dt = ip.splev(t[:-1], tck, der=1)
        nz = da_dt != 0.0
        dG_da = np.zeros_like(da_dt)
        dG_da[nz] = dG_dt[nz] / da_dt[nz] / b
        ax.plot(a_t[1:], dG_da, linewidth=3, color='green', alpha=0.4,
                label='dG/dt / da_dt')

#         ax.plot(a_t, G_t, linewidth=3, color='black')
#         ax.fill_between(a_t, 0, G_t, facecolor='blue', alpha=0.2)

        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class CrossSection(BMCSLeafNode):

    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    b = Float(100.0,
              CS=True,
              label='thickness',
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')

    traits_view = View(
        VGroup(
            Item('b', resizable=True),
            label='Cross section'
        )
    )

    tree_view = traits_view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    H = Float(100.0,
              label='beam depth',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='cross section height [mm2]')
    L = Float(600.0,
              label='beam length',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')
    a = Float(20.0,
              GEO=True,
              label='notch depth',
              auto_set=False, enter_set=True,
              desc='Depth of the notch')
    L_c = Float(4.0,
                GEO=True,
                label='crack band width',
                auto_set=False, enter_set=True,
                desc='Width of the crack band')

    traits_view = View(
        VGroup(
            Item('H', resizable=True),
            Item('L'),
            Item('a'),
            Item('L_c'),
            label='Geometry'
        )
    )

    tree_view = traits_view


class BendingTestModel(BMCSModel, Vis2D):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'bending test simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry
        ]

    #=========================================================================
    # Interactive control of the time loop
    #=========================================================================
    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.stop = True
        self.tloop.paused = True

    def stop(self):
        self.tloop.stop = True
        self.tloop.restart = True

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    cross_section = Instance(CrossSection)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(20,
                label='# of elems in x-dir',
                MESH=True, auto_set=False, enter_set=True)
    n_e_y = Int(8,
                label='# of elems in y-dir',
                MESH=True, auto_set=False, enter_set=True)

    w_max = Float(-50, BC=True, auto_set=False, enter_set=True)

    controlled_elem = Property

    def _get_controlled_elem(self):
        return 0

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('microplane damage (eeg)',
                           {'elastic': MATS2DElastic,
                            'microplane damage (eeq)': MATS2DMplDamageEEQ,
                            'microplane CSD (eeq)': MATS2DMplCSDEEQ,
                            #'microplane CSD (odf)': MATS2DMplCSDODF,
                            'scalar damage': MATS2DScalarDamage
                            },
                           MAT=True
                           )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

#     @on_trait_change('BC,MAT,MESH')
#     def reset_node_list(self):
#         self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         report=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS2D4Q),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS2D4Q()

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='GEO,CS,BC,MAT,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_right_bc,
                   self.fixed_x,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

    fixed_right_bc = Property(depends_on='CS,BC,GEO,MAT,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_right_bc(self):
        return BCSlice(slice=self.fe_grid[-1, 0, -1, 0],
                       var='u', dims=[1], value=0)

    n_a = Property
    '''Element at the notch
    '''

    def _get_n_a(self):
        a_L = self.geometry.a / self.geometry.H
        return int(a_L * self.n_e_y)

    fixed_x = Property(depends_on='CS,BC,GEO,MAT,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_x(self):
        return BCSlice(slice=self.fe_grid[0, self.n_a:, 0, -1],
                       var='u', dims=[0], value=0)

    control_bc = Property(depends_on='CS,BC,GEO,MAT,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(slice=self.fe_grid[0, -1, :, -1],
                       var='u', dims=[1], value=-self.w_max)

    dots_grid = Property(Instance(DOTSGrid),
                         depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        dgrid = DOTSGrid(L_x=self.geometry.L / 2., L_y=self.geometry.H,
                         integ_factor=self.cross_section.b,
                         n_x=self.n_e_x, n_y=self.n_e_y,
                         fets=self.fets_eval, mats=self.mats_eval)

        L = self.geometry.L / 2.0
        L_c = self.geometry.L_c
        x_x, x_y = dgrid.mesh.geo_grid.point_x_grid
        L_1 = x_x[1, 0]
        d_L = L_c - L_1
        x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
        return dgrid

    fe_grid = Property

    def _get_fe_grid(self):
        return self.dots_grid.mesh

    tline = Instance(TLine)

    def _tline_default(self):
        t_max = 1.0
        d_t = 0.1
        return TLine(min=0.0, step=d_t, max=t_max,
                     time_change_notifier=self.time_changed,
                     )

    alg_stiff = tr.Bool(False,
                        ALG=True,
                        label='use algorithmic stiffness')

    k_max = Int(600,
                label='Maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      label='tolerance',
                      ALG=True)

    tloop = Property(Instance(TimeLoop),
                     depends_on='MAT,GEO,MESH,CS,TIME,ALG,BC')
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_tloop(self):
        k_max = self.k_max
        tolerance = self.tolerance
        return TimeLoop(ts=self.dots_grid, k_max=k_max,
                        tolerance=tolerance,
                        tline=self.tline,
                        algorithmic=self.alg_stiff,
                        bc_mngr=self.bcond_mngr,
                        response_traces=list(self.response_traces.values()))

    response_traces = tr.Dict
    '''Response traces.
    '''

    def _response_traces_default(self):
        return {'energy': Vis2DEnergy(model=self),
                'crack band': Vis2DCrackBand(model=self),
                }

    t = Property

    def _get_t(self):
        return self.get_t()

    def get_t(self):
        return np.array(self.tloop.t_record, dtype=np.float_)

    def get_PW(self):
        record_dofs = np.unique(
            self.fe_grid[self.controlled_elem, -
                         1, :, -1].dofs[:, :, 1].flatten()
        )
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = -np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = -Ud_t[:, record_dofs[0]]
        return F_int_t, U_t

    viz2d_classes = {'F-w': Viz2DForceDeflection,
                     'load function': Viz2DLoadControlFunction,
                     }

    traits_view = View(  # UItem('mats_eval@', resizable=True),
        VGroup(
            Item('w_max', full_size=True, resizable=True),
        ),
        UItem('cross_section@'),
        UItem('geometry@'),
        VGroup(
            Item('n_e_x'),
            Item('n_e_y'),
            Item('k_max'),
            Item('tolerance'),
            Item('alg_stiff'),
            label='Numerical parameters',
        ),
    )

    tree_view = traits_view


def run_bending3pt_sdamage(*args, **kw):
    bt = BendingTestModel(n_e_x=20, n_e_y=30, k_max=500,
                          mats_eval_type='scalar damage'
                          #mats_eval_type='microplane damage (eeq)'
                          #mats_eval_type='microplane CSD (eeq)'
                          #mats_eval_type='microplane CSD (odf)'
                          )
    L_c = 5.0
    E = 30000.0
    f_t = 2.5
    G_f = 0.09
    bt.mats_eval.trait_set(
        stiffness='algorithmic',
        E=E,
        nu=0.2
    )
    bt.mats_eval.omega_fn.trait_set(
        f_t=f_t,
        G_f=G_f,
        L_s=L_c
    )
    # print 'Gf', h_b * bt.mats_eval.get_G_f()

    bt.w_max = 1.0
    bt.tline.step = 0.02
    bt.cross_section.b = 100.
    bt.geometry.trait_set(
        L=450,
        H=110,
        a=10,
        L_c=L_c
    )
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
    bt.add_viz2d('F-w', 'load-displacement')
    vis2d_energy = bt.response_traces['energy']
    vis2d_crack_band = bt.response_traces['crack band']
    viz2d_energy = Viz2DEnergy(name='dissipation',
                               vis2d=vis2d_energy)
    viz2d_energy_rates = Viz2DEnergyReleasePlot(name='dissipated energy',
                                                vis2d=vis2d_energy)
    viz2d_cb_strain = Viz2DStrainInCrack(name='strain in crack',
                                         vis2d=vis2d_crack_band)
    viz2d_cb_a = Viz2DTA(name='crack length',
                         vis2d=vis2d_crack_band,
                         visible=False)
    viz2d_cb_dGda = Viz2DdGdA(name='energy release per crack extension',
                              vis2d=vis2d_energy,
                              vis2d_cb=vis2d_crack_band,
                              visible=False)
    w.viz_sheet.viz2d_list.append(viz2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_cb_strain)
    w.viz_sheet.viz2d_list.append(viz2d_energy_rates)
    w.viz_sheet.viz2d_list.append(viz2d_cb_a)
    w.viz_sheet.viz2d_list.append(viz2d_cb_dGda)
    w.viz_sheet.monitor_chunk_size = 1
    return w


def run_bending3pt_sdamage_viz2d(*args, **kw):
    w = run_bending3pt_sdamage()
    w.run()
    w.offline = True
    w.configure_traits(*args, **kw)


def run_bending3pt_sdamage_viz3d(*args, **kw):
    w = run_bending3pt_sdamage()
#     w.run()
    w.offline = True
    bt = w.model
    vis3d_damage = Vis3DSDamage()
    bt.response_traces['damage'] = vis3d_damage
    viz3d_damage = Viz3DSDamage(vis3d=vis3d_damage)
    w.viz_sheet.add_viz3d(viz3d_damage)
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bending3pt_sdamage_viz3d()

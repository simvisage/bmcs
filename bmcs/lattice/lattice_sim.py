'''
Created on 12.01.2016
@author: ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''

import time

from bmcs.time_functions import \
    LoadingScenario
from ibvpy.api import BCDof, IMATSEval
from simulator.api import \
    Simulator, XDomainLattice
from traits.api import \
    Property, Instance, cached_property, \
    Bool, List, Float, Trait, Int, Enum, \
    Array
from traits.api import \
    on_trait_change, Tuple
from traitsui.api import \
    View, Item, Group
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.ui.bmcs_tree_node import itags_str
from view.window import BMCSWindow

from ibvpy.mats.mats3D_ifc import MATS3DIfcElastic
from ibvpy.mats.viz3d_lattice import Vis3DLattice, Viz3DLattice
import numpy as np
from simulator.xdomain.xdomain_lattice import LatticeTessellation


class LatticeRecord(Vis2D):

    Pw = Tuple()

    def _Pw_default(self):
        return ([0], [0])

    sig_t = List([])
    eps_t = List([])

    def setup(self):
        self.Pw = ([0], [0])
        self.eps_t = []
        self.sig_t = []

    def update(self):
        sim = self.sim
        c_dof = sim.control_dofs
        U_ti = self.sim.hist.U_t
        F_ti = self.sim.hist.F_t
        P = F_ti[:, c_dof]
        w = U_ti[:, c_dof]
        self.Pw = P, w

    def get_t(self):
        return self.sim.hist.t


class Viz2DLatticeFW(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-W'

    show_legend = Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sim = self.vis2d.sim
        P_t, w_t = sim.hist['Pw'].Pw
        ymin, ymax = np.min(P_t), np.max(P_t)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(w_t), np.max(w_t)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        ax.plot(w_t, P_t, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('pull-out force P [N]')
        ax.set_xlabel('pull-out slip w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        sim = self.vis2d.sim
        P_t, w_t = sim.hist['Pw'].Pw
        idx = sim.hist.get_time_idx(vot)
        P, w = P_t[idx], w_t[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
        Item('show_data')
    )


class Viz2DLatticeField(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = Property(depends_on='plot_fn')

    @cached_property
    def _get_label(self):
        return 'field: %s' % self.plot_fn

    plot_fn = Trait('eps_p',
                    {'eps_p': 'plot_eps_p',
                     'sig_p': 'plot_sig_p',
                     'u_p': 'plot_u_p',
                     's': 'plot_s',
                     'sf': 'plot_sf',
                     'omega': 'plot_omega',
                     'Fint_p': 'plot_Fint_p',
                     'eps_f(s)': 'plot_eps_s',
                     },
                    label='Field',
                    tooltip='Select the field to plot'
                    )

    def plot(self, ax, vot, *args, **kw):
        ymin, ymax = getattr(self.vis2d, self.plot_fn_)(ax, vot, *args, **kw)
        if self.adaptive_y_range:
            if self.initial_plot:
                self.y_max = ymax
                self.y_min = ymin
                self.initial_plot = False
                return
        self.y_max = max(ymax, self.y_max)
        self.y_min = min(ymin, self.y_min)
        ax.set_ylim(ymin=self.y_min, ymax=self.y_max)

    y_max = Float(1.0, label='Y-max value',
                  auto_set=False, enter_set=True)
    y_min = Float(0.0, label='Y-min value',
                  auto_set=False, enter_set=True)

    adaptive_y_range = Bool(True)
    initial_plot = Bool(True)

    traits_view = View(
        Item('plot_fn', resizable=True, full_size=True),
        Item('y_min', ),
        Item('y_max', ),
        Item('adaptive_y_range')
    )


class LatticeModelSim(Simulator):

    node_name = 'pull out simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.loading_scenario,
            self.mats_eval,
            self.lattice_tessellation,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.loading_scenario,
            self.mats_eval,
            self.lattice_tessellation,
        ]

    tree_view = View(
        Group(
            Item('mats_eval_type', resizable=True, full_size=True),
            Item('control_variable', resizable=True, full_size=True),
            Item('w_max', resizable=True, full_size=True),
            Group(
                Item('loading_scenario@', show_label=False),
            )
        )
    )

    #=========================================================================
    # Test setup parameters
    #=========================================================================
    loading_scenario = Instance(
        LoadingScenario,
        report=True,
        desc='object defining the loading scenario'
    )

    def _loading_scenario_default(self):
        return LoadingScenario()

    lattice_tessellation = Instance(
        LatticeTessellation,
        MESH=True,
        report=True,
        desc='cross section parameters'
    )

    def _lattice_tessellation_default(self):
        return LatticeTessellation()

    control_variable = Enum('u', 'f',
                            auto_set=False, enter_set=True,
                            BC=True)

    #=========================================================================
    # Algorithimc parameters
    #=========================================================================
    k_max = Int(400,
                unit='-',
                symbol='k_{\max}',
                desc='maximum number of iterations',
                ALG=True)

    tolerance = Float(1e-4,
                      unit='-',
                      symbol='\epsilon',
                      desc='required accuracy',
                      ALG=True)

    mats_eval_type = Trait('mats3d_ifc_elastic',
                           {'mats3d_ifc_elastic': MATS3DIfcElastic,
                            'mats3d_ifc_cumslide': MATS3DIfcElastic,
                            },
                           MAT=True,
                           desc='material model type')

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()
        self._update_node_list()

    mats_eval = Instance(IMATSEval, report=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    dots_lattice = Property(Instance(XDomainLattice),
                            depends_on=itags_str)
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_lattice(self):
        print('reconstruct DOTS')
        return XDomainLattice(
            mesh=self.lattice_tessellation
        )

    domains = Property(depends_on=itags_str)

    @cached_property
    def _get_domains(self):
        print('reconstruct DOMAIN')
        return [(self.dots_lattice, self.mats_eval)]

    #=========================================================================
    # Boundary conditions
    #=========================================================================
    w_max = Float(1, BC=True,
                  symbol='w_{\max}',
                  unit='mm',
                  desc='maximum pullout slip',
                  auto_set=False, enter_set=True)

    fixed_dofs = Array(np.int_, value=[], BC=True)

    fixed_bc_list = Property(depends_on=itags_str)
    r'''Foxed boundary condition'''
    @cached_property
    def _get_fixed_bc_list(self):
        return [
            BCDof(node_name='fixed dof %d' % dof, var='u',
                  dof=dof, value=0.0) for dof in self.fixed_dofs
        ]

    control_dofs = Array(np.int_, value=[], BC=True)

    control_bc_list = Property(depends_on=itags_str)
    r'''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc_list(self):
        return [
            BCDof(node_name='control dof %d' % dof,
                  var=self.control_variable,
                  dof=dof, value=self.w_max,
                  time_function=self.loading_scenario)
            for dof in self.control_dofs
        ]

    bc = Property(depends_on=itags_str)

    @cached_property
    def _get_bc(self):
        return self.control_bc_list + self.fixed_bc_list

    def get_window(self):
        self.record['Pw'] = LatticeRecord()
        self.record['eps'] = Vis3DLattice(var='eps')
        w = BMCSWindow(sim=self)
        fw = Viz2DLatticeFW(name='FW-curve', vis2d=self.hist['Pw'])
        w.viz_sheet.viz2d_list.append(fw)
        viz3d_u_Lb = Viz3DLattice(vis3d=self.hist['eps'])
        w.viz_sheet.add_viz3d(viz3d_u_Lb)
        w.viz_sheet.monitor_chunk_size = 1
        return w


def test01_couple():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
        ], dtype=np.float_
    )
    I_Li = np.array(
        [
            [0, 1],
        ], dtype=np.int_
    )
    fixed_dofs = [0, 1, 2, 3, 4, 5,
                  9, 10, 11]
    control_dofs = [7]
    w_max = 1
    return X_Ia, I_Li, fixed_dofs, control_dofs, w_max


def test02_penta():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0]
        ], dtype=np.float_
    )
    I_Li = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
        ], dtype=np.int_
    )
    fixed_dofs = [3, 4, 9, 10, 15, 16,
                  #         2, 3, 4,
                  #         8, 9, 10,
                  #         14, 15, 16,
                  18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29
                  ]

    control_dofs = [7]  # , 13]
    w_max = -0.1
    return X_Ia, I_Li, fixed_dofs, control_dofs, w_max


def test03_tetrahedron():
    X_Ia = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.3, 0.3, 1.0]
        ], dtype=np.float_
    )
    I_Li = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3]
        ], dtype=np.int_
    )
    fixed_dofs = [0, 1, 2, 3, 4, 5,
                  7, 8, 9, 10, 11,
                  12, 14, 15, 16, 17,
                  21, 23]
    control_dofs = [22]
    w_max = 0.1 * np.pi
    return X_Ia, I_Li, fixed_dofs, control_dofs, w_max


if __name__ == '__main__':
    X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test03_tetrahedron()
    #X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test01_couple()
    tes = LatticeTessellation(
        X_Ia=X_Ia,
        I_Li=I_Li
    )
    sim = LatticeModelSim(
        lattice_tessellation=tes,
        fixed_dofs=fixed_dofs,
        control_dofs=control_dofs,
        w_max=w_max
    )
    sim.tstep.init_state()
    sim.tloop.k_max = 3
    sim.tline.step = 1.0
    sim.tloop.verbose = True
    sim.tstep.debug = True
    w = sim.get_window()
    time.sleep(1)
    w.configure_traits()
    print(sim.hist.F_t[-1])
    print(sim.hist.U_t[-1])

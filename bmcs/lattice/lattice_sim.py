'''
Created on 12.01.2016
@author: ABaktheer, RChudoba

@todo: enable recalculation after the initial offline run
@todo: reset viz adapters upon recalculation to forget their axes lims
@todo: introduce a switch for left and right supports
'''

import os
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

from ibvpy.mats.mats3D_ifc import MATS3DIfcElastic, MATS3DIfcCumSlip
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
        P = np.sum(F_ti[:, c_dof], axis=-1)
        w = np.average(U_ti[:, c_dof], axis=-1)
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
                            'mats3d_ifc_cumslip': MATS3DIfcCumSlip,
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
        w.viz_sheet.monitor_chunk_size = 10
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


def test04_mgrid(dimensions=(1, 1, 5), shape=(10, 5, 5)):
    Lx, Ly, Lz = dimensions
    nx, ny, nz = shape
    nx_, ny_, nz_ = complex(nx), complex(ny), complex(nz)
    x, y, z = np.mgrid[:Lx:nx_, :Ly:ny_, :Lz:nz_]
    X_Ia = np.einsum('aI->Ia',
                     np.array([x.flatten(), y.flatten(), z.flatten()],
                              dtype=np.float_)
                     )
    I = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    Ix_Li = np.array([I[:-1, :, :].flatten(), I[1:, :, :].flatten()])
    Iy_Li = np.array([I[:, :-1, :].flatten(), I[:, 1:, :].flatten()])
    Iz_Li = np.array([I[:, :, :-1].flatten(), I[:, :, 1:].flatten()])
    I_Li = np.vstack([Ix_Li.T, Iy_Li.T, Iz_Li.T])
    print(I)
    bot_nodes = I[:, :, 0].flatten()
    top_nodes = I[:, :, -1].flatten()
    print('bot_nodes', bot_nodes)
    print('top_nodes', top_nodes)

    fix_rot = I.flatten()[:, np.newaxis] * 6 + np.array([3, 4, 5], np.int_)
    fix_top = top_nodes[:, np.newaxis] * 6 + np.array([1, 2], np.int_)
    fix_bot = bot_nodes[:, np.newaxis] * 6 + np.array([0, 1, 2], np.int_)
    print('fix_top', fix_top)
    print('fix_bot', fix_bot)
    fixed_dofs = np.hstack(
        [fix_bot.flatten(), fix_top.flatten(), fix_rot.flatten()]
    )

    control_dofs = top_nodes[:, np.newaxis] * 6 + np.array([0], np.int_)
    print('control_top', control_dofs)
    w_max = 0.1

    np.savez('myfile.npz', X_Ia=X_Ia, I_Li=I_Li, fixed_dofs=fixed_dofs,
             control_dofs=control_dofs)
    npzfile = np.load('myfile.npz')
    X_Ia = npzfile['X_Ia']

    return X_Ia, I_Li, fixed_dofs.flatten(), control_dofs.flatten(), w_max


def test05_lattice():
    home_dir = os.path.expanduser('~')
    ldir = os.path.join(home_dir, 'simdb', 'simdata', 'lattice_example')
    X_Ia = np.loadtxt(os.path.join(ldir, 'nodes.inp'),
                      skiprows=1, usecols=(1, 2, 3))
    vertices = np.loadtxt(os.path.join(ldir, 'vertices.inp'),
                          skiprows=1, usecols=(1, 2, 3))
    I_Li = np.loadtxt(os.path.join(ldir, 'mechElems.inp'),
                      skiprows=1, usecols=(1, 2), dtype=np.int_)
    fixed_dofs = np.arange(20, dtype=np.int_)
    n_nodes = len(X_Ia)
    control_dofs = np.arange(n_nodes - 20, n_nodes - 1, dtype=np.int_)
    return X_Ia, I_Li, fixed_dofs.flatten(), control_dofs.flatten(), -0.01


def run_elastic():
    X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test05_lattice()
    # print(X_Ia.shape)
    # print(I_Li.shape)
#     X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test04_mgrid(
#         shape=(1, 1, 2), dimensions=(1, 1, 1))
    #X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test03_tetrahedron()
    #X_Ia, I_Li, fixed_dofs, control_dofs, w_max = test01_couple()
    tes = LatticeTessellation(
        X_Ia=X_Ia,
        I_Li=I_Li
    )
    global sim
    sim = LatticeModelSim(
        lattice_tessellation=tes,
        mats_eval_type='mats3d_ifc_elastic',
        fixed_dofs=fixed_dofs,
        control_dofs=control_dofs,
        w_max=w_max
    )
#     sim.mats_eval.trait_set(E_T=1, E_N=1, gamma_T=0, K_T=0,
#                             tau_pi_bar=0.05)
#     sim.mats_eval.configure_traits()
    sim.mats_eval.trait_set(E_n=1, E_s=1)
    sim.tstep.init_state()
    sim.tloop.k_max = 200
    sim.tline.step = 0.1
    sim.tloop.verbose = True
    sim.tstep.debug = False
    import cProfile
#    cProfile.run("sim.run()")
#     w = sim.get_window()
#     time.sleep(1)
#     w.run()
#     w.configure_traits()


if __name__ == '__main__':
    run_elastic()

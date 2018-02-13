'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    IMATSEval, TLine, BCSlice
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow

from ibvpy.core.vtloop import TimeLoop
from ibvpy.dots.vdots_grid import DOTSGrid
import numpy as np
import traits.api as tr


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
        ax.plot(W, P, linewidth=2, color='black', alpha=0.4,
                label='P(w;x=L)')
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


class CrossSection(BMCSLeafNode):
    '''Parameters of the pull-out cross section
    '''
    node_name = 'cross-section'

    b = Float(50.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')
    h = Float(100.0,
              CS=True,
              auto_set=False, enter_set=True,
              desc='cross section height [mm2]')

    view = View(
        Item('h'),
        Item('b'),
    )

    tree_view = view


class Geometry(BMCSLeafNode):

    node_name = 'geometry'
    L = Float(600.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')

    view = View(
        Item('L_x'),
    )

    tree_view = view


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
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
            self.fixed_left_bc,
            self.fixed_right_bc,
            self.control_bc
        ]

    #=========================================================================
    # Interactive control of the time loop
    #=========================================================================
    def init(self):
        self.tloop.init()

    def eval(self):
        return self.tloop.eval()

    def pause(self):
        self.tloop.paused = True

    def stop(self):
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
    n_e_x = Int(20, auto_set=False, enter_set=True)
    n_e_y = Int(8, auto_set=False, enter_set=True)

    w_max = Float(-50, BC=True, auto_set=False, enter_set=True)

    free_end_dof = Property

    def _get_free_end_dof(self):
        return self.n_e_x + 1

    controlled_dof = Property

    def _get_controlled_dof(self):
        return 2 + 2 * self.n_e_x - 1

    fixed_dof = Property

    def _get_fixed_dof(self):
        return 0

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('microplane damage (eeg)',
                           {'elastic': MATS2DElastic,
                            'microplane damage (eeq)': MATS2DMplDamageEEQ,
                            'scalar damage': MATS2DScalarDamage
                            },
                           MAT=True
                           )

    @on_trait_change('mats_eval_type')
    def _set_mats_eval(self):
        self.mats_eval = self.mats_eval_type_()

    @on_trait_change('BC,MAT,MESH')
    def reset_node_list(self):
        self._update_node_list()

    mats_eval = Instance(IMATSEval,
                         MAT=True)
    '''Material model'''

    def _mats_eval_default(self):
        return self.mats_eval_type_()

    material = Property

    def _get_material(self):
        return self.mats_eval

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS2D4Q),
                         depends_on='CS,MAT')
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS2D4Q(h=self.cross_section.h,
                        b=self.cross_section.b)

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_left_bc,
                   self.fixed_right_bc,
                   self.fixed_middle_bc,
                   self.control_bc]
        return BCondMngr(bcond_list=bc_list)

    fixed_left_bc = Property(depends_on='BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_bc(self):
        return BCSlice(slice=self.fe_grid[0, 0, 0, 0],
                       var='u', dims=[1], value=0)

    fixed_right_bc = Property(depends_on='BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_right_bc(self):
        return BCSlice(slice=self.fe_grid[-1, 0, -1, 0],
                       var='u', dims=[1], value=0)

    fixed_middle_bc = Property(depends_on='BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_middle_bc(self):
        return BCSlice(slice=self.fe_grid[self.n_e_x / 2, -1, :, -1],
                       var='u', dims=[0], value=0)

    control_bc = Property(depends_on='BC,GEO,MESH')
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(slice=self.fe_grid[self.n_e_x / 2, -1, :, -1],
                       var='u', dims=[1], value=-self.w_max)

    dots_grid = Property(Instance(DOTSGrid), depends_on='MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        return DOTSGrid(L_x=self.geometry.L, L_y=self.cross_section.h,
                        n_x=self.n_e_x, n_y=self.n_e_y,
                        fets=self.fets_eval, mats=self.mats_eval)

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

    k_max = Int(200,
                ALG=True)
    tolerance = Float(1e-4,
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
                        bc_mngr=self.bcond_mngr)

    def get_PW(self):
        record_dofs = self.fe_grid[self.n_e_x /
                                   2, -1, :, -1].dofs[:, :, 1].flatten()
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = -np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = -Ud_t[:, record_dofs[0]]
        t_arr = np.array(self.tloop.t_record, dtype=np.float_)
        return F_int_t, U_t

    viz2d_classes = {'F-w': Viz2DForceDeflection,
                     'load function': Viz2DLoadControlFunction,
                     }

    traits_view = View(Item('mats_eval_type'),)

    tree_view = traits_view


from view.plot3d.viz3d_poll import Vis3DPoll, Viz3DPoll


def run_bending3pt_elastic():
    bt = BendingTestModel(n_e_x=30, k_max=500,
                          mats_eval_type='scalar damage'
                          #mats_eval_type='microplane damage (eeq)'
                          )
    bt.mats_eval.set(
        stiffness='algorithmic',
        epsilon_0=0.005,
        epsilon_f=1. * 1000
    )

    bt.w_max = 10
    bt.tline.step = 0.05
    bt.geometry.L = 600
    bt.loading_scenario.set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
    #bt.add_viz2d('load function', 'load-time')
    bt.add_viz2d('F-w', 'load-displacement')

    vis3d = Vis3DPoll()
    bt.tloop.response_traces.append(vis3d)
    viz3d = Viz3DPoll(vis3d=vis3d)
    w.viz_sheet.add_viz3d(viz3d)

#    w.run()
    w.offline = True
    w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_bending3pt_elastic()
    # run_with_new_state()

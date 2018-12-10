'''
Created on May 15, 2018

Created on 12.01.2016
@author: RChudoba, ABaktheer

'''

from bmcs.time_functions import \
    LoadingScenario, Viz2DLoadControlFunction
from ibvpy.api import \
    IMATSEval, TLine, BCSlice, BCDof
from ibvpy.core.bcond_mngr import BCondMngr
from ibvpy.core.vtloop import TimeLoop
from ibvpy.dots.vdots_grid import DOTSGrid
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.mats.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage, MATS2DMplCSDEEQ
from ibvpy.mats.mats3D.mats3D_sdamage.viz3d_sdamage import \
    Vis3DSDamage, Viz3DSDamage
from traits.api import \
    Property, Instance, cached_property, \
    List, Float, Trait, Int, on_trait_change
from traitsui.api import \
    View, Item
from view.plot2d import Viz2D, Vis2D
from view.ui import BMCSLeafNode
from view.window import BMCSModel, BMCSWindow
from ibvpy.mats.mats3D.viz3d_strain_field import \
    Vis3DStrainField, Viz3DStrainField
from ibvpy.mats.mats3D.viz3d_stress_field import \
    Vis3DStressField, Viz3DStressField
import numpy as np
import traits.api as tr
from viz3d_energy import Viz2DEnergy, Vis2DEnergy, Viz2DEnergyReleasePlot


class Viz2DForceDeflectionX(Viz2D):

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
    L = Float(20.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')
    a = Float(20.0,
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')

    view = View(
        Item('L'),
        Item('a'),
    )

    tree_view = view


class DCBTestModel(BMCSModel, Vis2D):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'double cantilever beam simulation'

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            self.mats_eval,
            self.cross_section,
            self.geometry,
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
    n_e_x = Int(2, auto_set=False, enter_set=True)
    n_e_y = Int(8, auto_set=False, enter_set=True)

    w_max = Float(0.01, BC=True, auto_set=False, enter_set=True)

    #=========================================================================
    # Material model
    #=========================================================================
    mats_eval_type = Trait('scalar damage',
                           {'elastic': MATS2DElastic,
                            'scalar damage': MATS2DScalarDamage,
                            'microplane damage (eeq)': MATS2DMplDamageEEQ,
                            #                            'microplane damage (odf)': MATS2DMplDamageODF,
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
        return FETS2D4Q()

    bcond_mngr = Property(Instance(BCondMngr),
                          depends_on='CS,BC,MESH')
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bcond_mngr(self):
        bc_list = [self.fixed_left_x,
                   self.fixed_top_y,
                   self.control_bc,
                   ] + self.link_right_x

        return BCondMngr(bcond_list=bc_list)

    fixed_left_x = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_left_x(self):
        a_L = self.geometry.a / self.geometry.L
        n_a = int(a_L * self.n_e_y)
        return BCSlice(slice=self.fe_grid[0, n_a:, 0, -1],
                       var='u', dims=[0], value=0)

    fixed_top_y = Property(depends_on='CS, BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_top_y(self):
        return BCSlice(slice=self.fe_grid[:, -1, :, -1],
                       var='u', dims=[1], value=0)

    link_right_x = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_link_right_x(self):
        top = self.fe_grid[-1, -1, -1, -1]
        bot = self.fe_grid[-1, 0, -1, 0]
        linked = self.fe_grid[-1, 1:, -1, 0]

        print('SHAPE', top.dof_X.shape)
        Ty = top.dof_X[0, 0, 1]
        By = bot.dof_X[0, 0, 1]

        Ly = linked.dof_X[:, :, 1].flatten()

        H = Ty - By
        link_ratios = Ly / H
        top_dof = top.dofs[0, 0, 0]
        bot_dof = bot.dofs[0, 0, 0]
        linked_dofs = linked.dofs[:, :, 0].flatten()
        bcdof_list = []
        for linked_dof, link_ratio in zip(linked_dofs, link_ratios):
            link_bc = BCDof(var='u',
                            dof=linked_dof,
                            value=0,
                            link_dofs=[bot_dof, top_dof],
                            link_coeffs=[1 - link_ratio, link_ratio]
                            )
            bcdof_list.append(link_bc)
        return bcdof_list

    control_bc = Property(depends_on='CS,BC,GEO,MESH')
    '''Foxed boundary condition'''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(
            #slice=self.fe_grid[0, 0, :, 0, :, :],
            slice=self.fe_grid[-1, 0, -1, 0],
            var='u', dims=[0], value=self.w_max
        )

    dots_grid = Property(Instance(DOTSGrid),
                         depends_on='CS,MAT,GEO,MESH,FE')
    '''Discretization object.
    '''
    @cached_property
    def _get_dots_grid(self):
        cs = self.cross_section
        geo = self.geometry
        print(self.mats_eval)
        return DOTSGrid(
            L_x=cs.h, L_y=geo.L,
            n_x=self.n_e_x, n_y=self.n_e_y,
            fets=self.fets_eval, mats=self.mats_eval
        )

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
        record_dofs = np.unique(
            self.fe_grid[-1, :, -1, :].dofs[:, :, 0].flatten()
        )
        Fd_int_t = np.array(self.tloop.F_int_record)
        Ud_t = np.array(self.tloop.U_record)
        F_int_t = np.sum(Fd_int_t[:, record_dofs], axis=1)
        U_t = Ud_t[:, record_dofs[0]]
        return F_int_t, U_t

    viz2d_classes = {'F-w': Viz2DForceDeflectionX,
                     'load function': Viz2DLoadControlFunction,
                     }

    traits_view = View(Item('mats_eval_type'),)

    tree_view = traits_view


def run_dcb_2d(*args, **kw):

    bt = DCBTestModel(n_e_x=2, n_e_y=30,
                      k_max=1200,
                      mats_eval_type='scalar damage'
                      #mats_eval_type='microplane damage (odf)'
                      )
    bt.mats_eval.trait_set(
        stiffness='algorithmic',
        E=30000.0,
        nu=0.2
    )
    bt.mats_eval.omega_fn.trait_set(
        f_t=3.0,
        G_f=0.004
    )

    bt.w_max = 0.1
    bt.tline.step = 0.1
    bt.cross_section.h = 200
    bt.cross_section.b = 50
    bt.geometry.L = 6000
    bt.geometry.a = 300
    bt.loading_scenario.trait_set(loading_type='monotonic')
    w = BMCSWindow(model=bt)
    bt.add_viz2d('load function', 'load-time')
    bt.add_viz2d('F-w', 'load-displacement')

    vis2d_energy = Vis2DEnergy(model=bt)
    viz2d_energy = Viz2DEnergy(name='dissipation', vis2d=vis2d_energy)
    viz2d_energy_rates = Viz2DEnergyReleasePlot(
        name='dissipation rate', vis2d=vis2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_energy)
    w.viz_sheet.viz2d_list.append(viz2d_energy_rates)

    vis3d = Vis3DStressField()
    bt.tloop.response_traces.append(vis3d)
    bt.tloop.response_traces.append(vis2d_energy)
    viz3d = Viz3DStressField(vis3d=vis3d, warp_factor=1000.0)
    w.viz_sheet.add_viz3d(viz3d)
    w.viz_sheet.monitor_chunk_size = 1

#     w.run()
#     w.offline = True
#     w.finish_event = True
    w.configure_traits()


if __name__ == '__main__':
    run_dcb_2d()
